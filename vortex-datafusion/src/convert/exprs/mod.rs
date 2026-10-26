// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_common::format::DEFAULT_FORMAT_OPTIONS;
use datafusion_expr::Operator as DFOperator;
use datafusion_expr::type_coercion::binary::BinaryTypeCoercer;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_functions::string::octet_length::OctetLengthFunc;
use datafusion_functions_nested::array_has::ArrayHas;
use datafusion_functions_nested::length::ArrayLength;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::expr::Expression;
use vortex::expr::byte_length;
use vortex::expr::cast;
use vortex::expr::get_item;
use vortex::expr::in_list_of;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::list_length;
use vortex::expr::lit;
use vortex::expr::nested_case_when;
use vortex::expr::not;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::transform::replace;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use crate::convert::scalar_from_df;

#[cfg(test)]
mod tests;

/// Result of splitting a projection into Vortex expressions and leftover DataFusion projections.
pub struct ProcessedProjection {
    /// Projection evaluated by the Vortex scan.
    pub scan_projection: Expression,
    /// Arrow reference types and metadata for the scan output.
    pub scan_reference_schema: Schema,
    /// Projection evaluated by DataFusion after the Vortex scan.
    pub leftover_projection: ProjectionExprs,
}

/// Trait for converting DataFusion expressions to Vortex ones.
///
/// Custom converters implement a single schema-aware decision. Conversion should preserve
/// DataFusion values, nulls, and evaluation errors; see [`DefaultExpressionConverter`] for
/// the arithmetic and nested field access exceptions. Unsupported expressions remain in
/// DataFusion, including when a file's schema adapter introduces them.
///
/// # Implementing a custom converter
///
/// ```
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use datafusion_common::Result as DFResult;
/// use datafusion_physical_expr::PhysicalExpr;
/// use vortex::expr::Expression;
/// use vortex_datafusion::convert::DefaultExpressionConverter;
/// use vortex_datafusion::convert::ExpressionConverter;
///
/// struct CustomExpressionConverter(DefaultExpressionConverter);
///
/// impl ExpressionConverter for CustomExpressionConverter {
///     fn try_convert(
///         &self,
///         expr: &Arc<dyn PhysicalExpr>,
///         schema: &Schema,
///     ) -> DFResult<Option<Expression>> {
///         self.0.try_convert(expr, schema)
///     }
/// }
///
/// let _converter: Arc<dyn ExpressionConverter> = Arc::new(CustomExpressionConverter(
///     DefaultExpressionConverter::default(),
/// ));
/// ```
pub trait ExpressionConverter: Send + Sync {
    /// Convert an expression for native evaluation against this schema.
    ///
    /// Returns None for valid but unsupported expressions. Malformed expressions and
    /// conversion failures return errors. Callers must retain unsupported exact predicates
    /// for DataFusion evaluation.
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>>;

    /// Convert `expr`, a subexpression of an expression being converted in `scope`.
    ///
    /// [`DefaultExpressionConverter::try_convert_with`] converts children through this. The
    /// default converts with [`Self::try_convert`], which checks `expr` as a whole expression
    /// again. Converters that delegate should override it, handling their own expressions and
    /// passing the rest to [`DefaultExpressionConverter::convert_subexpression_with`].
    fn convert_subexpression(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        scope: &ConversionScope<'_>,
    ) -> DFResult<Option<ConvertedExpression>> {
        let Some(expression) = self.try_convert(expr, scope.schema())? else {
            return Ok(None);
        };
        Ok(ConvertedExpression::new(
            expression,
            expr.data_type(scope.schema())?,
            scope,
        ))
    }

    /// Split a projection into Vortex expressions that can be pushed down and leftover
    /// DataFusion projections that need to be evaluated after the scan.
    ///
    /// If any expression is unsupported, evaluate the complete projection in DataFusion
    /// over deduplicated raw inputs. This avoids mixing input names with output aliases.
    fn split_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
        output_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        // Duplicate output names cannot identify native pack fields unambiguously.
        if !source_projection
            .iter()
            .map(|projection| &projection.alias)
            .all_unique()
        {
            return self.no_pushdown_projection(source_projection, input_schema);
        }
        let mut scan_projection = Vec::with_capacity(source_projection.as_ref().len());
        for projection in source_projection.as_ref() {
            let Some(expr) = self.try_convert(&projection.expr, input_schema)? else {
                return self.no_pushdown_projection(source_projection, input_schema);
            };
            scan_projection.push((projection.alias.clone(), expr));
        }
        // The output schema names its fields after the projection aliases.
        let output_indices = (0..scan_projection.len()).collect_vec();
        Ok(ProcessedProjection {
            scan_projection: pack(scan_projection, Nullability::NonNullable),
            scan_reference_schema: output_schema.clone(),
            leftover_projection: ProjectionExprs::from_indices(&output_indices, output_schema),
        })
    }

    /// Create a projection that reads only the required columns without pushing down
    /// any expressions. All projection logic is applied after the scan.
    fn no_pushdown_projection(
        &self,
        source_projection: ProjectionExprs,
        input_schema: &Schema,
    ) -> DFResult<ProcessedProjection> {
        let (scan_projection, scan_reference_schema) =
            raw_projection(&source_projection.column_indices(), input_schema)?;
        Ok(ProcessedProjection {
            scan_projection,
            scan_reference_schema,
            leftover_projection: source_projection,
        })
    }
}

/// Read the raw columns at `indices` in file order, without involving custom expression
/// conversion. Returns the scan projection and the Arrow schema of its output.
pub(crate) fn raw_projection(
    indices: &[usize],
    input_schema: &Schema,
) -> DFResult<(Expression, Schema)> {
    let schema = input_schema.project(indices)?;
    let scan_columns = schema.fields().iter().map(|field| {
        (
            field.name().clone(),
            get_item(field.name().as_str(), root()),
        )
    });
    Ok((pack(scan_columns, Nullability::NonNullable), schema))
}

/// Why an expression was not converted.
enum Unconverted {
    /// A valid expression that Vortex cannot evaluate; it stays in DataFusion.
    Unsupported,
    /// A malformed expression or a failed conversion.
    Failed(DataFusionError),
}

impl From<DataFusionError> for Unconverted {
    fn from(error: DataFusionError) -> Self {
        Self::Failed(error)
    }
}

/// Conversion result where `?` propagates both unsupported expressions and errors.
type Conversion<T> = Result<T, Unconverted>;

/// Where a subexpression is converted: the Arrow schema DataFusion types it against, and the
/// Vortex dtype the converted expression reads, which holds the columns of the whole
/// expression being converted.
#[derive(Clone, Copy)]
pub struct ConversionScope<'a> {
    schema: &'a Schema,
    input_dtype: &'a DType,
}

impl<'a> ConversionScope<'a> {
    fn new(schema: &'a Schema, input_dtype: &'a DType) -> Self {
        Self {
            schema,
            input_dtype,
        }
    }

    /// The Arrow schema the DataFusion expression is typed against.
    pub fn schema(&self) -> &'a Schema {
        self.schema
    }

    /// The Vortex dtype the converted expression is evaluated against.
    pub fn input_dtype(&self) -> &'a DType {
        self.input_dtype
    }
}

/// A converted subexpression and the types of its result.
///
/// The default converter tracks these from the children it converts, so that its checks do
/// not walk the subexpression again at every node.
pub struct ConvertedExpression {
    expression: Expression,
    /// The Vortex dtype the expression returns.
    dtype: DType,
    /// The Arrow type the DataFusion expression returns.
    data_type: DataType,
    /// Whether evaluating the expression cannot fail; see [`is_infallible`].
    infallible: bool,
}

impl ConvertedExpression {
    /// Describe `expression`, the conversion of a DataFusion expression that returns
    /// `data_type`, by evaluating its types in `scope`. This walks `expression`, so custom
    /// converters use it for their own, typically small, expressions.
    ///
    /// Returns `None` when Vortex cannot type the expression in `scope`.
    pub fn new(
        expression: Expression,
        data_type: DataType,
        scope: &ConversionScope<'_>,
    ) -> Option<Self> {
        let dtype = expression.return_dtype(scope.input_dtype).ok()?;
        let infallible = is_infallible(&expression);
        Some(Self {
            expression,
            dtype,
            data_type,
            infallible,
        })
    }

    /// The converted Vortex expression.
    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// The Vortex dtype the expression returns.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// The converted Vortex expression, consuming the description.
    pub fn into_expression(self) -> Expression {
        self.expression
    }

    /// Describe a converted subexpression by walking it, e.g. a column or a literal without
    /// converted children; unresolvable dtypes are unsupported.
    fn describe(
        expression: Expression,
        data_type: DataType,
        scope: &ConversionScope<'_>,
    ) -> Conversion<Self> {
        Self::new(expression, data_type, scope).ok_or(Unconverted::Unsupported)
    }

    /// Describe `expression`, a scalar function whose children are exactly the converted
    /// `children`, in order, from their tracked types.
    fn node(
        expression: Expression,
        data_type: DataType,
        children: &[&ConvertedExpression],
    ) -> Conversion<Self> {
        let scalar_fn = expression
            .as_scalar()
            .ok_or_else(|| exec_datafusion_err!("Converted node {expression} has no function"))?;
        let child_dtypes = children
            .iter()
            .map(|child| child.dtype.clone())
            .collect::<Vec<_>>();
        let dtype = scalar_fn
            .return_dtype(&child_dtypes)
            .map_err(|_| Unconverted::Unsupported)?;
        let infallible =
            scalar_fn.signature().is_infallible() && children.iter().all(|child| child.infallible);
        Ok(Self {
            expression,
            dtype,
            data_type,
            infallible,
        })
    }

    /// The same expression, typed as DataFusion's `data_type`, e.g. for a dropped cast.
    fn with_data_type(self, data_type: DataType) -> Self {
        Self { data_type, ..self }
    }
}

/// The default [`ExpressionConverter`] implementation.
///
/// Supported arithmetic is pushed down using Vortex semantics, including its checked
/// integer arithmetic. Matching DataFusion's overflow behavior is deferred to a future patch.
/// Nested struct field access also uses Vortex semantics: a null parent produces a null child,
/// whereas DataFusion extracts the child without applying parent validity. These differences
/// are accepted for both filters and projections.
/// Searched `CASE WHEN` expressions preserve short-circuit evaluation: conditions and
/// branch values only execute on the rows that reach them, including fallible branches.
/// Other expressions require compatible SQL semantics or remain in DataFusion.
pub struct DefaultExpressionConverter {
    /// Session used to resolve Arrow → Vortex dtypes through the extension
    /// plugin registry, so registered extension types (e.g. UUID ⇄
    /// `FixedSizeBinary[16]`) convert correctly instead of hitting the static,
    /// non-plugin-aware `DType::from_arrow`.
    session: VortexSession,
}

impl Default for DefaultExpressionConverter {
    fn default() -> Self {
        Self {
            session: VortexSession::default(),
        }
    }
}

impl DefaultExpressionConverter {
    /// Create a converter that resolves Arrow extension types using `session`'s
    /// dtype registry.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Convert `expr` with the built-in rules, converting every child expression through
    /// `delegate`.
    ///
    /// Custom converters use this to support their own expressions (e.g. UDFs) anywhere in an
    /// expression tree: implement [`ExpressionConverter::try_convert`] by handling the custom
    /// expressions and passing everything else here with `self` as the delegate. Children go
    /// through the delegate's [`ExpressionConverter::convert_subexpression`]. Override that too,
    /// handling the custom expressions and passing everything else to
    /// [`Self::convert_subexpression_with`], so that conversion stays linear in the size of the
    /// expression.
    pub fn try_convert_with(
        &self,
        delegate: &dyn ExpressionConverter,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        self.try_convert_inner(Some(delegate), expr, schema)
    }

    /// Convert `expr`, a subexpression of an expression being converted in `scope`, with the
    /// built-in rules, converting every child expression through `delegate`.
    pub fn convert_subexpression_with(
        &self,
        delegate: &dyn ExpressionConverter,
        expr: &Arc<dyn PhysicalExpr>,
        scope: &ConversionScope<'_>,
    ) -> DFResult<Option<ConvertedExpression>> {
        finish(self.convert_node(Some(delegate), expr, scope))
    }

    fn try_convert_inner(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        // We currently do not support pushdown of dynamic expressions in DF.
        // See issue: https://github.com/vortex-data/vortex/issues/4034
        if DynamicFilterTracking::classify(expr).contains_dynamic_filter() {
            return Ok(None);
        }
        finish(self.convert_checked(delegate, expr, schema))
            .map(|converted| converted.map(ConvertedExpression::into_expression))
    }

    /// Convert a child expression: through `delegate` when there is one, otherwise with the
    /// built-in rules.
    fn convert_child(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        child: &Arc<dyn PhysicalExpr>,
        scope: &ConversionScope<'_>,
    ) -> Conversion<ConvertedExpression> {
        match delegate {
            Some(delegate) => delegate
                .convert_subexpression(child, scope)?
                .ok_or(Unconverted::Unsupported),
            None => self.convert_node(None, child, scope),
        }
    }

    /// Resolve an Arrow field through the session registry; unknown types are unsupported.
    fn arrow_dtype(&self, field: &Field) -> Conversion<DType> {
        self.session
            .arrow()
            .from_arrow_field(field)
            .map_err(|_| Unconverted::Unsupported)
    }

    /// Convert `expr` and check that it returns the dtype DataFusion expects.
    ///
    /// The schema checks and the input dtype apply to the whole expression, so only the root
    /// computes them. Every node tracks its own dtypes as it converts.
    fn convert_checked(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Conversion<ConvertedExpression> {
        let columns = collect_columns(expr);
        let mut column_indices = Vec::with_capacity(columns.len());
        for column in columns {
            let field = schema.fields().get(column.index()).ok_or_else(|| {
                exec_datafusion_err!(
                    "Column {}@{} is out of bounds",
                    column.name(),
                    column.index()
                )
            })?;
            if field.name() != column.name() {
                return Err(exec_datafusion_err!(
                    "Column {}@{} refers to field {}",
                    column.name(),
                    column.index(),
                    field.name()
                )
                .into());
            }
            column_indices.push(column.index());
        }
        column_indices.sort_unstable();
        column_indices.dedup();
        let input_dtype = self
            .session
            .arrow()
            .from_arrow_schema(
                &schema
                    .project(&column_indices)
                    .map_err(DataFusionError::from)?,
            )
            .map_err(|_| Unconverted::Unsupported)?;
        let scope = ConversionScope::new(schema, &input_dtype);
        // The root uses the built-in rules: a delegate handles its own expressions before it
        // passes the root here.
        let converted = self.convert_node(delegate, expr, &scope)?;
        let expected_dtype = self.arrow_dtype(&expected_field(expr, &converted, schema)?)?;
        if !converted.dtype.eq_ignore_nullability(&expected_dtype) {
            return Err(Unconverted::Unsupported);
        }
        Ok(converted)
    }

    fn convert_node(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        expr: &Arc<dyn PhysicalExpr>,
        scope: &ConversionScope<'_>,
    ) -> Conversion<ConvertedExpression> {
        let schema = scope.schema;
        if let Some(binary_expr) = expr.downcast_ref::<df_expr::BinaryExpr>() {
            let operator =
                try_operator_from_df(binary_expr.op()).ok_or(Unconverted::Unsupported)?;
            let boolean_operator = matches!(operator, Operator::And | Operator::Or);
            let left = self.convert_child(delegate, binary_expr.left(), scope)?;
            let right = self.convert_child(delegate, binary_expr.right(), scope)?;
            if boolean_operator {
                if left.data_type != DataType::Boolean || right.data_type != DataType::Boolean {
                    return Err(exec_datafusion_err!(
                        "Boolean operator requires Boolean operands: {expr}"
                    )
                    .into());
                }
                // DataFusion may evaluate the RHS only on rows selected by the LHS.
                if !(left.infallible && right.infallible) {
                    return Err(Unconverted::Unsupported);
                }
            } else if !supported_data_types(&left.data_type)
                || !supported_data_types(&right.data_type)
            {
                return Err(Unconverted::Unsupported);
            }
            if !left.dtype.eq_ignore_nullability(&right.dtype) {
                return Err(Unconverted::Unsupported);
            }
            let data_type =
                BinaryTypeCoercer::new(&left.data_type, binary_expr.op(), &right.data_type)
                    .get_result_type()?;
            let expression = Binary.new_expr(
                operator,
                [left.expression.clone(), right.expression.clone()],
            );
            return ConvertedExpression::node(expression, data_type, &[&left, &right]);
        }

        if let Some(col_expr) = expr.downcast_ref::<df_expr::Column>() {
            let data_type = col_expr.data_type(schema)?;
            return ConvertedExpression::describe(
                get_item(col_expr.name(), root()),
                data_type,
                scope,
            );
        }

        if let Some(literal) = expr.downcast_ref::<df_expr::Literal>() {
            let field = literal.return_field(schema)?;
            // Plain scalars convert directly; extension types and other Arrow types resolve
            // through the session's Arrow conversion below.
            if field.metadata().is_empty() && is_plain_scalar(literal.value()) {
                let scalar = scalar_from_df(literal.value(), &self.session)
                    .map_err(|e| exec_datafusion_err!("Failed to convert literal {expr}: {e}"))?;
                return ConvertedExpression::describe(
                    lit(scalar),
                    field.data_type().clone(),
                    scope,
                );
            }
            let array = literal.value().to_array()?;
            if array.len() != 1 {
                return Err(exec_datafusion_err!(
                    "Literal must contain exactly one value, found {}",
                    array.len()
                )
                .into());
            }
            // Literals of unknown Arrow types stay in DataFusion; conversion failures are errors.
            self.arrow_dtype(&field)?;
            let scalar = self
                .session
                .arrow()
                .from_arrow_array(array, &field)
                .and_then(|array| array.execute_scalar(0, &mut self.session.create_execution_ctx()))
                .map_err(|e| exec_datafusion_err!("Failed to convert literal {expr}: {e}"))?;
            return ConvertedExpression::describe(lit(scalar), field.data_type().clone(), scope);
        }

        if let Some(cast_expr) = expr.downcast_ref::<df_expr::CastExpr>() {
            let child = self.convert_child(delegate, cast_expr.expr(), scope)?;
            let pushdown = classify_cast(cast_expr, &child.data_type)?;
            // DataFusion casts preserve input nulls even when the target field is declared
            // non-nullable. Match the expression's runtime nullability rather than narrowing
            // to the target field's metadata.
            let cast_dtype = self
                .arrow_dtype(cast_expr.target_field())?
                .with_nullability(child.dtype.nullability());
            let data_type = cast_expr.cast_type().clone();
            return match pushdown {
                // Matching Arrow types do not imply matching Vortex dtypes, e.g. for extensions.
                CastPushdown::Drop if child.dtype.eq_ignore_nullability(&cast_dtype) => {
                    Ok(child.with_data_type(data_type))
                }
                CastPushdown::Drop => Err(Unconverted::Unsupported),
                CastPushdown::Keep => {
                    // Matching Arrow storage types do not imply matching extension semantics.
                    if (child.dtype.is_extension() || cast_dtype.is_extension())
                        && !child.dtype.eq_ignore_nullability(&cast_dtype)
                    {
                        return Err(Unconverted::Unsupported);
                    }
                    ConvertedExpression::node(
                        cast(child.expression.clone(), cast_dtype),
                        data_type,
                        &[&child],
                    )
                }
            };
        }

        if let Some(not_expr) = expr.downcast_ref::<df_expr::NotExpr>() {
            let arg = self.convert_child(delegate, not_expr.arg(), scope)?;
            if arg.data_type != DataType::Boolean {
                return Err(exec_datafusion_err!("NOT requires a Boolean operand: {expr}").into());
            }
            return ConvertedExpression::node(
                not(arg.expression.clone()),
                DataType::Boolean,
                &[&arg],
            );
        }

        if let Some(is_null_expr) = expr.downcast_ref::<df_expr::IsNullExpr>() {
            let arg = self.convert_child(delegate, is_null_expr.arg(), scope)?;
            return ConvertedExpression::node(
                is_null(arg.expression.clone()),
                DataType::Boolean,
                &[&arg],
            );
        }

        if let Some(is_not_null_expr) = expr.downcast_ref::<df_expr::IsNotNullExpr>() {
            let arg = self.convert_child(delegate, is_not_null_expr.arg(), scope)?;
            return ConvertedExpression::node(
                is_not_null(arg.expression.clone()),
                DataType::Boolean,
                &[&arg],
            );
        }

        if let Some(like) = expr.downcast_ref::<df_expr::LikeExpr>() {
            let child = self.convert_child(delegate, like.expr(), scope)?;
            let pattern = self.convert_child(delegate, like.pattern(), scope)?;
            if !child.data_type.is_string() || !pattern.data_type.is_string() {
                return Err(Unconverted::Unsupported);
            }
            let expression = Like.new_expr(
                LikeOptions {
                    negated: like.negated(),
                    case_insensitive: like.case_insensitive(),
                },
                [child.expression.clone(), pattern.expression.clone()],
            );
            return ConvertedExpression::node(expression, DataType::Boolean, &[&child, &pattern]);
        }

        if let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>() {
            return self.convert_in_list(delegate, in_list, scope);
        }

        if let Some(scalar_fn) = expr.downcast_ref::<ScalarFunctionExpr>() {
            return self.convert_scalar_function(delegate, scalar_fn, scope);
        }

        if let Some(case_expr) = expr.downcast_ref::<df_expr::CaseExpr>() {
            return self.convert_case_expr(delegate, case_expr, scope);
        }

        Err(Unconverted::Unsupported)
    }

    fn convert_in_list(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        in_list: &df_expr::InListExpr,
        scope: &ConversionScope<'_>,
    ) -> Conversion<ConvertedExpression> {
        if in_list.is_empty()
            || !in_list
                .list()
                .iter()
                .all(|expr| expr.is::<df_expr::Literal>())
        {
            return Err(Unconverted::Unsupported);
        }
        let value = self.convert_child(delegate, in_list.expr(), scope)?;
        // Boolean rewrites may skip evaluating the input, particularly for all-null lists.
        if !supported_data_types(&value.data_type) || !value.infallible {
            return Err(Unconverted::Unsupported);
        }

        let elements = in_list
            .list()
            .iter()
            .map(|element| {
                let element = self.convert_child(delegate, element, scope)?;
                element
                    .expression
                    .as_opt::<Literal>()
                    .cloned()
                    .ok_or(Unconverted::Unsupported)
            })
            .collect::<Conversion<Vec<_>>>()?;

        // Build the membership test over the value's dtype alone, then put the value in place
        // of the root, so that its type checks do not walk the value again.
        // An element of another type than the value declines the pushdown: no implicit casts.
        let membership = in_list_of(root(), value.dtype.clone(), elements)
            .map_err(|_| Unconverted::Unsupported)?;
        let membership = if in_list.negated() {
            not(membership)
        } else {
            membership
        };
        let dtype = converted_dtype(&membership, &value.dtype)?;
        let infallible = is_infallible(&membership) && value.infallible;
        let expression = replace(membership, &root(), value.expression);
        Ok(ConvertedExpression {
            expression,
            dtype,
            data_type: DataType::Boolean,
            infallible,
        })
    }

    fn convert_scalar_function(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        scalar_fn: &ScalarFunctionExpr,
        scope: &ConversionScope<'_>,
    ) -> Conversion<ConvertedExpression> {
        let schema = scope.schema;
        let data_type = scalar_fn.return_type().clone();
        if ScalarFunctionExpr::try_downcast_func::<ArrayHas>(scalar_fn).is_some() {
            let [list, needle] = scalar_fn.args() else {
                return Err(exec_datafusion_err!("array_has requires two arguments").into());
            };
            // DataFusion's array_has is null for a null needle even against an empty list,
            // where list_contains is false. Otherwise their null semantics match: a null list
            // is null and null elements never match. Only push down needles that cannot be null.
            if needle.nullable(schema)? {
                return Err(Unconverted::Unsupported);
            }
            let list = self.convert_child(delegate, list, scope)?;
            let needle = self.convert_child(delegate, needle, scope)?;
            return ConvertedExpression::node(
                list_contains(list.expression.clone(), needle.expression.clone()),
                data_type,
                &[&list, &needle],
            );
        }
        if ScalarFunctionExpr::try_downcast_func::<GetFieldFunc>(scalar_fn).is_some() {
            // DataFusion's GetFieldFunc flattens nested field access into a single call
            // with multiple field name arguments, e.g. `outer.inner.leaf` becomes
            // get_field(Column("outer"), "inner", "leaf").
            let [source, paths @ ..] = scalar_fn.args() else {
                return Err(
                    exec_datafusion_err!("get_field requires a source and field path").into(),
                );
            };
            if paths.is_empty() {
                return Err(exec_datafusion_err!("get_field requires a field path").into());
            }
            let mut result = self.convert_child(delegate, source, scope)?;
            for path in paths {
                let name = path
                    .downcast_ref::<df_expr::Literal>()
                    .and_then(|literal| literal.value().try_as_str().flatten())
                    .ok_or_else(|| {
                        exec_datafusion_err!("get_field path must be a non-null string literal")
                    })?;
                let DataType::Struct(fields) = &result.data_type else {
                    return Err(Unconverted::Unsupported);
                };
                let (_, field) = fields.find(name).ok_or_else(|| {
                    exec_datafusion_err!("get_field references missing field {name}")
                })?;
                let field_type = field.data_type().clone();
                result = ConvertedExpression::node(
                    get_item(name, result.expression.clone()),
                    field_type,
                    &[&result],
                )?;
            }
            return Ok(result);
        }

        let (input, length, accepts): LengthFunction<'_> =
            if ScalarFunctionExpr::try_downcast_func::<OctetLengthFunc>(scalar_fn).is_some() {
                let [input] = scalar_fn.args() else {
                    return Err(
                        exec_datafusion_err!("octet_length requires exactly one argument").into(),
                    );
                };
                (input, byte_length, is_byte_array_type)
            } else if ScalarFunctionExpr::try_downcast_func::<ArrayLength>(scalar_fn).is_some() {
                let input = array_length_input(scalar_fn)?.ok_or(Unconverted::Unsupported)?;
                (input, list_length, is_list_type)
            } else {
                return Err(Unconverted::Unsupported);
            };
        let input = self.convert_child(delegate, input, scope)?;
        if !accepts(&input.data_type) {
            return Err(Unconverted::Unsupported);
        }
        let return_dtype = self.arrow_dtype(&Field::new(
            "",
            scalar_fn.return_type().clone(),
            scalar_fn.nullable(),
        ))?;
        let length_of = ConvertedExpression::node(
            length(input.expression.clone()),
            data_type.clone(),
            &[&input],
        )?;
        ConvertedExpression::node(
            cast(length_of.expression.clone(), return_dtype),
            data_type,
            &[&length_of],
        )
    }

    fn convert_case_expr(
        &self,
        delegate: Option<&dyn ExpressionConverter>,
        case_expr: &df_expr::CaseExpr,
        scope: &ConversionScope<'_>,
    ) -> Conversion<ConvertedExpression> {
        // Only the searched form (CASE WHEN cond THEN value ...) is supported.
        if case_expr.expr().is_some() {
            return Err(Unconverted::Unsupported);
        }
        let mut pairs = Vec::with_capacity(case_expr.when_then_expr().len());
        for (when_expr, then_expr) in case_expr.when_then_expr() {
            let condition = self.convert_child(delegate, when_expr, scope)?;
            if condition.data_type != DataType::Boolean {
                return Err(exec_datafusion_err!("CASE WHEN must be Boolean").into());
            }
            let value = self.convert_child(delegate, then_expr, scope)?;
            pairs.push((condition.expression, value.expression));
        }
        let else_value = case_expr
            .else_expr()
            .map(|e| self.convert_child(delegate, e, scope))
            .transpose()?
            .map(|value| value.expression);
        // CASE evaluates conditions on remaining rows and values only on selected rows,
        // so fallible children retain DataFusion's short-circuit behavior.
        let data_type = case_expr.data_type(scope.schema)?;
        ConvertedExpression::describe(nested_case_when(pairs, else_value), data_type, scope)
    }
}

impl ExpressionConverter for DefaultExpressionConverter {
    fn try_convert(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> DFResult<Option<Expression>> {
        self.try_convert_inner(None, expr, schema)
    }

    fn convert_subexpression(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        scope: &ConversionScope<'_>,
    ) -> DFResult<Option<ConvertedExpression>> {
        finish(self.convert_node(None, expr, scope))
    }
}

/// The Arrow field DataFusion expects `expr` to return. Columns, literals, casts and scalar
/// functions report fields that may carry extension metadata, so their own fields are used.
/// Other expressions return plain types: their default `return_field` formats the expression
/// to name the field, so the type tracked during conversion is used instead.
fn expected_field(
    expr: &Arc<dyn PhysicalExpr>,
    converted: &ConvertedExpression,
    schema: &Schema,
) -> DFResult<Field> {
    if expr.is::<df_expr::Column>()
        || expr.is::<df_expr::Literal>()
        || expr.is::<df_expr::CastExpr>()
        || expr.is::<ScalarFunctionExpr>()
    {
        return Ok(expr.return_field(schema)?.as_ref().clone());
    }
    Ok(Field::new("", converted.data_type.clone(), true))
}

/// Whether a literal holds a non-null boolean, number or string, which converts to the same
/// Vortex scalar without going through Arrow.
fn is_plain_scalar(value: &ScalarValue) -> bool {
    !value.is_null()
        && matches!(
            value,
            ScalarValue::Boolean(_)
                | ScalarValue::Int8(_)
                | ScalarValue::Int16(_)
                | ScalarValue::Int32(_)
                | ScalarValue::Int64(_)
                | ScalarValue::UInt8(_)
                | ScalarValue::UInt16(_)
                | ScalarValue::UInt32(_)
                | ScalarValue::UInt64(_)
                | ScalarValue::Float32(_)
                | ScalarValue::Float64(_)
                | ScalarValue::Utf8(_)
                | ScalarValue::LargeUtf8(_)
                | ScalarValue::Utf8View(_)
        )
}

/// A length function's input, the Vortex function, and the input types it accepts.
type LengthFunction<'a> = (
    &'a Arc<dyn PhysicalExpr>,
    fn(Expression) -> Expression,
    fn(&DataType) -> bool,
);

/// Whether `octet_length` over this type can become a Vortex `byte_length`.
fn is_byte_array_type(data_type: &DataType) -> bool {
    let data_type = match data_type {
        DataType::Dictionary(_, value) => value.as_ref(),
        data_type => data_type,
    };
    data_type.is_binary() || data_type.is_string()
}

/// Whether `array_length` over this type can become a Vortex `list_length`.
fn is_list_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    )
}

/// Map a conversion outcome to the public result: unsupported expressions are `None`.
fn finish<T>(conversion: Conversion<T>) -> DFResult<Option<T>> {
    match conversion {
        Ok(converted) => Ok(Some(converted)),
        Err(Unconverted::Unsupported) => Ok(None),
        Err(Unconverted::Failed(error)) => Err(error),
    }
}

/// Whether evaluating `expr` cannot fail, so Vortex may evaluate it on rows DataFusion skips.
fn is_infallible(expr: &Expression) -> bool {
    expr.as_scalar()
        .is_none_or(|scalar_fn| scalar_fn.signature().is_infallible())
        && expr.children().iter().all(is_infallible)
}

/// The Vortex return dtype of a converted expression; unresolvable dtypes are unsupported.
fn converted_dtype(expr: &Expression, input_dtype: &DType) -> Conversion<DType> {
    expr.return_dtype(input_dtype)
        .map_err(|_| Unconverted::Unsupported)
}

/// How a supported DataFusion cast is converted.
enum CastPushdown {
    /// The Arrow cast cannot fail and keeps the Vortex dtype, so the converter omits it.
    Drop,
    /// The cast becomes a Vortex `cast`.
    Keep,
}

/// Classifies a cast for pushdown; unsupported casts stay in DataFusion.
fn classify_cast(cast: &df_expr::CastExpr, source: &DataType) -> Conversion<CastPushdown> {
    use DataType::*;
    let options = cast.cast_options();
    if options.safe || options.format_options != DEFAULT_FORMAT_OPTIONS {
        return Err(Unconverted::Unsupported);
    }
    let target = cast.cast_type();
    Ok(match (source, target) {
        _ if source == target => CastPushdown::Drop,
        // Arrow expands runs by gathering with `Int32` indices, which fit Int16/Int32 run ends.
        // Gathering a dictionary only moves keys and reuses the values, so nothing can overflow.
        (RunEndEncoded(run_ends, values), Dictionary(..))
            if matches!(run_ends.data_type(), Int16 | Int32) && values.data_type() == target =>
        {
            CastPushdown::Drop
        }
        (Int8, Int16 | Int32 | Int64)
        | (Int16, Int32 | Int64)
        | (Int32, Int64)
        | (UInt8, UInt16 | UInt32 | UInt64)
        | (UInt16, UInt32 | UInt64)
        | (UInt32, UInt64)
        | (Float32, Float64) => CastPushdown::Keep,
        _ => return Err(Unconverted::Unsupported),
    })
}

fn try_operator_from_df(value: &DFOperator) -> Option<Operator> {
    match value {
        DFOperator::Eq => Some(Operator::Eq),
        DFOperator::NotEq => Some(Operator::NotEq),
        DFOperator::Lt => Some(Operator::Lt),
        DFOperator::LtEq => Some(Operator::Lte),
        DFOperator::Gt => Some(Operator::Gt),
        DFOperator::GtEq => Some(Operator::Gte),
        DFOperator::And => Some(Operator::And),
        DFOperator::Or => Some(Operator::Or),
        DFOperator::Plus => Some(Operator::Add),
        DFOperator::Minus => Some(Operator::Sub),
        DFOperator::Multiply => Some(Operator::Mul),
        DFOperator::Divide => Some(Operator::Div),
        DFOperator::IsDistinctFrom
        | DFOperator::IsNotDistinctFrom
        | DFOperator::RegexMatch
        | DFOperator::RegexIMatch
        | DFOperator::RegexNotMatch
        | DFOperator::RegexNotIMatch
        | DFOperator::LikeMatch
        | DFOperator::ILikeMatch
        | DFOperator::NotLikeMatch
        | DFOperator::NotILikeMatch
        | DFOperator::BitwiseAnd
        | DFOperator::BitwiseOr
        | DFOperator::BitwiseXor
        | DFOperator::BitwiseShiftRight
        | DFOperator::BitwiseShiftLeft
        | DFOperator::StringConcat
        | DFOperator::AtArrow
        | DFOperator::ArrowAt
        | DFOperator::Modulo
        | DFOperator::Arrow
        | DFOperator::LongArrow
        | DFOperator::HashArrow
        | DFOperator::HashLongArrow
        | DFOperator::AtAt
        | DFOperator::IntegerDivide
        | DFOperator::HashMinus
        | DFOperator::AtQuestion
        | DFOperator::Question
        | DFOperator::QuestionAnd
        | DFOperator::QuestionPipe
        | DFOperator::Colon => None,
    }
}

fn supported_data_types(dt: &DataType) -> bool {
    use DataType::*;

    // For dictionary types, check if the value type is supported.
    if let Dictionary(_, value_type) = dt {
        return supported_data_types(value_type.as_ref());
    }

    let is_supported = dt.is_null()
        || dt.is_numeric()
        || dt.is_binary()
        || dt.is_string()
        || matches!(
            dt,
            Boolean | Date32 | Date64 | Timestamp(_, _) | Time32(_) | Time64(_)
        );

    if !is_supported {
        tracing::debug!("DataFusion data type {dt:?} is not supported");
    }

    is_supported
}

/// Returns the list argument of an `array_length` call that can be rewritten to `list_length`:
/// the single-argument form `array_length(arr)` or the equivalent explicit first dimension
/// `array_length(arr, 1)`. Other dimensions are unsupported (`None`); other arities are errors.
fn array_length_input(scalar_fn: &ScalarFunctionExpr) -> DFResult<Option<&Arc<dyn PhysicalExpr>>> {
    match scalar_fn.args() {
        [input] => Ok(Some(input)),
        [input, dimension]
            if dimension
                .downcast_ref::<df_expr::Literal>()
                .is_some_and(|literal| matches!(literal.value(), ScalarValue::Int64(Some(1)))) =>
        {
            Ok(Some(input))
        }
        [_, _] => Ok(None),
        _ => Err(exec_datafusion_err!(
            "array_length requires one or two arguments"
        )),
    }
}
