// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::StructArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::display::ExprDisplay;
use crate::matcher::Matcher;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Projects one field from every struct element of a list.
#[derive(Clone)]
pub struct ListGetField;

impl ScalarFnVTable for ListGetField {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.get_field");
        *ID
    }

    fn serialize(&self, field_name: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(field_name.to_string().into_bytes()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let field_name = String::from_utf8(metadata.to_vec())
            .map_err(|error| vortex_err!("Invalid UTF-8 list field name: {error}"))?;
        Ok(FieldName::from(field_name))
    }

    fn arity(&self, _field_name: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _field_name: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_get_field()"),
        }
    }

    fn fmt_sql(
        &self,
        field_name: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "list_get_field({}, '{field_name}')",
            expr.display_child(0)
        )
    }

    fn return_dtype(
        &self,
        field_name: &Self::Options,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        let DType::List(element_dtype, nullability) = &arg_dtypes[0] else {
            return Err(vortex_err!(
                "list_get_field() requires List<Struct>, got {}",
                arg_dtypes[0]
            ));
        };
        let field_dtype = element_dtype
            .as_struct_fields_opt()
            .and_then(|fields| fields.field(field_name))
            .ok_or_else(|| vortex_err!("List element has no field named {field_name}"))?;
        Ok(DType::List(Arc::new(field_dtype), *nullability))
    }

    fn execute(
        &self,
        field_name: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?.execute_until::<ListOnly>(ctx)?;
        let input = input
            .as_opt::<List>()
            .vortex_expect("ListOnly matcher returned a non-list array");
        let elements = input.elements().clone().execute::<StructArray>(ctx)?;
        let field = elements.unmasked_field_by_name(field_name).cloned()?;
        let field = match elements.dtype().nullability() {
            Nullability::NonNullable => field,
            Nullability::Nullable => field.mask(elements.validity()?.to_array(elements.len()))?,
        };
        Ok(ListArray::try_new(field, input.offsets().clone(), input.list_validity())?.into_array())
    }

    fn validity(
        &self,
        _field_name: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_strict(&self, _field_name: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _field_name: &Self::Options) -> bool {
        true
    }
}

struct ListOnly;

impl Matcher for ListOnly {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        array.as_opt::<List>().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::expr::list_get_field;
    use crate::expr::root;
    use crate::Canonical;
    use crate::VortexSessionExecute;

    #[test]
    fn projects_field_from_list_elements() -> VortexResult<()> {
        let elements = StructArray::from_fields(&[
            ("selected", buffer![1_i32, 2, 3].into_array()),
            ("ignored", buffer![10_i64, 20, 30].into_array()),
        ])?
        .into_array();
        let list = ListArray::try_new(
            elements,
            buffer![0_u32, 2, 3].into_array(),
            crate::validity::Validity::NonNullable,
        )?
        .into_array();
        let expected = ListArray::try_new(
            buffer![1_i32, 2, 3].into_array(),
            buffer![0_u32, 2, 3].into_array(),
            crate::validity::Validity::NonNullable,
        )?
        .into_array();

        let expression = list_get_field("selected", root());
        let mut ctx = crate::array_session().create_execution_ctx();
        let result = list.apply(&expression)?.execute::<Canonical>(&mut ctx)?.into_array();
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
