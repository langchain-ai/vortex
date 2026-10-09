// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_array::scalar_fn::fns::list_contains::ListContainsOptions;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Zstd;

/// Minimum number of list elements for which hashing them beats one comparison per element.
const HASH_LIST_LENGTH_THRESHOLD: usize = 32;
/// Minimum number of needle values for which building the hash set pays off.
const HASH_VALUES_LENGTH_THRESHOLD: usize = 256;

impl ListContainsElementKernel for Zstd {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // The generic path compares every list element against the needle, which would
        // decompress the Zstd array once per element. Decompress it once instead.
        let values = Zstd::decompress(&element.into_owned(), ctx)?;

        if let Some(result) = hash_list_contains(list, &values, options, ctx)? {
            return Ok(Some(result));
        }

        Ok(Some(
            ListContains::try_new_opts(list.clone(), values, *options)?.into_array(),
        ))
    }
}

/// Computes `list_contains` for a constant list of strings or binaries by hashing the list's
/// elements, instead of comparing each element against every value.
///
/// Returns `None` when the list isn't a non-null constant, or when the list or the values are
/// too short for hashing to pay off.
fn hash_list_contains(
    list: &ArrayRef,
    values: &ArrayRef,
    options: &ListContainsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if values.len() < HASH_VALUES_LENGTH_THRESHOLD
        || !matches!(values.dtype(), DType::Utf8(_) | DType::Binary(_))
    {
        return Ok(None);
    }
    let Some(list_scalar) = list.as_constant() else {
        return Ok(None);
    };
    // A null list falls back to the generic path, which yields all-null.
    let Some(elements) = list_scalar.as_list().elements() else {
        return Ok(None);
    };
    if elements.len() < HASH_LIST_LENGTH_THRESHOLD {
        return Ok(None);
    }

    let mut has_null_element = false;
    let mut element_set = HashSet::with_capacity(elements.len());
    for element in &elements {
        match element_bytes(element) {
            Some(bytes) => {
                element_set.insert(bytes);
            }
            None => has_null_element = true,
        }
    }

    let values = values.clone().execute::<VarBinViewArray>(ctx)?;
    let matches = (0..values.len())
        .map(|index| element_set.contains(values.bytes_at(index).as_slice()))
        .collect::<BitBuffer>();

    // A null value is null against a non-empty list. Under SQL null semantics, a null element
    // also makes every non-match unknown.
    let nullability = options.result_nullability(list.dtype(), values.dtype());
    let values_valid = values
        .validity()?
        .execute_mask(values.len(), ctx)?
        .into_bit_buffer();
    let validity = if options.sql_null_semantics && has_null_element {
        Validity::from_bit_buffer(&matches & &values_valid, nullability)
    } else if values.dtype().is_nullable() {
        Validity::from_bit_buffer(values_valid, nullability)
    } else {
        nullability.into()
    };

    Ok(Some(BoolArray::new(matches, validity).into_array()))
}

/// The bytes of a string or binary list element, or `None` if it is null.
fn element_bytes(element: &Scalar) -> Option<&[u8]> {
    match element.dtype() {
        DType::Utf8(_) => element
            .as_utf8()
            .value()
            .map(|value| value.as_str().as_bytes()),
        DType::Binary(_) => element.as_binary().value().map(|value| value.as_slice()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::list_contains::ListContains;
    use vortex_array::scalar_fn::fns::list_contains::ListContainsOptions;
    use vortex_session::VortexSession;

    use super::HASH_LIST_LENGTH_THRESHOLD;
    use super::HASH_VALUES_LENGTH_THRESHOLD;
    use super::hash_list_contains;
    use crate::Zstd;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    const SQL: ListContainsOptions = ListContainsOptions {
        sql_null_semantics: true,
    };

    /// Values `value-0` .. `value-{n-1}`, with every 100th value null.
    fn values(dtype: &DType) -> VarBinViewArray {
        VarBinViewArray::from_iter(
            (0..HASH_VALUES_LENGTH_THRESHOLD).map(|i| (i % 100 != 3).then(|| format!("value-{i}"))),
            dtype.clone(),
        )
    }

    /// A constant list of every 8th value plus a value that never occurs, optionally with a null.
    fn constant_list(dtype: &DType, with_null: bool, len: usize) -> ArrayRef {
        let element = |value: &str| match dtype {
            DType::Binary(_) => Scalar::binary(value.as_bytes().to_vec(), Nullability::Nullable),
            _ => Scalar::utf8(value, Nullability::Nullable),
        };
        let mut elements = (0..HASH_LIST_LENGTH_THRESHOLD)
            .map(|i| element(&format!("value-{}", i * 8)))
            .chain([element("missing")])
            .collect::<Vec<_>>();
        if with_null {
            elements.push(Scalar::null(dtype.as_nullable()));
        }
        ConstantArray::new(
            Scalar::list(
                Arc::new(dtype.as_nullable()),
                elements,
                Nullability::NonNullable,
            ),
            len,
        )
        .into_array()
    }

    fn generic(list: ArrayRef, needle: ArrayRef, options: ListContainsOptions) -> BoolArray {
        let mut ctx = SESSION.create_execution_ctx();
        ListContains::try_new_opts(list, needle, options)
            .expect("valid list_contains")
            .into_array()
            .execute::<BoolArray>(&mut ctx)
            .expect("generic list_contains executes")
    }

    #[test]
    fn list_contains_zstd_utf8() {
        let values = VarBinViewArray::from_iter(
            [Some("a"), Some("b"), None, Some("d")],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let values = Zstd::from_var_bin_view(&values, 0, 4, &mut ctx).unwrap();
        let list = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Utf8(Nullability::Nullable)),
                vec![
                    Scalar::utf8("b", Nullability::Nullable),
                    Scalar::utf8("d", Nullability::Nullable),
                ],
                Nullability::NonNullable,
            ),
            values.len(),
        )
        .into_array();

        let result = generic(list, values.into_array(), ListContainsOptions::default());
        // A null needle is null against a non-empty list.
        let expected = BoolArray::from_iter([Some(false), Some(true), None, Some(true)]);

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    /// The hash path agrees with the generic path, including null needles and null elements,
    /// for both null semantics.
    #[rstest]
    fn hash_path_matches_generic(
        #[values(
            DType::Utf8(Nullability::Nullable),
            DType::Binary(Nullability::Nullable)
        )]
        dtype: DType,
        #[values(false, true)] with_null: bool,
        #[values(ListContainsOptions::default(), SQL)] options: ListContainsOptions,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let values = values(&dtype).into_array();
        let list = constant_list(&dtype, with_null, values.len());

        let hashed = hash_list_contains(&list, &values, &options, &mut ctx)
            .expect("hash path executes")
            .expect("hash path applies above the thresholds");
        let expected = generic(list, values, options);

        assert_arrays_eq!(hashed, expected, &mut ctx);
    }

    /// End to end over a Zstd needle: the kernel decompresses once and keeps the null semantics.
    #[rstest]
    fn zstd_needle_matches_generic(
        #[values(false, true)] with_null: bool,
        #[values(ListContainsOptions::default(), SQL)] options: ListContainsOptions,
    ) {
        let dtype = DType::Utf8(Nullability::Nullable);
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = values(&dtype);
        let zstd = Zstd::from_var_bin_view(&canonical, 0, canonical.len(), &mut ctx).unwrap();
        let list = constant_list(&dtype, with_null, canonical.len());

        let result = generic(list.clone(), zstd.into_array(), options);
        let expected = generic(list, canonical.into_array(), options);

        assert_arrays_eq!(result, expected, &mut ctx);
    }
}
