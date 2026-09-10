// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::Zstd;

const HASH_LIST_LENGTH_THRESHOLD: usize = 32;
const HASH_VALUES_LENGTH_THRESHOLD: usize = 256;

impl ListContainsElementKernel for Zstd {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Avoid decompressing the Zstd array once for every item in a constant list.
        let values = Zstd::decompress(&element.into_owned(), ctx)?;

        if let Some(list_scalar) = list.as_constant() {
            let list_scalar = list_scalar.as_list();
            if list_scalar.len() >= HASH_LIST_LENGTH_THRESHOLD
                && values.len() >= HASH_VALUES_LENGTH_THRESHOLD
                && matches!(values.dtype(), DType::Utf8(_) | DType::Binary(_))
                && let Some(elements) = list_scalar.elements()
            {
                let elements = elements
                    .iter()
                    .filter(|element| !element.is_null())
                    .map(|element| element.cast(values.dtype()))
                    .collect::<VortexResult<HashSet<_>>>()?;
                let matches = (0..values.len())
                    .map(|index| {
                        values
                            .execute_scalar(index, ctx)
                            .map(|value| !value.is_null() && elements.contains(&value))
                    })
                    .collect::<VortexResult<BitBuffer>>()?;
                let nullability = list.dtype().nullability() | values.dtype().nullability();

                return Ok(Some(
                    BoolArray::new(matches, Validity::from(nullability)).into_array(),
                ));
            }
        }

        list.clone().list_contains(values).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_session::VortexSession;

    use super::HASH_LIST_LENGTH_THRESHOLD;
    use super::HASH_VALUES_LENGTH_THRESHOLD;
    use crate::Zstd;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

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

        let result = list
            .list_contains(values.into_array())
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        let expected = BoolArray::from_iter([Some(false), Some(true), Some(false), Some(true)]);

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn list_contains_zstd_utf8_hash_path() {
        let strings = (0..HASH_VALUES_LENGTH_THRESHOLD)
            .map(|i| format!("value-{i}"))
            .collect::<Vec<_>>();
        let values = VarBinViewArray::from_iter(
            strings
                .iter()
                .enumerate()
                .map(|(i, value)| (i != 128).then_some(value.as_str())),
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let values = Zstd::from_var_bin_view(&values, 0, values.len(), &mut ctx).unwrap();
        let list = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Utf8(Nullability::NonNullable)),
                (0..HASH_LIST_LENGTH_THRESHOLD)
                    .map(|i| Scalar::utf8(format!("value-{}", i * 8), Nullability::NonNullable))
                    .collect(),
                Nullability::NonNullable,
            ),
            values.len(),
        )
        .into_array();

        let result = list
            .list_contains(values.into_array())
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        let mut expected = (0..HASH_VALUES_LENGTH_THRESHOLD)
            .map(|i| Some(i % 8 == 0))
            .collect::<Vec<_>>();
        expected[128] = Some(false);

        assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
    }
}
