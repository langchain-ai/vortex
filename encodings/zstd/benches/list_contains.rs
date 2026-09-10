// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;
use vortex_zstd::ZstdArray;

const ROW_COUNT: usize = 8 * 1024;
const LIST_LENGTHS: &[usize] = &[4, 32, 256, 2 * 1024, 16 * 1024, 24_999];

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_zstd::initialize(&session);
    session
});

struct Inputs {
    list: ArrayRef,
    elements: Vec<Scalar>,
    values: ZstdArray,
}

fn inputs(list_len: usize) -> Inputs {
    let strings = (0..ROW_COUNT)
        .map(|i| format!("{i:08x}-0000-0000-0000-{i:012x}"))
        .collect::<Vec<_>>();
    let values = VarBinViewArray::from_iter_str(&strings);
    let values =
        Zstd::from_var_bin_view(&values, 3, ROW_COUNT, &mut SESSION.create_execution_ctx())
            .unwrap();
    let elements = (0..list_len)
        .map(|i| Scalar::from(strings[i * ROW_COUNT / list_len].as_str()))
        .collect::<Vec<_>>();
    let list = ConstantArray::new(
        Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            elements.clone(),
            Nullability::NonNullable,
        ),
        ROW_COUNT,
    )
    .into_array();

    Inputs {
        list,
        elements,
        values,
    }
}

/// Registered Zstd kernel: hash large lists and decompress the values once otherwise.
#[divan::bench(args = LIST_LENGTHS, sample_count = 10)]
fn hybrid_hash(bencher: Bencher, list_len: usize) {
    let inputs = inputs(list_len);
    let mut ctx = SESSION.create_execution_ctx();

    bencher.bench_local(|| {
        inputs
            .list
            .list_contains(inputs.values.clone().into_array())
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

/// Decompress once, then execute generic `list_contains` comparisons.
#[divan::bench(args = LIST_LENGTHS, sample_count = 10)]
fn decompress_once(bencher: Bencher, list_len: usize) {
    let inputs = inputs(list_len);
    let mut ctx = SESSION.create_execution_ctx();

    bencher.bench_local(|| {
        let values = Zstd::decompress(&inputs.values, &mut ctx).unwrap();
        inputs
            .list
            .list_contains(values)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

/// Reproduction of the generic fallback: compare against Zstd once per list element.
#[divan::bench(args = LIST_LENGTHS, sample_count = 10)]
fn decompress_per_element(bencher: Bencher, list_len: usize) {
    let inputs = inputs(list_len);
    let mut ctx = SESSION.create_execution_ctx();

    bencher.bench_local(|| {
        old_fallback(&inputs.values, &inputs.elements)
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

fn old_fallback(values: &ZstdArray, elements: &[Scalar]) -> ArrayRef {
    let len = values.len();
    let false_scalar = Scalar::bool(false, Nullability::NonNullable);

    let comparisons = elements
        .iter()
        .map(|element| {
            ConstantArray::new(element.clone(), len)
                .into_array()
                .binary(values.clone().into_array(), Operator::Eq)
                .unwrap()
                .fill_null(false_scalar.clone())
                .unwrap()
        })
        .collect();

    reduce_or_balanced(comparisons)
}

fn reduce_or_balanced(mut arrays: Vec<ArrayRef>) -> ArrayRef {
    while arrays.len() > 1 {
        let mut next = Vec::with_capacity(arrays.len().div_ceil(2));
        let mut iter = arrays.into_iter();
        while let Some(lhs) = iter.next() {
            next.push(match iter.next() {
                Some(rhs) => lhs.binary(rhs, Operator::Or).unwrap(),
                None => lhs,
            });
        }
        arrays = next;
    }

    arrays.pop().unwrap()
}

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}
