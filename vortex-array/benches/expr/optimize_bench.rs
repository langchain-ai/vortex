// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::make_free_field_annotator;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::merge;
use vortex_array::expr::or;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::expr::transform::replace_root_fields;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

fn struct_scope() -> DType {
    DType::Struct(
        StructFields::new(
            ["x"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    )
}

fn build_or_chain(n: usize) -> Expression {
    let base = eq(get_item("x", root()), lit(0i32));
    (1..n).fold(base, |acc, i| {
        or(acc, eq(get_item("x", root()), lit(i as i32)))
    })
}

#[divan::bench(args = [200])]
fn optimize_or_chain(bencher: Bencher, n: usize) {
    let expr = build_or_chain(n);
    let scope = struct_scope();
    bencher.bench(|| expr.optimize_recursive(&scope).unwrap());
}

fn ingestion_projection(field_count: usize) -> (Expression, DType, StructFields) {
    let field_names = (0..field_count)
        .map(|idx| FieldName::from(format!("field_{idx}")))
        .collect::<Vec<_>>();
    let fields = StructFields::new(
        field_names.clone().into(),
        (0..field_count)
            .map(|_| DType::Primitive(PType::U64, Nullability::NonNullable))
            .collect(),
    );
    let scope = DType::Struct(fields.clone(), Nullability::NonNullable);
    let expr = select(
        field_names
            .into_iter()
            .chain(["input_row_idx".into(), "input_file_idx".into()])
            .collect::<Vec<_>>(),
        merge([
            root(),
            pack(
                [
                    ("input_row_idx", lit(0_u64)),
                    ("input_file_idx", lit(1_u64)),
                ],
                Nullability::NonNullable,
            ),
        ]),
    );

    let expr = replace(expr, &root(), replace_root_fields(root(), &fields));
    (expr, scope, fields)
}

#[divan::bench(args = [10, 40, 100])]
fn optimize_ingestion_projection(bencher: Bencher, field_count: usize) {
    let (expr, scope, _) = ingestion_projection(field_count);
    bencher.bench(|| expr.optimize_recursive(&scope).unwrap());
}

#[divan::bench(args = [10, 40, 100])]
fn partition_ingestion_projection(bencher: Bencher, field_count: usize) {
    let (expr, scope, fields) = ingestion_projection(field_count);
    bencher.bench(|| {
        let optimized = expr.optimize_recursive(&scope).unwrap();
        partition(optimized, &scope, make_free_field_annotator(&fields)).unwrap()
    });
}
