// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Converting DataFusion predicates to Vortex expressions: each node's checks must not walk its
//! subtree again, so conversion stays linear in the size of the predicate.

#![allow(clippy::unwrap_used)]

use std::ops::Range;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::expressions as df_expr;
use divan::Bencher;
use vortex_datafusion::convert::DefaultExpressionConverter;
use vortex_datafusion::convert::ExpressionConverter;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[16, 256, 4096];

fn schema() -> Schema {
    Schema::new(vec![Field::new("id", DataType::Utf8, false)])
}

fn id_column() -> Arc<dyn PhysicalExpr> {
    Arc::new(df_expr::Column::new("id", 0))
}

fn id_literal(index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(df_expr::Literal::new(ScalarValue::Utf8(Some(format!(
        "id-{index}"
    )))))
}

fn id_equals(index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(df_expr::BinaryExpr::new(
        id_column(),
        Operator::Eq,
        id_literal(index),
    ))
}

fn balanced_or(range: Range<usize>) -> Arc<dyn PhysicalExpr> {
    if range.len() == 1 {
        return id_equals(range.start);
    }
    let mid = range.start + range.len() / 2;
    Arc::new(df_expr::BinaryExpr::new(
        balanced_or(range.start..mid),
        Operator::Or,
        balanced_or(mid..range.end),
    ))
}

fn left_deep_or(size: usize) -> Arc<dyn PhysicalExpr> {
    (1..size).fold(id_equals(0), |left, index| {
        Arc::new(df_expr::BinaryExpr::new(
            left,
            Operator::Or,
            id_equals(index),
        ))
    })
}

fn in_list(size: usize) -> Arc<dyn PhysicalExpr> {
    df_expr::in_list(
        id_column(),
        (0..size).map(id_literal).collect(),
        &false,
        &schema(),
    )
    .unwrap()
}

fn bench_conversion(bencher: Bencher, expr: Arc<dyn PhysicalExpr>) {
    let schema = schema();
    let converter = DefaultExpressionConverter::default();
    bencher.bench(|| converter.try_convert(&expr, &schema).unwrap().unwrap());
}

#[divan::bench(args = SIZES)]
fn convert_balanced_or(bencher: Bencher, size: usize) {
    bench_conversion(bencher, balanced_or(0..size));
}

#[divan::bench(args = SIZES)]
fn convert_left_deep_or(bencher: Bencher, size: usize) {
    bench_conversion(bencher, left_deep_or(size));
}

#[divan::bench(args = SIZES)]
fn convert_in_list(bencher: Bencher, size: usize) {
    bench_conversion(bencher, in_list(size));
}
