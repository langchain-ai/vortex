// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks destruction of uniquely owned expression trees.

use std::hint::black_box;

use divan::Bencher;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::not;
use vortex_array::expr::root;

fn main() {
    divan::main();
}

fn unary_expression(depth: usize) -> Expression {
    (0..depth).fold(root(), |expr, _| not(expr))
}

fn balanced_expression(depth: usize) -> Expression {
    if depth == 0 {
        root()
    } else {
        and(
            balanced_expression(depth - 1),
            balanced_expression(depth - 1),
        )
    }
}

#[divan::bench(args = [1, 16, 256])]
fn drop_unique(bencher: Bencher, depth: usize) {
    bencher
        .with_inputs(|| unary_expression(depth))
        .bench_values(|expr| drop(black_box(expr)));
}

#[divan::bench(args = [4, 8])]
fn drop_balanced(bencher: Bencher, depth: usize) {
    bencher
        .with_inputs(|| balanced_expression(depth))
        .bench_values(|expr| drop(black_box(expr)));
}
