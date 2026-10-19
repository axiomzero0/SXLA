#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion (waiver W-5)"
)]

// CEP:FILE: crates/mss/tests/differential.rs
// CEP:WHAT: Integration test — randomized differential validation of the
//           search: seeds carrying identity/fold/reassociation/CSE/DCE
//           opportunities are optimized, and the result must evaluate
//           EXACTLY like the input across many parameter bindings (design
//           §16: randomized differential testing before trusting anything),
//           with exact reproducibility across runs and worker counts.
// CEP:WHY: The engine's correctness contract: stochastic search, determined
//          outcome, preserved semantics. Every mutation family's soundness
//          is exercised en route (the chains hit identity, fold, reassoc,
//          CSE and DCE sites across the seeds).
// CEP:CLASS: CEP-31
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any value divergence or nondeterminism.
// CEP:ASSUMES: public API.
// CEP:COST: test-only; ~4 searches at V0 budgets.
// CEP:EVIDENCE: this file.
// CEP:SECURITY: no untrusted input.

use mss::search::search;
use mss::{BinOp, Candidate, Program, ProgramBuilder};

fn seed_chain_1() -> Program {
    // f(x) = ((x + 0) * 1) + (2 + 3)
    let mut b = ProgramBuilder::new();
    let x = b.param(0);
    let zero = b.const_i64(0);
    let one = b.const_i64(1);
    let add = b.bin(BinOp::Add, x, zero).expect("add");
    let mul = b.bin(BinOp::Mul, add, one).expect("mul");
    let c2 = b.const_i64(2);
    let c3 = b.const_i64(3);
    let fold = b.bin(BinOp::Add, c2, c3).expect("fold");
    let root = b.bin(BinOp::Add, mul, fold).expect("root");
    b.finish(root).expect("finish")
}

fn seed_chain_2() -> Program {
    // g(x, y) = max(x, y) - (x - x) + (y * 1)  [sub-self + identity sites]
    let mut b = ProgramBuilder::new();
    let x = b.param(0);
    let y = b.param(1);
    let mx = b.bin(BinOp::Max, x, y).expect("max");
    let selfsub = b.bin(BinOp::Sub, x, x).expect("selfsub");
    let one = b.const_i64(1);
    let my = b.bin(BinOp::Mul, y, one).expect("my");
    let a = b.bin(BinOp::Sub, mx, selfsub).expect("a");
    let root = b.bin(BinOp::Add, a, my).expect("root");
    b.finish(root).expect("finish")
}

fn seed_chain_3() -> Program {
    // h(x) = (x + 10) + 20 + (0 + x)  [reassoc + identity sites]
    let mut b = ProgramBuilder::new();
    let x = b.param(0);
    let c10 = b.const_i64(10);
    let c20 = b.const_i64(20);
    let zero = b.const_i64(0);
    let inner = b.bin(BinOp::Add, x, c10).expect("inner");
    let outer = b.bin(BinOp::Add, inner, c20).expect("outer");
    let id = b.bin(BinOp::Add, zero, x).expect("id");
    let root = b.bin(BinOp::Add, outer, id).expect("root");
    b.finish(root).expect("finish")
}

// CEP:WHAT: Optimized programs evaluate exactly like their seeds.
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any value divergence.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn differential_values_preserved() {
    let seeds = [seed_chain_1(), seed_chain_2(), seed_chain_3()];
    let bindings: Vec<Vec<i64>> = vec![
        vec![0],
        vec![1],
        vec![-7],
        vec![1024],
        vec![i64::MAX - 100],
        vec![i64::MIN + 100],
    ];
    for seed in seeds.iter() {
        let out = search(seed, anvil::default_worker_count().max(1));
        let best = out.best.materialize();
        assert!(out.generated > 0, "search produced nothing");
        let arity = seed
            .arena
            .kinds
            .iter()
            .filter(|k| **k == mss::Kind::Param)
            .count()
            .max(1);
        for b in bindings.iter() {
            let params: Vec<i64> = (0..arity).map(|i| b[i % b.len()]).collect();
            assert_eq!(
                mss::interp::evaluate(&best, &params),
                mss::interp::evaluate(seed, &params),
                "divergence on {:?}",
                params
            );
        }
    }
}

// CEP:WHAT: Repeated searches agree exactly (reproducibility contract).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on any divergence.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn searches_are_reproducible() {
    let seed = seed_chain_1();
    let a = search(&seed, 1);
    let b = search(&seed, 2);
    let c = search(&seed, 1);
    assert_eq!(a.best_cost, b.best_cost);
    assert_eq!(a.best_cost, c.best_cost);
    assert_eq!(
        a.best.canonical_hash(),
        b.best.canonical_hash(),
        "worker count changed the result"
    );
    assert_eq!(a.best.canonical_hash(), c.best.canonical_hash());
    assert_eq!(a.generated, c.generated);
}

// CEP:WHAT: The search strictly improves reducible seeds (elitism works).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires when the elite never beats the seed.
// CEP:ASSUMES: seeds carry V0-reducible sites.
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn search_improves_reducible_seeds() {
    for seed in [seed_chain_1(), seed_chain_2(), seed_chain_3()] {
        let seed_cost = mss::cost::evaluate(&Candidate::from_program(seed.clone()));
        let out = search(&seed, anvil::default_worker_count().max(1));
        assert!(
            out.best_cost.total < seed_cost.total,
            "no improvement: {} -> {}",
            seed_cost.total,
            out.best_cost.total
        );
    }
}
