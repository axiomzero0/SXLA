// CEP:FILE: crates/mss/src/lib.rs
// CEP:WHAT: mss — the Massive Stochastic Superoptimizer V0: a high-throughput
//           stochastic search engine over Sea-of-Nodes programs (ECS arena,
//           CoW overlay candidates, mutation families, tiered filtering,
//           canonical hashing, per-band transposition tables, parallel-
//           tempering MCMC with elitism, differential interpreter).
// CEP:WHY: Dual-engine architecture: MSS owns the scalar/control SoN domain;
//          SuperTensor-WPA (crates/supertensor) owns deterministic tensor
//          scheduling. Hard separation: no shared internal state, no shared
//          concurrency primitives beyond Anvil; the engines meet only at
//          the XIR boundaries (A: pre-tensor scalar glue; B: post-scheduling
//          recovery; C: knowledge transfer) which are CEP-31 follow-ups —
//          V0 proves the engine on its own dialect.
// CEP:CLASS: CEP-31
// CEP:STATUS: partial (V0: throughput substrate + correctness; the design's
//             V1-V4 layers — LSM persistence, speculation/guards, memory
//             SSA, CRISPR plasmids, sketches, gossip — are ticketed TODOs).
// CEP:FAILURE: loud Results throughout (builder/interpreter); the search
//              never fails (bounded budgets, elite fallback = seed).
// CEP:ASSUMES: V0 pure-dataflow i64 dialect (wrapping semantics), programs
//              built topologically; determinism from fixed seeds + total
//              orders (38.10).
// CEP:COST: search = O(bands x iters x n); see per-module CEP:COST.
// CEP:EVIDENCE: crate test suite; tests/differential.rs (search results
//           evaluate equal to inputs across bindings); determinism tests.
// CEP:SECURITY: no untrusted input (no text front-end in V0); no unsafe.
// CEP:HPC-DETERMINISM: seeded rng + deterministic hashers + total-order
//           reductions — identical results on every run and worker count.
// CEP:TODO(main-agent): CEP-31: V1 (LSM-TT persistence + cost-biased
//                       dispatch learning + CDCL clauses), V2 (Assume/Guard
//                       + memory SSA + boundary A/B integration), V3
//                       (CRISPR plasmids + sketches + niching), V4 (NUMA
//                       islands + gossip + daemon).
//! # mss — Massive Stochastic Superoptimizer (V0)
//!
//! ECS Sea-of-Nodes + CoW candidates + parallel-tempering MCMC search.

pub mod cand;
pub mod cost;
pub mod interp;
pub mod mutate;
pub mod rng;
pub mod search;
pub mod son;
pub mod tt;

pub use cand::{Candidate, NodeDef, Overlay};
pub use cost::Cost;
pub use search::{search, Band, SearchOutcome, BANDS, ITERS_PER_BAND};
pub use son::{BinOp, BuildError, Kind, Program, ProgramBuilder, SonArena, UnOp};

/// CEP:WHAT: Convenience entry: search a program and return the optimized
///           materialized program.
/// CEP:WHY: The single-call surface callers use; the full SearchOutcome is
///          available through `search::search` for diagnostics.
/// CEP:STATUS: complete
/// CEP:FAILURE: never (elite fallback = seed).
/// CEP:ASSUMES: program well-formed.
/// CEP:COST: as search::search.
/// CEP:EVIDENCE: differential suite.
/// CEP:SECURITY: no untrusted input.
pub fn optimize(program: &Program) -> Program {
    let out = search::search(program, anvil::default_worker_count().max(1));
    out.best.materialize()
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;

    // CEP:WHAT: optimize() preserves values on a reducible program.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn optimize_preserves_values() {
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let one = b.const_i64(1);
        let m1 = b.bin(BinOp::Mul, x, one).expect("m1");
        let zero = b.const_i64(0);
        let add = b.bin(BinOp::Add, m1, zero).expect("add");
        let p = b.finish(add).expect("finish");
        let optimized = optimize(&p);
        for x in [-5i64, 0, 9, 12345] {
            assert_eq!(
                interp::evaluate(&optimized, &[x]),
                interp::evaluate(&p, &[x])
            );
        }
    }
}
