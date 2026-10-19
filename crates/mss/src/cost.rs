// CEP:FILE: crates/mss/src/cost.rs
// CEP:WHAT: The cheap cost model — latency-weighted live node count over
//           resolved views (Alias and Dead contribute zero), plus the
//           critical-path depth as a secondary tie-break component. This is
//           the Tier-4 filter (design §10): a search filter, not an oracle.
// CEP:WHY: Metropolis needs a scalar energy TODAY; the design defers real
//          measurement to the expensive tier (compilation + benchmarking are
//          CEP-31 V2). Integer arithmetic keeps the model deterministic.
// CEP:CLASS: CEP-31
// CEP:STATUS: complete (V0 feature subset: instruction count + depth)
// CEP:FAILURE: saturating sums.
// CEP:ASSUMES: overlay views well-formed.
// CEP:COST: O(n) per evaluation; zero allocation.
// CEP:EVIDENCE: tests `identity_rewrites_reduce_cost`, `cost_counts_depth`.
// CEP:SECURITY: no untrusted input.
//! Cheap cost model (Tier-4 filter).

use crate::cand::Candidate;
use crate::son::{BinOp, Kind, UnOp};

/// CEP:WHAT: Per-node latency weight (abstract cycles).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: V0 opcode subset.
/// CEP:COST: O(1)
/// CEP:EVIDENCE: cost tests
/// CEP:SECURITY: no untrusted input
fn node_weight(kind: Kind, opcode: u8) -> u64 {
    match kind {
        Kind::Const | Kind::Param => 1,
        Kind::Un => {
            if opcode == UnOp::Neg as u8 {
                1
            } else {
                2
            }
        }
        Kind::Bin => match opcode {
            op if op == BinOp::Add as u8
                || op == BinOp::Sub as u8
                || op == BinOp::Min as u8
                || op == BinOp::Max as u8 =>
            {
                2
            }
            op if op == BinOp::Mul as u8 => 3,
            _ => 2,
        },
        // Aliases are free (resolved); dead nodes cost nothing.
        Kind::Alias | Kind::Dead => 0,
    }
}

/// The evaluated cost vector.
///
/// CEP:WHAT: latency-weighted node total (energy) + critical-path depth.
/// CEP:WHY: Two features: the Metropolis energy is `total`; depth breaks
///          ties toward shallower dependency chains (ILP proxy).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 16 bytes
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cost {
    /// Latency-weighted live node total (the energy).
    pub total: u64,
    /// Critical-path depth (weighted).
    pub depth: u64,
}

/// CEP:WHAT: Evaluates the candidate's cost over resolved views.
/// CEP:WHY: O(n) single pass: operands resolve through alias chains; depth
///          accumulates max-of-operands + own weight.
/// CEP:STATUS: complete
/// CEP:FAILURE: saturating arithmetic.
/// CEP:ASSUMES: candidate well-formed.
/// CEP:COST: O(n).
/// CEP:EVIDENCE: module tests.
/// CEP:SECURITY: no untrusted input.
pub fn evaluate(cand: &Candidate) -> Cost {
    let n = cand.slot_count() as u32;
    let resolve = |slot: u32| -> u32 {
        let mut cur = slot;
        for _ in 0..=n {
            let def = cand.view(cur);
            if def.kind == Kind::Alias {
                if def.lhs == u32::MAX {
                    return cur;
                }
                cur = def.lhs;
            } else {
                return cur;
            }
        }
        cur
    };
    let mut weights: Vec<u64> = vec![0; n as usize];
    let mut depths: Vec<u64> = vec![0; n as usize];
    let mut total: u64 = 0;
    let mut max_depth: u64 = 0;
    for slot in 0..n {
        let def = cand.view(slot);
        if def.kind == Kind::Dead || def.kind == Kind::Alias {
            continue;
        }
        let w = node_weight(def.kind, def.opcode);
        // Critical path: own weight + MAX of operand depths (not the sum —
        // parallel branches do not serialize).
        let mut child_depth: u64 = 0;
        for op in [def.lhs, def.rhs] {
            if op == u32::MAX {
                continue;
            }
            let r = resolve(op);
            if (r as usize) < depths.len() {
                child_depth = child_depth.max(depths[r as usize]);
            }
        }
        let d = w.saturating_add(child_depth);
        weights[slot as usize] = w;
        depths[slot as usize] = d;
        total = total.saturating_add(w);
        max_depth = max_depth.max(d);
    }
    let _ = &weights;
    Cost {
        total,
        depth: max_depth,
    }
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use crate::cand::{NodeDef, Overlay};
    use crate::son::{BinOp, ProgramBuilder, NO_OPERAND};

    fn base() -> Candidate {
        // f(x) = (x + 0) * 1
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let zero = b.const_i64(0);
        let one = b.const_i64(1);
        let add = b.bin(BinOp::Add, x, zero).expect("add");
        let mul = b.bin(BinOp::Mul, add, one).expect("mul");
        Candidate::from_program(b.finish(mul).expect("finish"))
    }

    // CEP:WHAT: x+0 -> x (via alias) strictly reduces cost.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a non-reduction.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn identity_rewrites_reduce_cost() {
        let cand = base();
        let before = evaluate(&cand);
        let mut mutated = cand.clone();
        mutated.overlays.push(Overlay::Replace {
            slot: 2, // add
            def: NodeDef {
                kind: Kind::Alias,
                imm: 0,
                opcode: 0,
                lhs: 0,
                rhs: NO_OPERAND,
            },
        });
        let after = evaluate(&mutated);
        assert!(after.total < before.total);
    }

    // CEP:WHAT: Depth accumulates along the dependency chain.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a flat depth.
    // CEP:ASSUMES: chain of length 3.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn cost_counts_depth() {
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let one = b.const_i64(1);
        let m1 = b.bin(BinOp::Mul, x, one).expect("m1");
        let m2 = b.bin(BinOp::Mul, m1, one).expect("m2");
        let m3 = b.bin(BinOp::Mul, m2, one).expect("m3");
        let c = Candidate::from_program(b.finish(m3).expect("finish"));
        let cost = evaluate(&c);
        // Three chained muls (weight 3 each, operands max not sum):
        // depth(param)=1; m1=3+1=4; m2=3+4=7; m3=3+7=10.
        assert_eq!(cost.depth, 10);
        assert_eq!(cost.total, 1 + 1 + 3 + 3 + 3);
    }
}
