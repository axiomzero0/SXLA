// CEP:FILE: crates/mss/src/mutate.rs
// CEP:WHAT: The mutation engine (V0 family set): local algebraic rewrites
//           (x+0, x*1, x-x, constant folding, commutative normalization),
//           reassociation (x+c1)+c2 -> x+(c1+c2), CSE-lite over resolved
//           views, and DCE (tombstoning unreachable slots). Each operator is
//           a PURE function (candidate + rng) -> Option<Candidate>; every
//           rewrite carries its Tier-1 semantic precondition (wrapping-i64
//           checked arithmetic where folding can overflow).
// CEP:WHY: Design §5: many independent mutation families dispatched by a
//          biased sampler; mutations are pure delta producers, never in-place
//           edits of shared state. The V0 set targets the pure-dataflow
//          dialect; structural/control/memory families are CEP-31 V2+.
// CEP:CLASS: CEP-31
// CEP:STATUS: complete (V0 family set)
// CEP:FAILURE: operators return None when no applicable site exists (the
//              caller retries); precondition failures return None (loud skip,
//              never an unsound rewrite).
// CEP:ASSUMES: candidate overlays well-formed.
// CEP:COST: O(n) site scan + O(overlays) clone per application.
// CEP:EVIDENCE: tests `identity_add_rewrites_to_alias`,
//           `const_fold_respects_overflow`, `reassoc_folds_constants`,
//           `cse_dedups_twins`, `dce_tombstones_unreachable`.
// CEP:SECURITY: no untrusted input.
//! Mutation operators (pure functions over candidates).

use crate::cand::{Candidate, NodeDef, Overlay};
use crate::rng::Rng;
use crate::son::{BinOp, Kind, NO_OPERAND};

/// CEP:WHAT: The mutation family selector (cost-biased sampler, V0: uniform
///           over the family table with deterministic rng dispatch).
/// CEP:WHY: Design §5.9 / §12: dispatch is sampled (not exhaustive) so the
///          search explores; Thompson-sampling/UCB effectiveness histograms
///          are the CEP-31 V1 learning layer.
/// CEP:STATUS: complete (V0 uniform sampler)
/// CEP:FAILURE: None when no family applies (caller retries).
/// CEP:ASSUMES: none
/// CEP:COST: O(n) per application
/// CEP:EVIDENCE: search tests
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationFamily {
    /// x + 0 -> x (and 0 + x -> x).
    IdentityAdd,
    /// x * 1 -> x (and 1 * x -> x).
    IdentityMul,
    /// x - x -> 0.
    SubSelf,
    /// const <op> const -> const (checked; skips on overflow).
    ConstFold,
    /// Commutative operand normalization (canonical orientation).
    CommuteNorm,
    /// (x + c1) + c2 -> x + (c1 + c2) (checked).
    ReassocAdd,
    /// Duplicate value nodes dedup to aliases.
    Cse,
    /// Unreachable slots tombstoned.
    Dce,
}

/// All V0 families (deterministic order).
pub const FAMILIES: [MutationFamily; 8] = [
    MutationFamily::IdentityAdd,
    MutationFamily::IdentityMul,
    MutationFamily::SubSelf,
    MutationFamily::ConstFold,
    MutationFamily::CommuteNorm,
    MutationFamily::ReassocAdd,
    MutationFamily::Cse,
    MutationFamily::Dce,
];

/// CEP:WHAT: Picks a random live, non-Alias slot.
/// CEP:STATUS: complete
/// CEP:FAILURE: None when no such slot exists.
/// CEP:ASSUMES: none
/// CEP:COST: O(n)
/// CEP:EVIDENCE: mutation tests
/// CEP:SECURITY: no untrusted input
fn random_live_slot(cand: &Candidate, rng: &mut Rng) -> Option<u32> {
    let n = cand.slot_count();
    let live: Vec<u32> = (0..n as u32)
        .filter(|&s| {
            let k = cand.view(s).kind;
            k != Kind::Dead && k != Kind::Alias
        })
        .collect();
    if live.is_empty() {
        return None;
    }
    Some(live[rng.below(live.len() as u64) as usize])
}

fn alias_def(target: u32) -> NodeDef {
    NodeDef {
        kind: Kind::Alias,
        imm: 0,
        opcode: 0,
        lhs: target,
        rhs: NO_OPERAND,
    }
}

fn const_def(value: i64) -> NodeDef {
    NodeDef {
        kind: Kind::Const,
        imm: value,
        opcode: 0,
        lhs: NO_OPERAND,
        rhs: NO_OPERAND,
    }
}

/// CEP:WHAT: Resolves a slot through alias chains (bounded).
/// CEP:STATUS: complete
/// CEP:FAILURE: returns the input slot on a broken chain.
/// CEP:ASSUMES: none
/// CEP:COST: O(chain)
/// CEP:EVIDENCE: tests
/// CEP:SECURITY: no untrusted input
fn resolve(cand: &Candidate, slot: u32) -> u32 {
    let n = cand.slot_count() as u32;
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
}

/// CEP:WHAT: Applies one mutation family to the candidate.
/// CEP:WHY: Pure function: clone + overlay append; None when inapplicable
///          or when the Tier-1 precondition fails (overflow checks).
/// CEP:STATUS: complete
/// CEP:FAILURE: None (caller retries with a different site/family).
/// CEP:ASSUMES: candidate well-formed.
/// CEP:COST: O(n) scan + O(overlays) clone.
/// CEP:EVIDENCE: module tests.
/// CEP:SECURITY: no untrusted input.
pub fn apply(cand: &Candidate, family: MutationFamily, rng: &mut Rng) -> Option<Candidate> {
    match family {
        MutationFamily::IdentityAdd => {
            let slot = random_live_slot(cand, rng)?;
            let def = cand.view(slot);
            if def.kind != Kind::Bin || def.opcode != BinOp::Add as u8 {
                return None;
            }
            let l = resolve(cand, def.lhs);
            let r = resolve(cand, def.rhs);
            let ld = cand.view(l);
            let rd = cand.view(r);
            // x + 0 / 0 + x -> x (const side must BE 0; params are unknown).
            if ld.kind == Kind::Const && ld.imm == 0 {
                let mut out = cand.clone();
                out.overlays.push(Overlay::Replace {
                    slot,
                    def: alias_def(r),
                });
                return Some(out);
            }
            if rd.kind == Kind::Const && rd.imm == 0 {
                let mut out = cand.clone();
                out.overlays.push(Overlay::Replace {
                    slot,
                    def: alias_def(l),
                });
                return Some(out);
            }
            None
        }
        MutationFamily::IdentityMul => {
            let slot = random_live_slot(cand, rng)?;
            let def = cand.view(slot);
            if def.kind != Kind::Bin || def.opcode != BinOp::Mul as u8 {
                return None;
            }
            let l = resolve(cand, def.lhs);
            let r = resolve(cand, def.rhs);
            let ld = cand.view(l);
            let rd = cand.view(r);
            if ld.kind == Kind::Const && ld.imm == 1 {
                let mut out = cand.clone();
                out.overlays.push(Overlay::Replace {
                    slot,
                    def: alias_def(r),
                });
                return Some(out);
            }
            if rd.kind == Kind::Const && rd.imm == 1 {
                let mut out = cand.clone();
                out.overlays.push(Overlay::Replace {
                    slot,
                    def: alias_def(l),
                });
                return Some(out);
            }
            None
        }
        MutationFamily::SubSelf => {
            let slot = random_live_slot(cand, rng)?;
            let def = cand.view(slot);
            if def.kind != Kind::Bin || def.opcode != BinOp::Sub as u8 {
                return None;
            }
            let l = resolve(cand, def.lhs);
            let r = resolve(cand, def.rhs);
            if l == r {
                let mut out = cand.clone();
                out.overlays.push(Overlay::Replace {
                    slot,
                    def: const_def(0),
                });
                return Some(out);
            }
            None
        }
        MutationFamily::ConstFold => {
            let slot = random_live_slot(cand, rng)?;
            let def = cand.view(slot);
            if def.kind != Kind::Bin {
                return None;
            }
            let l = resolve(cand, def.lhs);
            let r = resolve(cand, def.rhs);
            let ld = cand.view(l);
            let rd = cand.view(r);
            if ld.kind != Kind::Const || rd.kind != Kind::Const {
                return None;
            }
            // Tier-1 precondition: checked arithmetic — wrapping would NOT
            // preserve semantics under a fold-to-constant (the interpreter
            // wraps; folding with checked arithmetic keeps the fold EXACT:
            // wrapping(c1 op c2) == checked(c1 op c2) only when no overflow,
            // and when it overflows we simply skip the fold).
            let folded = match def.opcode {
                op if op == BinOp::Add as u8 => ld.imm.checked_add(rd.imm),
                op if op == BinOp::Sub as u8 => ld.imm.checked_sub(rd.imm),
                op if op == BinOp::Mul as u8 => ld.imm.checked_mul(rd.imm),
                op if op == BinOp::Min as u8 => Some(ld.imm.min(rd.imm)),
                op if op == BinOp::Max as u8 => Some(ld.imm.max(rd.imm)),
                _ => None,
            }?;
            let mut out = cand.clone();
            out.overlays.push(Overlay::Replace {
                slot,
                def: const_def(folded),
            });
            Some(out)
        }
        MutationFamily::CommuteNorm => {
            let slot = random_live_slot(cand, rng)?;
            let def = cand.view(slot);
            if def.kind != Kind::Bin {
                return None;
            }
            let commutative = matches!(
                def.opcode,
                op if op == BinOp::Add as u8
                    || op == BinOp::Mul as u8
                    || op == BinOp::Min as u8
                    || op == BinOp::Max as u8
            );
            if !commutative {
                return None;
            }
            let l = resolve(cand, def.lhs);
            let r = resolve(cand, def.rhs);
            // Canonical orientation: (kind, imm, slot) key ascending.
            let key = |s: u32| {
                let d = cand.view(s);
                (d.kind as u8, d.imm, s)
            };
            if key(l) <= key(r) {
                return None; // already canonical — no-op mutations are wasted
            }
            let mut out = cand.clone();
            out.overlays.push(Overlay::Replace {
                slot,
                def: NodeDef {
                    kind: def.kind,
                    imm: def.imm,
                    opcode: def.opcode,
                    lhs: def.rhs,
                    rhs: def.lhs,
                },
            });
            Some(out)
        }
        MutationFamily::ReassocAdd => {
            let slot = random_live_slot(cand, rng)?;
            let def = cand.view(slot);
            if def.kind != Kind::Bin || def.opcode != BinOp::Add as u8 {
                return None;
            }
            let l = resolve(cand, def.lhs);
            let r = resolve(cand, def.rhs);
            let ld = cand.view(l);
            if ld.kind != Kind::Bin || ld.opcode != BinOp::Add as u8 {
                return None;
            }
            let inner_l = resolve(cand, ld.lhs);
            let inner_r = resolve(cand, ld.rhs);
            let ild = cand.view(inner_l);
            let ird = cand.view(inner_r);
            let rd = cand.view(r);
            // (x + c1) + c2 with c1, c2 constant (either side of the inner).
            let (x, c1, c2) = if ird.kind == Kind::Const && rd.kind == Kind::Const {
                (inner_l, ird.imm, rd.imm)
            } else if ild.kind == Kind::Const && rd.kind == Kind::Const {
                (inner_r, ild.imm, rd.imm)
            } else {
                return None;
            };
            // Tier-1: checked fold of c1 + c2.
            let c = c1.checked_add(c2)?;
            let mut out = cand.clone();
            // New const node c; the outer add rewires to (x, c).
            out.overlays.push(Overlay::Extend { def: const_def(c) });
            let new_const = (out.slot_count() - 1) as u32;
            out.overlays.push(Overlay::Replace {
                slot,
                def: NodeDef {
                    kind: Kind::Bin,
                    imm: 0,
                    opcode: BinOp::Add as u8,
                    lhs: x,
                    rhs: new_const,
                },
            });
            Some(out)
        }
        MutationFamily::Cse => {
            // Find two live value nodes with identical resolved definitions.
            let n = cand.slot_count() as u32;
            let mut defs: Vec<(u32, u8, u8, i64, u32, u32)> = Vec::new();
            for s in 0..n {
                let d = cand.view(s);
                if d.kind != Kind::Const
                    && d.kind != Kind::Param
                    && d.kind != Kind::Bin
                    && d.kind != Kind::Un
                {
                    continue;
                }
                let l = if d.lhs == NO_OPERAND {
                    NO_OPERAND
                } else {
                    resolve(cand, d.lhs)
                };
                let r = if d.rhs == NO_OPERAND {
                    NO_OPERAND
                } else {
                    resolve(cand, d.rhs)
                };
                defs.push((s, d.kind as u8, d.opcode, d.imm, l, r));
            }
            defs.sort_by_key(|t| (t.1, t.2, t.3, t.4, t.5));
            for w in defs.windows(2) {
                let (a, b) = (w[0], w[1]);
                if (a.1, a.2, a.3, a.4, a.5) == (b.1, b.2, b.3, b.4, b.5) {
                    let mut out = cand.clone();
                    out.overlays.push(Overlay::Replace {
                        slot: b.0,
                        def: alias_def(a.0),
                    });
                    return Some(out);
                }
            }
            None
        }
        MutationFamily::Dce => {
            // Mark reachability from the root over resolved operands.
            let n = cand.slot_count() as u32;
            let mut reachable: Vec<bool> = vec![false; n as usize];
            // Worklist (deterministic order).
            let mut stack = vec![resolve(cand, cand.root)];
            while let Some(s) = stack.pop() {
                if s == NO_OPERAND || s as usize >= reachable.len() || reachable[s as usize] {
                    continue;
                }
                reachable[s as usize] = true;
                let d = cand.view(s);
                for op in [d.lhs, d.rhs] {
                    if op == NO_OPERAND {
                        continue;
                    }
                    stack.push(resolve(cand, op));
                }
            }
            // Tombstone the FIRST unreachable live node (deterministic scan).
            for s in 0..n {
                let d = cand.view(s);
                if !reachable[s as usize] && d.kind != Kind::Dead && d.kind != Kind::Alias {
                    let mut out = cand.clone();
                    out.overlays.push(Overlay::Replace {
                        slot: s,
                        def: NodeDef {
                            kind: Kind::Dead,
                            imm: 0,
                            opcode: 0,
                            lhs: NO_OPERAND,
                            rhs: NO_OPERAND,
                        },
                    });
                    return Some(out);
                }
            }
            None
        }
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
    use crate::son::{ProgramBuilder, UnOp};

    fn cand_x_plus_zero_times_one() -> Candidate {
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let zero = b.const_i64(0);
        let one = b.const_i64(1);
        let add = b.bin(BinOp::Add, x, zero).expect("add");
        let mul = b.bin(BinOp::Mul, add, one).expect("mul");
        Candidate::from_program(b.finish(mul).expect("finish"))
    }

    // CEP:WHAT: x+0 rewrites to Alias(x).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires when the family misses the site.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn identity_add_rewrites_to_alias() {
        let cand = cand_x_plus_zero_times_one();
        let mut rng = Rng::new(1);
        // The family targets a random live slot; retry until it hits add.
        let mut found = false;
        for _ in 0..200 {
            if let Some(out) = apply(&cand, MutationFamily::IdentityAdd, &mut rng) {
                // Slots: 0=x, 1=zero, 2=one, 3=add, 4=mul — the family
                // tombstones-by-alias the ADD (slot 3).
                assert_eq!(out.view(3).kind, Kind::Alias);
                assert_eq!(crate::interp::evaluate(&out.materialize(), &[41]), Ok(41));
                found = true;
                break;
            }
        }
        assert!(found, "family never fired");
    }

    // CEP:WHAT: Constant folding respects overflow (i64::MAX + 1 skipped).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on an unsound fold.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn const_fold_respects_overflow() {
        let mut b = ProgramBuilder::new();
        let mx = b.const_i64(i64::MAX);
        let one = b.const_i64(1);
        let add = b.bin(BinOp::Add, mx, one).expect("add");
        let cand = Candidate::from_program(b.finish(add).expect("finish"));
        // Interpreter: wrapping -> MIN.
        assert_eq!(
            crate::interp::evaluate(&cand.materialize(), &[]),
            Ok(i64::MIN)
        );
        // The fold MUST refuse (checked overflow) — exhaustively try all slots.
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let applied = apply(&cand, MutationFamily::ConstFold, &mut rng);
            assert!(
                applied.is_none(),
                "unsound fold applied: {:?}",
                applied.map(|o| o.overlays)
            );
        }
    }

    // CEP:WHAT: (x+2)+3 reassociates to x+5.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a wrong constant.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn reassoc_folds_constants() {
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let c1 = b.const_i64(2);
        let c2 = b.const_i64(3);
        let inner = b.bin(BinOp::Add, x, c1).expect("inner");
        let outer = b.bin(BinOp::Add, inner, c2).expect("outer");
        let cand = Candidate::from_program(b.finish(outer).expect("finish"));
        let mut rng = Rng::new(5);
        let mut done = false;
        for _ in 0..200 {
            if let Some(out) = apply(&cand, MutationFamily::ReassocAdd, &mut rng) {
                let m = out.materialize();
                assert_eq!(crate::interp::evaluate(&m, &[10]), Ok(15));
                // The appended const is 5 (the folded 2+3).
                let ext = out.slot_count() - 1;
                assert_eq!(out.view(ext as u32).imm, 5);
                done = true;
                break;
            }
        }
        assert!(done, "reassoc never fired");
    }

    // CEP:WHAT: CSE dedups twin value nodes.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires when twins survive.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn cse_dedups_twins() {
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let y = b.param(1);
        let m1 = b.bin(BinOp::Mul, x, y).expect("m1");
        let m2 = b.bin(BinOp::Mul, x, y).expect("m2");
        let add = b.bin(BinOp::Add, m1, m2).expect("add");
        let cand = Candidate::from_program(b.finish(add).expect("finish"));
        let out = apply(&cand, MutationFamily::Cse, &mut Rng::new(9))
            .expect("cse applies deterministically");
        // The later twin (slot 3) aliases the earlier (slot 2).
        assert_eq!(out.view(3).kind, Kind::Alias);
        assert_eq!(out.view(3).lhs, 2);
        // Values preserved: x*y + x*y == 2xy.
        assert_eq!(crate::interp::evaluate(&out.materialize(), &[3, 4]), Ok(24));
    }

    // CEP:WHAT: DCE tombstones unreachable nodes.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a missed tombstone.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn dce_tombstones_unreachable() {
        // Root is neg(x); a dead mul chain hangs unused.
        let mut b = ProgramBuilder::new();
        let x = b.param(0);
        let one = b.const_i64(1);
        let _dead_mul = b.bin(BinOp::Mul, x, one).expect("dead");
        let neg = b.un(UnOp::Neg, x).expect("neg");
        let cand = Candidate::from_program(b.finish(neg).expect("finish"));
        let out = apply(&cand, MutationFamily::Dce, &mut Rng::new(2))
            .expect("dce applies deterministically");
        // Slots: 0=x, 1=one, 2=dead-mul, 3=neg(root). The FIRST unreachable
        // live node in slot order is the const ONE (slot 1).
        assert_eq!(out.view(1).kind, Kind::Dead);
        assert_eq!(crate::interp::evaluate(&out.materialize(), &[8]), Ok(-8));
    }
}
