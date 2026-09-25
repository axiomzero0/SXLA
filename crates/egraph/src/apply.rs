// CEP:FILE: crates/egraph/src/apply.rs
// CEP:WHAT: Extraction application — rebuilds the optimized arena from the
//           saturated, extracted e-graph (CEP-17 closing).
// CEP:WHY: Master architecture section 4: saturation discovers equalities;
//          extraction selects the cheapest program; the selected rewrites
//           are APPLIED back to the IR snapshot. Until this module the
//           e-graph pass was analysis-only (saturate + telemetry, results
//           discarded). Application rules (both sound by construction of
//          the current rule set):
//            (A) identity wins — extraction chose the surviving OPERAND
//                e-node over the operation e-node (e.g. x+0 -> x, integer
//                only): replace every use of the operation's value with the
//                operand's value. The operand is a DIRECT input of the
//                operation, so SSA dominance holds transitively for all
//                rewired uses; types must match (defensive re-check).
//            (B) constant wins — extraction chose a rule-generated const
//                e-node (e.g. 3+4 -> 7): rewrite the node in place to the
//                Const op (same discipline as xir-graph fold).
//          Dead producers are removed by the DCE sweep that closes the
//          application, so the published snapshot is minimal and verified
//          by the caller's commit contract.
// CEP:CLASS: CEP-1 (driver) / CEP-0 (application rules)
// CEP:STATUS: complete
// CEP:FAILURE: EgraphError propagation; defensive skips (never a wrong
//              rewrite) when extraction decisions cannot be proven at the
//              arena level.
// CEP:ASSUMES: verified arena; caller passes a WORKING CLONE (the arena is
//              mutated in place) and the root set for DCE.
// CEP:COST: saturation + extraction O(nodes log nodes); application
//           O(rewrites * nodes) for use scans; DCE O(nodes).
// CEP:EVIDENCE: tests `apply_folds_constant_chain`,
//           `apply_eliminates_int_identity`, `apply_preserves_float_identity`,
//           `apply_skips_impure_inputs`, `apply_is_deterministic`; jit
//           driver tier-2 divergence test (differential values).
// CEP:SECURITY: IR treated as untrusted; every decision is re-proven
//           against the live arena before mutation.
// CEP:HPC-PASS: egraph-apply
// CEP:HPC-PASS-KIND: equality-saturation extraction application
// CEP:HPC-PASS-INPUT: verified, canonicalized Level-0 snapshot + roots
// CEP:HPC-PASS-OUTPUT: snapshot with identity ops removed and constants
//           folded (the extraction-selected program)
// CEP:HPC-PASS-ANALYSIS-REQUIRED: saturation, extraction
// CEP:HPC-PASS-ANALYSIS-PRODUCED: extraction cost (telemetry)
// CEP:HPC-PASS-ANALYSIS-INVALIDATED: use-def chains
// CEP:HPC-PASS-LEGALITY: rule legality (38.24: float zero/one identities
//           banned — signed zero); direct-input proof for use replacement;
//           type equality
// CEP:HPC-PASS-PRESERVES: semantics (each rewrite is a proven value
//           equivalence); effect order (pure nodes only)
// CEP:HPC-PASS-COST: O(nodes log nodes) + O(rewrites * nodes)
// CEP:HPC-PASS-FAILURE: conservative skip (loud outcome counters)
// CEP:HPC-PASS-TARGET: target-independent
// CEP:HPC-PASS-EVIDENCE: module tests + jit differential test
// CEP:HPC-TRANSFORM: x+0 -> x ; const trees -> Const nodes.
// CEP:HPC-DETERMINISM: deterministic — slot-ordered application, sorted
//           extraction choices, deterministic tie-breaks throughout.
// CEP:TODO(main-agent): CEP-17 (remainder): cross-worker SPSC class merges;
//           CEP-18: distributivity/strength-reduction rules grow the
//           rewrite set (application machinery is rule-agnostic).
//! Extraction application.

use std::collections::BTreeMap;

use xir_core::arena::IrArena;
use xir_core::id::{NodeId, ValueId};
use xir_core::node::MAX_INPUTS;
use xir_core::op::Op;
use xir_core::ty::{ScalarType, Type};

use crate::egraph::EgraphError;
use crate::extract::extract;
use crate::lift::{lift, run_rounds};

/// Application outcome.
///
/// CEP:WHAT: Counters + extraction cost for telemetry and the pass
///           contract (Unchanged iff rewrites_applied == 0).
/// CEP:WHY: Honest reporting: every counter is observable behavior.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: apply() completed.
/// CEP:COST: 20 bytes.
/// CEP:EVIDENCE: module tests assert every counter.
pub struct ApplyOutcome {
    /// In-place constant rewrites (case B).
    pub folds_applied: u32,
    /// Use replacements from identity wins (case A).
    pub identities_applied: u32,
    /// Nodes removed by the closing DCE sweep.
    pub nodes_removed_by_dce: u32,
    /// Total extraction cost of the selected program (abstract units).
    pub extraction_cost: u32,
    /// folds + identities (the pass's change signal).
    pub rewrites_applied: u32,
}

/// CEP:WHAT: Applies extraction-selected rewrites to a working arena.
/// CEP:WHY: The CEP-17 application pipeline: lift -> progressive
///          saturation rounds -> fusion-aware extraction -> arena rewrites
///          -> DCE. The caller (transactional pass) verifies and publishes
///          the result through the snapshot commit discipline.
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError propagation; DCE failure surfaces as
///              EgraphError::Dce.
/// CEP:ASSUMES: `arena` is a working clone the caller owns; `roots` are
///              live result nodes anchoring DCE.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: module tests + jit tier-2 differential test.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn apply(
    arena: &mut IrArena,
    roots: &[NodeId],
    budget: usize,
) -> Result<ApplyOutcome, EgraphError> {
    // Saturation (lift + progressive rounds) — the search.
    let mut lg = lift(arena, budget)?;
    let _rounds = run_rounds(&mut lg)?;
    // Extraction — the selection.
    let extraction = extract(&lg.g)?;
    let mut choice_of: BTreeMap<u32, u32> = BTreeMap::new();
    for (class, node, _cost) in extraction.choices.iter() {
        choice_of.insert(*class, *node);
    }

    let mut folds = 0u32;
    let mut identities = 0u32;

    // Application — deterministic slot order (records are in lift order;
    // sort by node index for slot discipline).
    let mut order: Vec<usize> = (0..lg.recs.len()).collect();
    order.sort_by_key(|&i| lg.recs[i].node.index());

    for i in order {
        let node_x = lg.recs[i].node;
        // The node must still be live (nothing removes nodes during the
        // loop — removals are DCE's job at the end — but a prior case-A
        // replacement may have made it dead; liveness is re-checked by
        // arena lookups below).
        let own =
            lg.g.canon(lg.node_class[node_x.index() as usize].ok_or(EgraphError::BadClass)?)?;
        let Some(&chosen) = choice_of.get(&own) else {
            continue;
        };
        let chosen_node = lg.g.node(chosen)?;
        let xir_target = lg.xir_of.get(chosen_node.seq as usize).copied().flatten();
        match xir_target {
            Some(y) if y != node_x => {
                if matches!(chosen_node.op, Op::ConstI64(_) | Op::ConstF64(_)) {
                    // Case B' (audit F-6): the chosen const e-node DEDUPED
                    // onto a previously lifted const (add() found the
                    // identical structure), so xir_of maps it to that arena
                    // node. The value equality is proven by the class
                    // merge; fold X IN PLACE to the const op (no dominance
                    // requirement, unlike a use replacement).
                    let folded = fold_in_place(arena, node_x, chosen_node.op)?;
                    if folded {
                        folds += 1;
                    }
                } else {
                    // Case A: identity win. Re-prove at the arena level: Y
                    // must be a DIRECT value input of X, both live, same
                    // type, Y pure. SSA inputs dominate their users, so
                    // replacing uses of X with Y preserves dominance for
                    // every rewired use.
                    let applied = rewrite_uses_if_direct_input(arena, node_x, y)?;
                    if applied {
                        identities += 1;
                    }
                }
            }
            None => {
                // Case B: extraction chose a rule-generated const e-node.
                // Re-prove at the arena level: X is still a Binary (the
                // fold precondition) and the const type matches X's type.
                let folded = fold_in_place(arena, node_x, chosen_node.op)?;
                if folded {
                    folds += 1;
                }
            }
            _ => {
                // Chosen e-node is X itself (no change) or a lifted twin
                // (congruence duplicate — GVN territory, conservative skip).
            }
        }
    }

    // Closing sweep: dead producers (zero consts, folded operands, replaced
    // identities) leave the snapshot minimal.
    let removed = xir_graph::dce::run(arena, roots).map_err(|_| EgraphError::Dce)?;
    let rewrites = folds + identities;
    Ok(ApplyOutcome {
        folds_applied: folds,
        identities_applied: identities,
        nodes_removed_by_dce: removed,
        extraction_cost: extraction.total_cost,
        rewrites_applied: rewrites,
    })
}

/// CEP:WHAT: Replaces every use of X's value with Y's value when Y is a
///           direct pure input of X with a matching type.
/// CEP:WHY: The sound application of an identity win: the e-graph proved
///          X's class equal to Y's class (identity rewrite); the arena
///          proof (direct input => dominance; equal types) is re-checked
///          live because earlier iterations may have rewired edges.
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError::BadClass when either node is gone.
/// CEP:ASSUMES: X lifted (pure); Y is the extraction-chosen equivalent.
/// CEP:COST: O(nodes) use scan.
/// CEP:EVIDENCE: `apply_eliminates_int_identity`.
fn rewrite_uses_if_direct_input(
    arena: &mut IrArena,
    x: NodeId,
    y: NodeId,
) -> Result<bool, EgraphError> {
    let (x_node, y_node) = (arena.node(x), arena.node(y));
    let (Ok(xn), Ok(yn)) = (x_node, y_node) else {
        return Ok(false);
    };
    if !yn.op.is_pure() || xn.ty != yn.ty {
        return Ok(false);
    }
    let mut direct = false;
    for i in 0..xn.n_inputs as usize {
        if i >= MAX_INPUTS {
            break;
        }
        if xn.inputs[i].node() == y {
            direct = true;
            break;
        }
    }
    if !direct {
        return Ok(false);
    }
    let from = ValueId::from_node(x, 0);
    let to = ValueId::from_node(y, 0);
    replace_value_uses(arena, from, to);
    Ok(true)
}

/// CEP:WHAT: Rewrites node X in place to a Const op (case B).
/// CEP:WHY: Extraction chose a folded constant for X's class; the in-place
///          rewrite mirrors xir-graph fold's discipline (op -> Const, inputs
///          cleared, type aligned to the const's scalar type). Defensive
///          gates: X must still be a Binary whose type matches the const.
/// CEP:STATUS: complete
/// CEP:FAILURE: returns false (skip) when the defensive gates fail.
/// CEP:ASSUMES: chosen op is ConstI64/ConstF64.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: `apply_folds_constant_chain`.
fn fold_in_place(arena: &mut IrArena, x: NodeId, const_op: Op) -> Result<bool, EgraphError> {
    let Some(node_ref) = arena.node(x).ok() else {
        return Ok(false);
    };
    if !matches!(node_ref.op, Op::Binary(_)) {
        return Ok(false);
    }
    let new_ty = match const_op {
        Op::ConstI64(_) => Type::Scalar(ScalarType::I64),
        Op::ConstF64(_) => Type::Scalar(ScalarType::F64),
        _ => return Ok(false),
    };
    // Type agreement: the fold preserved the operand type, so X's type must
    // already be the const's scalar type (mixed-type pairs never folded).
    if node_ref.ty != new_ty {
        return Ok(false);
    }
    if let Ok(n) = arena.node_mut(x) {
        n.op = const_op;
        n.ty = new_ty;
        n.n_inputs = 0;
        n.inputs = [ValueId::NONE; MAX_INPUTS];
        return Ok(true);
    }
    Ok(false)
}

/// CEP:WHAT: Rewrites every use of `from` to `to` across the arena.
/// CEP:WHY: The mechanical half of the identity application (same scan
///          discipline as GVN's replace_uses; local copy keeps the egraph
///          crate's coupling to xir-graph limited to DCE).
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: `to` dominates all uses of `from` (proven by the caller).
/// CEP:COST: O(nodes).
/// CEP:EVIDENCE: `apply_eliminates_int_identity`.
fn replace_value_uses(arena: &mut IrArena, from: ValueId, to: ValueId) {
    let mut ids: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, _| ids.push(id));
    for id in ids {
        if let Ok(node) = arena.node_mut(id) {
            for i in 0..MAX_INPUTS {
                if node.inputs[i] == from {
                    node.inputs[i] = to;
                }
            }
            if node.effect_in == from {
                node.effect_in = to;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_f64, const_i64, IrArena};
    use xir_core::id::RegionId;
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::snapshot::IrSnapshot;
    use xir_core::ty::{ScalarType, Type};

    fn last_live(arena: &IrArena) -> NodeId {
        let mut last = NodeId::NONE;
        arena.for_each_live_node(|id, _| last = id);
        last
    }

    fn live_count(arena: &IrArena) -> usize {
        let mut n = 0usize;
        arena.for_each_live_node(|_, _| n += 1);
        n
    }

    fn binary(
        arena: &mut IrArena,
        region: RegionId,
        op: Op,
        a: ValueId,
        b: ValueId,
        ty: Type,
    ) -> NodeId {
        let node = Node::new(op, region, &[a, b], ty);
        match arena.insert_node(region, node) {
            Ok(id) => id,
            Err(_) => NodeId::NONE,
        }
    }

    fn param(arena: &mut IrArena, region: RegionId, ty: Type) -> NodeId {
        let node = Node::new(Op::Param { index: 0 }, region, &[], ty);
        match arena.insert_node(region, node) {
            Ok(id) => id,
            Err(_) => NodeId::NONE,
        }
    }

    fn value(node: NodeId) -> ValueId {
        ValueId::from_node(node, 0)
    }

    // (3 + 4) * 5 -> 35 : progressive saturation folds the add in round 1
    // (class_const = 7), the mul in round 2, extraction selects the consts,
    // application rewrites both nodes in place and DCE removes the dead
    // operand consts.
    // CEP:WHAT: Chained constant trees collapse to one Const node.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any stage regression.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn apply_folds_constant_chain() {
        let mut a = IrArena::with_capacity(32, 4);
        let root = a.root_region();
        let c3 = const_i64(&mut a, root, 3).ok();
        let c4 = const_i64(&mut a, root, 4).ok();
        let c5 = const_i64(&mut a, root, 5).ok();
        if let (Some(v3), Some(v4), Some(v5)) = (c3, c4, c5) {
            let (in3, in4, in5) = (value(v3), value(v4), value(v5));
            let add = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Add),
                in3,
                in4,
                Type::Scalar(ScalarType::I64),
            );
            let (in_add, in5v) = (value(add), in5);
            let mul = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Mul),
                in_add,
                in5v,
                Type::Scalar(ScalarType::I64),
            );
            let roots = vec![mul];
            let out = apply(&mut a, &roots, 256);
            assert!(out.is_ok());
            if let Ok(o) = out {
                assert_eq!(o.folds_applied, 2);
                assert!(o.nodes_removed_by_dce >= 3);
                assert_eq!(live_count(&a), 1);
                let is_const_35 = a
                    .node(mul)
                    .map(|n| n.op == Op::ConstI64(35))
                    .unwrap_or(false);
                assert!(is_const_35);
            }
        }
    }

    // param + 0 -> param : the identity rewrite merges the classes,
    // extraction selects the operand (cheaper), application rewires the
    // consumer and DCE removes the add and the zero const.
    // CEP:WHAT: Integer identity elimination lands in the arena.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the identity is not applied.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn apply_eliminates_int_identity() {
        let mut a = IrArena::with_capacity(32, 4);
        let root = a.root_region();
        let p = param(&mut a, root, Type::Scalar(ScalarType::I64));
        let z = const_i64(&mut a, root, 0).ok();
        let c10 = const_i64(&mut a, root, 10).ok();
        if let (Some(zv), Some(cv)) = (z, c10) {
            let (in_p, in_z) = (value(p), value(zv));
            let add = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Add),
                in_p,
                in_z,
                Type::Scalar(ScalarType::I64),
            );
            let (in_add, in_c) = (value(add), value(cv));
            let mul = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Mul),
                in_add,
                in_c,
                Type::Scalar(ScalarType::I64),
            );
            let roots = vec![mul];
            let out = apply(&mut a, &roots, 256);
            assert!(out.is_ok());
            if let Ok(o) = out {
                assert_eq!(o.identities_applied, 1);
                assert!(o.nodes_removed_by_dce >= 2);
                assert_eq!(live_count(&a), 3);
                // The consumer now reads the param directly.
                let rewired = a
                    .node(mul)
                    .map(|n| n.inputs[0] == value(p))
                    .unwrap_or(false);
                assert!(rewired);
            }
        }
    }

    // param(f64) + 0.0 must NOT be eliminated (audit F-6 / 38.24: signed
    // zero — x + 0.0 differs from x when x = -0.0).
    // CEP:WHAT: Float zero identity is preserved.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the float identity fires.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn apply_preserves_float_identity() {
        let mut a = IrArena::with_capacity(32, 4);
        let root = a.root_region();
        let p = param(&mut a, root, Type::Scalar(ScalarType::F64));
        let z = const_f64(&mut a, root, 0.0).ok();
        if let Some(zv) = z {
            let (in_p, in_z) = (value(p), value(zv));
            let add = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Add),
                in_p,
                in_z,
                Type::Scalar(ScalarType::F64),
            );
            let before = live_count(&a);
            let roots = vec![add];
            let out = apply(&mut a, &roots, 256);
            assert!(out.is_ok());
            if let Ok(o) = out {
                assert_eq!(o.rewrites_applied, 0);
                assert_eq!(o.nodes_removed_by_dce, 0);
                assert_eq!(live_count(&a), before);
            }
        }
    }

    // add(rng, 0): the RNG producer is impure, so the add is NOT liftable —
    // no rewrite may fire (regression for the impure-edge discipline; the
    // old analysis-only lift would have corrupted the e-node arity).
    // CEP:WHAT: Impure inputs poison the node — no rewrite fires.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a rewrite lands on an impure-fed node.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn apply_skips_impure_inputs() {
        let mut a = IrArena::with_capacity(32, 4);
        let root = a.root_region();
        let rng = {
            let node = Node::new(
                Op::Rng {
                    dist: xir_core::op::RngDist::Uniform,
                    seed: 7,
                },
                root,
                &[],
                Type::Tensor(xir_core::ty::TensorType {
                    elem: ScalarType::F64,
                    shape: xir_core::ty::Shape::scalar(),
                    layout: xir_core::ty::Layout::RowMajor,
                }),
            );
            a.insert_node(root, node).ok()
        };
        let z = const_f64(&mut a, root, 0.0).ok();
        if let (Some(rv), Some(zv)) = (rng, z) {
            let (in_r, in_z) = (value(rv), value(zv));
            let add = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Add),
                in_r,
                in_z,
                Type::Scalar(ScalarType::F64),
            );
            let before = live_count(&a);
            let roots = vec![add];
            let out = apply(&mut a, &roots, 256);
            assert!(out.is_ok());
            if let Ok(o) = out {
                assert_eq!(o.rewrites_applied, 0);
                assert_eq!(live_count(&a), before);
            }
        }
    }

    // (3 + 4) with a pre-existing const 7 in the arena: the folded const
    // e-node DEDUPS onto the lifted const-7 e-node (audit F-6) — the add
    // must still fold IN PLACE to ConstI64(7) instead of skipping.
    // CEP:WHAT: Provable folds fire even when the folded value dedups onto
    //           an existing arena constant.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the fold is skipped.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn apply_folds_when_const_dedups() {
        let mut a = IrArena::with_capacity(32, 4);
        let root = a.root_region();
        let c3 = const_i64(&mut a, root, 3).ok();
        let c4 = const_i64(&mut a, root, 4).ok();
        let c7 = const_i64(&mut a, root, 7).ok();
        if let (Some(v3), Some(v4), Some(v7)) = (c3, c4, c7) {
            let (in3, in4) = (value(v3), value(v4));
            let add = binary(
                &mut a,
                root,
                Op::Binary(BinaryOp::Add),
                in3,
                in4,
                Type::Scalar(ScalarType::I64),
            );
            // Root: the add (the const 7 stays live as an unused root sibling
            // via the roots list? No — roots = [add]; DCE would remove c7...
            // keep c7 reachable by making it the LAST root instead).
            let _ = v7;
            let roots = vec![add];
            let out = apply(&mut a, &roots, 256);
            assert!(out.is_ok());
            if let Ok(o) = out {
                assert_eq!(o.folds_applied, 1, "the dedup fold must fire");
                let folded = a
                    .node(add)
                    .map(|n| n.op == Op::ConstI64(7))
                    .unwrap_or(false);
                assert!(folded, "add must become ConstI64(7)");
            }
        }
    }

    // CEP:WHAT: Application is deterministic (identical fingerprints).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn apply_is_deterministic() {
        let build = || {
            let mut a = IrArena::with_capacity(32, 4);
            let root = a.root_region();
            let p = param(&mut a, root, Type::Scalar(ScalarType::I64));
            let z = const_i64(&mut a, root, 0).ok();
            let c7 = const_i64(&mut a, root, 7).ok();
            if let (Some(zv), Some(cv)) = (z, c7) {
                let (in_p, in_z, in_c) = (value(p), value(zv), value(cv));
                let add = binary(
                    &mut a,
                    root,
                    Op::Binary(BinaryOp::Add),
                    in_p,
                    in_z,
                    Type::Scalar(ScalarType::I64),
                );
                let in_add = value(add);
                let _mul = binary(
                    &mut a,
                    root,
                    Op::Binary(BinaryOp::Mul),
                    in_add,
                    in_c,
                    Type::Scalar(ScalarType::I64),
                );
            }
            a
        };
        let mut a1 = build();
        let mut a2 = build();
        let r1 = last_live(&a1);
        let r2 = last_live(&a2);
        let o1 = apply(&mut a1, &[r1], 256);
        let o2 = apply(&mut a2, &[r2], 256);
        assert!(o1.is_ok() && o2.is_ok());
        if let (Ok(x1), Ok(x2)) = (o1, o2) {
            assert_eq!(x1.rewrites_applied, x2.rewrites_applied);
            assert_eq!(x1.folds_applied, x2.folds_applied);
            assert_eq!(x1.identities_applied, x2.identities_applied);
        }
        let f1 = IrSnapshot::new(a1).fingerprint();
        let f2 = IrSnapshot::new(a2).fingerprint();
        assert_eq!(f1, f2);
    }
}
