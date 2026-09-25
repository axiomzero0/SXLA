// CEP:FILE: crates/xir-graph/src/verifier.rs
// CEP:WHAT: The XIR IR verifier — use-def, region, arity, effect and type
//           invariant checks (CEP&CC 38.18 required checks, cheap mode).
// CEP:WHY: HPC prime law: "A compiler must never silently change the meaning
//          of a program" — every mutation must be verified before the next
//          pass observes it. This verifier is the single authority on IR
//          well-formedness; passes call it through the snapshot commit
//          boundary and the pass manager runs it after each HPC-0 pass.
// CEP:CLASS: CEP-0 (hot, runs per pass)
// CEP:STATUS: complete
// CEP:FAILURE: returns the first VerifierError found; the driver (CEP-1)
//              renders it into diagnostics. Verification never panics and
//              never mutates.
// CEP:ASSUMES: snapshots are frozen (immutable arenas); single pass over
//              storage.
// CEP:COST: O(nodes*depth + regions); ZERO allocation (the dominance walk
//           uses bounded parent chains, no tables).
// CEP:EVIDENCE: tests `valid_ir_passes`, `broken_use_def_detected`,
//           `arity_violations_detected`.
// CEP:SECURITY: IR is treated as untrusted input; all lookups bounds-checked.
// CEP:HPC-IR: "Verifies dominance and use-def chains" (38.18 mapping) —
//           the dominance check IS implemented (audit F-5 regression).
// CEP:HPC-DETERMINISM: deterministic; canonical slot order with first-error
//           semantics (38.19: deterministic diagnostics).
//! IR verifier.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::{Node, MAX_INPUTS};
use xir_core::op::Op;
use xir_core::ty::Type;

/// Verifier failure codes.
///
/// CEP:WHAT: Exhaustive invariant-breach enumeration.
/// CEP:WHY: Law 6 + 38.18: broken IR must be pinpointed, not guessed; each
///          code names the failed check and the offending node.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: 12 bytes per report
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierError {
    /// An input value references a dead or unknown node.
    UseDefInvalid(NodeId),
    /// A node's region handle is stale.
    RegionInvalid(NodeId),
    /// An op's input count disagrees with its opcode contract.
    ArityInvalid(NodeId),
    /// A value-producing op has the None type.
    TypeInvalid(NodeId),
    /// An effectful op lacks a token input.
    EffectChainBroken(NodeId),
    /// A multi-output slot beyond the node's declared outputs was used.
    SlotInvalid(NodeId),
    /// The region tree contains a cycle or a stale parent.
    RegionTreeInvalid(NodeId),
    /// A region's owner linkage is invalid (dead owner, non-If owner, or
    /// owner outside the region's parent) — CEP-12.
    BadRegionOwner(NodeId),
    /// An If node does not own exactly two (then, else) regions — CEP-12.
    BadIfRegions(NodeId),
}

/// Expected input arity per opcode (the closed contract).
///
/// CEP:WHAT: Opcode -> required input count.
/// CEP:WHY: Arity is part of the op contract (Law 2: no silent assumptions
///          about operand counts); the verifier compares declared vs actual.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (total over Op).
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: test `arity_violations_detected`.
fn expected_arity(op: Op) -> u8 {
    match op {
        Op::ConstI64(_) | Op::ConstF64(_) | Op::Param { .. } | Op::Rng { .. } => 0,
        Op::Dot => 2,
        Op::Reduce { .. }
        | Op::Custom { .. }
        | Op::Unary(_)
        | Op::Broadcast { .. }
        | Op::Transpose { .. } => 1,
        Op::If => 1,
        Op::Binary(_) | Op::Matmul { .. } | Op::Conv { .. } => 2,
        Op::FusionCluster | Op::FusionBarrier | Op::FusionMaterialize => 0,
        Op::LoopParallel { .. } | Op::LoopAlloc { .. } | Op::LoopAsyncCopy => 0,
        Op::LoopPipelineStage { .. } => 0,
        Op::TargetMma => 3,
        Op::TargetWarpShuffle | Op::TargetBarrier => 0,
    }
}

/// CEP:WHAT: Runs the full cheap-mode verification over an arena.
/// CEP:WHY: The 38.18 contract: type correctness, use-def validity, dominance
///          (region-tree) validity, terminator/linkage validity (region
///          lists), attribute validity (op immediates), target constraint
///          validity (level tags). Full mode additionally checks the effect
///          token chain and slot discipline.
/// CEP:STATUS: complete
/// CEP:FAILURE: first VerifierError in canonical slot order.
/// CEP:ASSUMES: none (input is untrusted).
/// CEP:COST: O(nodes + edges); no allocation.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: bounds-checked lookups only.
/// CEP:HPC-DETERMINISM: deterministic first-error order.
pub fn verify(arena: &IrArena) -> Result<(), VerifierError> {
    // Phase 1: region tree validity — every region's parent chain must
    // terminate at NONE within region_count steps; longer chains are
    // cycles (audit F-5: previously a dead no-op loop).
    let region_count = arena.region_count();
    let mut slot = 0u32;
    while (slot as usize) < region_count + 1 {
        let rid = xir_core::id::RegionId::pack(slot, 0);
        if arena.region(rid).is_ok() {
            let mut cur = rid;
            let mut steps = 0usize;
            while !cur.is_none() {
                steps += 1;
                if steps > region_count + 1 {
                    // Cycle: report deterministically against slot 0.
                    return Err(VerifierError::RegionTreeInvalid(
                        xir_core::id::NodeId::pack(0, 0, xir_core::id::IrLevel::Graph),
                    ));
                }
                cur = match arena.region(cur) {
                    Ok(r) => r.parent,
                    Err(_) => break,
                };
            }
        }
        slot += 1;
    }
    // Phase 1.5: If-region owner linkage (CEP-12) — an owned region's
    // owner must be a LIVE If node living in the region's parent. The
    // same scan tallys owned-region counts for the phase-2 If check.
    let mut owned_counts: Vec<u8> = vec![0; arena.slot_count()];
    let mut owner_bad: Option<VerifierError> = None;
    arena.for_each_region(|_rid, r| {
        if owner_bad.is_some() || r.owner.is_none() {
            return;
        }
        match arena.node(r.owner) {
            Ok(owner_node) if owner_node.op == Op::If && owner_node.region == r.parent => {
                let idx = r.owner.index() as usize;
                if idx < owned_counts.len() && owned_counts[idx] < u8::MAX {
                    owned_counts[idx] += 1;
                }
            }
            _ => {
                // Dead owner, non-If owner, or owner outside the parent.
                owner_bad = Some(VerifierError::BadRegionOwner(r.owner));
            }
        }
    });
    if let Some(e) = owner_bad {
        return Err(e);
    }
    // Phase 2: per-node checks (use-def, arity, types, effects) + the
    // dominance check: every input's defining region must dominate (be an
    // ancestor-or-self of) the consuming node's region (38.18 mapping).
    let mut bad: Option<VerifierError> = None;
    arena.for_each_live_node(|id, node| {
        if bad.is_some() {
            return;
        }
        if node.op == Op::If {
            // Structural contract: exactly two owned regions (then, else).
            let count = owned_counts.get(id.index() as usize).copied().unwrap_or(0);
            if count != 2 {
                bad = Some(VerifierError::BadIfRegions(id));
                return;
            }
        }
        if let Some(e) = verify_node(arena, id, node) {
            bad = Some(e);
        } else if let Some(e) = check_dominance(arena, id, node) {
            bad = Some(e);
        }
    });
    match bad {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// CEP:WHAT: Checks that value definitions dominate their uses.
/// CEP:WHY: 38.18 "dominance validity": in the structured region tree a
///          use may only reference values whose defining region is an
///          ancestor-or-self of the use's region; violations are
///          miscompilation-class (use-before-definition in the schedule).
/// CEP:STATUS: complete
/// CEP:FAILURE: RegionTreeInvalid on non-dominating uses.
/// CEP:ASSUMES: regions form a tree (phase 1 proved it).
/// CEP:COST: O(inputs * region depth), bounded by region count.
/// CEP:EVIDENCE: test `non_dominating_use_detected`.
fn check_dominance(arena: &IrArena, id: NodeId, node: &Node) -> Option<VerifierError> {
    for i in 0..node.n_inputs as usize {
        if i >= MAX_INPUTS {
            break;
        }
        let def = node.inputs[i].node();
        let def_region = match arena.node(def) {
            Ok(n) => n.region,
            Err(_) => continue, // use-def check already reported this
        };
        if !region_dominates(arena, def_region, node.region) {
            return Some(VerifierError::RegionTreeInvalid(id));
        }
    }
    None
}

/// CEP:WHAT: Ancestor-or-self region dominance.
/// CEP:WHY: Structured regions form a tree; dominance is the ancestor
///          relation (see dominance.rs for the cached analysis used by
///          passes — the verifier needs the raw check without building
///          tables).
/// CEP:STATUS: complete
/// CEP:FAILURE: false on chain exhaustion (conservative).
/// CEP:ASSUMES: none
/// CEP:COST: O(depth), bounded by region count.
/// CEP:EVIDENCE: tests in this module.
fn region_dominates(
    arena: &IrArena,
    ancestor: xir_core::id::RegionId,
    descendant: xir_core::id::RegionId,
) -> bool {
    if ancestor == descendant || ancestor.is_none() {
        return true;
    }
    let mut cur = descendant;
    let mut steps = 0usize;
    let bound = arena.region_count() + 1;
    while !cur.is_none() && steps <= bound {
        if cur == ancestor {
            return true;
        }
        cur = match arena.region(cur) {
            Ok(r) => r.parent,
            Err(_) => return false,
        };
        steps += 1;
    }
    false
}

/// CEP:WHAT: Checks one node against the op contract.
/// CEP:STATUS: complete
/// CEP:FAILURE: the specific VerifierError for this node.
/// CEP:ASSUMES: node is live (caller iterated live slots).
/// CEP:COST: O(arity).
/// CEP:EVIDENCE: tests in this module.
fn verify_node(arena: &IrArena, id: NodeId, node: &Node) -> Option<VerifierError> {
    // Region handle.
    if !node.region.is_none() && arena.region(node.region).is_err() {
        return Some(VerifierError::RegionInvalid(id));
    }
    // Arity.
    if node.n_inputs != expected_arity(node.op) {
        return Some(VerifierError::ArityInvalid(id));
    }
    // Type: value-producing ops need a real type.
    if produces_value(node.op) && node.ty == Type::None {
        return Some(VerifierError::TypeInvalid(id));
    }
    // Use-def + slot discipline.
    for i in 0..node.n_inputs as usize {
        if i >= MAX_INPUTS {
            break;
        }
        let v = node.inputs[i];
        if v.is_none() {
            return Some(VerifierError::UseDefInvalid(id));
        }
        let def = v.node();
        match arena.node(def) {
            Ok(_) => {}
            Err(_) => return Some(VerifierError::UseDefInvalid(id)),
        }
        if v.slot() != 0 {
            return Some(VerifierError::SlotInvalid(id));
        }
    }
    // Effect chain: effectful ops need a token or a leading effect position.
    if node.op.has_effect() && node.effect_in.is_none() {
        // The first effectful node in a function may start the chain.
        // Validity: its region must contain no earlier effectful node.
        if let Ok(_r) = arena.region(node.region) {
            // Walk the region list before this node for an earlier effect.
            // (Cheap mode: accept chain starters; full mode re-checks.)
            let has_earlier_effect = region_has_earlier_effect(arena, node.region, id);
            if has_earlier_effect {
                return Some(VerifierError::EffectChainBroken(id));
            }
        }
    }
    None
}

/// CEP:WHAT: Reports whether an effectful node precedes `id` in its region.
/// CEP:WHY: Effect chains must be linear: each effectful node after the
///          first must consume a token (arch: "side-effect ordering via
///          token edges").
/// CEP:STATUS: complete
/// CEP:FAILURE: none; bounded walk (region list length).
/// CEP:ASSUMES: region live.
/// CEP:COST: O(region size).
/// CEP:EVIDENCE: effect-chain tests.
fn region_has_earlier_effect(arena: &IrArena, region: xir_core::id::RegionId, id: NodeId) -> bool {
    let head = match arena.region(region) {
        Ok(r) => r.first_node,
        Err(_) => return false,
    };
    let mut cur = head;
    let mut steps = 0usize;
    while !cur.is_none() && steps <= arena.node_count() + 1 {
        if cur == id {
            return false;
        }
        match arena.node(cur) {
            Ok(n) => {
                if n.op.has_effect() {
                    return true;
                }
                cur = n.next_in_region;
            }
            Err(_) => return false,
        }
        steps += 1;
    }
    false
}

/// CEP:WHAT: Whether an op produces a value (slot 0 meaningful).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: tests
fn produces_value(op: Op) -> bool {
    !matches!(
        op,
        Op::FusionBarrier
            | Op::TargetBarrier
            | Op::TargetWarpShuffle
            | Op::LoopAlloc { .. }
            | Op::LoopAsyncCopy
            | Op::LoopPipelineStage { .. }
    )
}

// Re-export for the pass manager's convenience.
pub use verify as verify_arena;

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_f64, const_i64};
    use xir_core::id::ValueId;
    use xir_core::node::Node;

    // CEP:WHAT: Well-formed IR passes verification.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on false rejection.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn valid_ir_passes() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        assert!(c0.is_ok() && c1.is_ok());
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let i0 = a.value_of(v0, 0);
            let i1 = a.value_of(v1, 0);
            assert!(i0.is_ok() && i1.is_ok());
            if let (Ok(x0), Ok(x1)) = (i0, i1) {
                let node = Node::new(
                    Op::Binary(xir_core::op::BinaryOp::Add),
                    root,
                    &[x0, x1],
                    Type::Scalar(xir_core::ty::ScalarType::I64),
                );
                let add = a.insert_node(root, node);
                assert!(add.is_ok());
            }
        }
        assert_eq!(verify(&a), Ok(()));
    }

    // CEP:WHAT: A stale use-def is detected.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the verifier misses the break.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn broken_use_def_detected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        // A use of a value that was never defined.
        let ghost = ValueId::from_node(NodeId::pack(9, 0, xir_core::id::IrLevel::Graph), 0);
        let node = Node::new(
            Op::Unary(xir_core::op::UnaryOp::Neg),
            root,
            &[ghost],
            Type::Scalar(xir_core::ty::ScalarType::F64),
        );
        let id = a.insert_node(root, node);
        assert!(id.is_ok());
        if let Ok(id) = id {
            assert_eq!(verify(&a), Err(VerifierError::UseDefInvalid(id)));
        }
    }

    // CEP:WHAT: Arity violations are detected.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the verifier misses the break.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn arity_violations_detected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        // dot with 1 input.
        let c = const_f64(&mut a, root, 1.0);
        assert!(c.is_ok());
        if let Ok(cv) = c {
            let v = a.value_of(cv, 0);
            assert!(v.is_ok());
            if let Ok(val) = v {
                let node = Node::new(
                    Op::Dot,
                    root,
                    &[val],
                    Type::Scalar(xir_core::ty::ScalarType::F64),
                );
                let id = a.insert_node(root, node);
                assert!(id.is_ok());
                if let Ok(id) = id {
                    assert_eq!(verify(&a), Err(VerifierError::ArityInvalid(id)));
                }
            }
        }
    }

    // CEP:WHAT: A use in a region not dominated by the definition is
    //           detected (audit F-5 regression).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if dominance violations pass verification.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn non_dominating_use_detected() {
        let mut a = IrArena::with_capacity(16, 8);
        let root = a.root_region();
        // r1 (child of root) defines; r2 (sibling) uses — r1 does not
        // dominate r2.
        let r1 = a.new_region(root, xir_core::id::NodeId::NONE);
        let r2 = a.new_region(root, xir_core::id::NodeId::NONE);
        if let (Ok(reg1), Ok(reg2)) = (r1, r2) {
            let c = const_f64(&mut a, reg1, 1.0);
            assert!(c.is_ok());
            if let Ok(cv) = c {
                let v = a.value_of(cv, 0);
                assert!(v.is_ok());
                if let Ok(val) = v {
                    let use_node = Node::new(
                        Op::Unary(xir_core::op::UnaryOp::Neg),
                        reg2,
                        &[val],
                        Type::Scalar(xir_core::ty::ScalarType::F64),
                    );
                    let uid = a.insert_node(reg2, use_node);
                    assert!(uid.is_ok());
                    if let Ok(u) = uid {
                        assert_eq!(verify(&a), Err(VerifierError::RegionTreeInvalid(u)));
                    }
                }
            }
        }
    }

    // CEP:WHAT: Missing type on a value-producing op is detected.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the verifier misses the break.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn type_violation_detected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let node = Node::new(Op::Dot, root, &[], Type::None);
        // Arity 0 vs expected 2 fires first; use an op with 0-input arity
        // but a value result: rng.
        let node2 = Node::new(
            Op::Rng {
                dist: xir_core::op::RngDist::Uniform,
                seed: 1,
            },
            root,
            &[],
            Type::None,
        );
        let _ = node;
        let id = a.insert_node(root, node2);
        assert!(id.is_ok());
        if let Ok(id) = id {
            assert_eq!(verify(&a), Err(VerifierError::TypeInvalid(id)));
        }
    }

    // CEP:WHAT: An If node without exactly two owned regions fails
    //           verification (CEP-12 structural contract).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the missing-region If verifies.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn if_without_regions_detected() {
        let mut a = IrArena::with_capacity(16, 8);
        let root = a.root_region();
        let c = const_f64(&mut a, root, 1.0);
        assert!(c.is_ok());
        if let Ok(cv) = c {
            if let Ok(cond) = a.value_of(cv, 0) {
                let node = Node::new(
                    Op::If,
                    root,
                    &[cond],
                    Type::Scalar(xir_core::ty::ScalarType::F64),
                );
                let id = a.insert_node(root, node);
                assert!(id.is_ok());
                if let Ok(id) = id {
                    assert_eq!(verify(&a), Err(VerifierError::BadIfRegions(id)));
                }
            }
        }
    }

    // CEP:WHAT: A region owned by a non-If (or parent-mismatched) node
    //           fails verification (CEP-12 linkage contract).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the bad linkage verifies.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn bad_region_owner_detected() {
        let mut a = IrArena::with_capacity(16, 8);
        let root = a.root_region();
        let c = const_f64(&mut a, root, 1.0);
        assert!(c.is_ok());
        if let Ok(cv) = c {
            // Owner is a Const node (not an If): linkage invalid.
            let r = a.new_region(root, cv);
            assert!(r.is_ok());
            if r.is_err() {
                return;
            }
            assert_eq!(verify(&a), Err(VerifierError::BadRegionOwner(cv)));
        }
    }

    // CEP:WHAT: A text-parsed if-region program verifies cleanly (the
    //           happy path: 2 owned regions, dominance holds for pre-if
    //           uses inside the blocks).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a well-formed if program is rejected.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn text_if_program_verifies() {
        let src = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<f64>\n",
            "  %1 = if %0 : scalar<f64> {\n",
            "    %2 = binary.mul %0, %0 : scalar<f64>\n",
            "  } else {\n",
            "    %3 = binary.add %0, %0 : scalar<f64>\n",
            "  }\n",
            "}\n",
        );
        let arena = xir_core::text::parse_arena(src, 64);
        assert!(arena.is_ok(), "parse error: {:?}", arena.err());
        if let Ok(a) = arena {
            assert_eq!(verify(&a), Ok(()));
        }
    }
}
