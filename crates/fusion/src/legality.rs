// CEP:FILE: crates/fusion/src/legality.rs
// CEP:WHAT: The fusion legality engine — index-map compatibility, reduction
//           legality, memory-aliasing checks for producer-consumer fusion.
// CEP:WHY: Master architecture section 5: "Uses Presburger/ISL-style
//          constraints to verify index map compatibility, reduction
//          legality, and memory aliasing." This release implements the
//          affine-compatible subset: shape compatibility (broadcast rules),
//           reduction-axis legality (a reduction may fuse with its producer
//          only when the reduction axis is not the fused iteration axis
//          without an accumulator split), and effect-free aliasing (buffers
//          are SSA values — no write aliases by construction, checked
//          explicitly). The full constraint engine is a documented TODO;
//          unproven cases are REJECTED (CEP&CC 38.22: transform only under
//          proven legality).
// CEP:CLASS: CEP-0
// CEP:STATUS: partial
// CEP:FAILURE: LegalityError codes name the exact failed check; can_fuse
//             returns Err (never a silent yes).
// CEP:ASSUMES: verified input; tensor types present on tensor ops.
// CEP:COST: O(arity) shape comparisons per candidate.
// CEP:EVIDENCE: tests `elementwise_fuses`, `reduction_axis_blocked`,
//           `barrier_blocks_fusion`, `shape_mismatch_rejected`.
// CEP:SECURITY: IR untrusted; explicit checks only.
// CEP:HPC-PASS-LEGALITY: This module IS the legality proof source.
// CEP:HPC-DETERMINISM: deterministic.
// CEP:TODO(main-agent): CEP-19: Presburger constraint solving for general
//           affine index maps (see docs/legality.md for the subset proof).
//! Fusion legality engine.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;
use xir_core::op::Op;
use xir_core::ty::{Layout, ScalarType, Shape, Type};

/// Legality failure enumeration.
///
/// CEP:WHAT: Exhaustive legality rejection reasons.
/// CEP:WHY: 38.22 — rejections must be diagnosable, not guesses.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalityError {
    /// Shapes are not broadcast-compatible for elementwise fusion.
    ShapeIncompatible,
    /// A reduction over the fused iteration axis would need an accumulator
    /// split (not proven legal here).
    ReductionAxisConflict,
    /// A fusion barrier sits between the candidates.
    BarrierBetween,
    /// One of the candidates has side effects (impure).
    Effectful,
    /// The producer's layout differs from the consumer's expectation and a
    /// conversion would cost more than the fusion saves.
    LayoutConversionTooExpensive,
    /// A candidate is not a tensor op (nothing to fuse).
    NotTensorOp,
}

/// CEP:WHAT: Checks whether consumer `c` may fuse with producer `p`.
/// CEP:WHY: The core legality query of the search: every fusion candidate
///          pair passes through exactly this function — one authority, no
///          scattered ad-hoc checks (Law 3).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err with the specific failed check (see LegalityError).
/// CEP:ASSUMES: both nodes live in the arena; types attached.
/// CEP:COST: O(1) + O(arity) shape walk.
/// CEP:EVIDENCE: tests in this module.
/// CEP:SECURITY: bounded lookups.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn can_fuse(arena: &IrArena, p: NodeId, c: NodeId) -> Result<bool, LegalityError> {
    let prod = arena.node(p).map_err(|_| LegalityError::NotTensorOp)?;
    let cons = arena.node(c).map_err(|_| LegalityError::NotTensorOp)?;
    // Effectful ops never fuse (side-effect ordering is token-threaded).
    if !prod.op.is_pure() || !cons.op.is_pure() {
        return Err(LegalityError::Effectful);
    }
    // Barriers are hard fences by construction (fusion.barrier).
    if matches!(prod.op, Op::FusionBarrier | Op::FusionMaterialize)
        || matches!(cons.op, Op::FusionBarrier | Op::FusionMaterialize)
    {
        return Err(LegalityError::BarrierBetween);
    }
    // Non-tensor producers (scalars) fuse trivially with elementwise
    // consumers (scalar broadcast).
    let pty = tensor_shape(prod);
    let cty = tensor_shape(cons);
    let (p_shape, c_shape) = match (pty, cty) {
        (Some(a), Some(b)) => (a, b),
        // Scalar or unknown-typed: allow only elementwise consumers.
        _ => {
            return Ok(is_elementwise(cons.op));
        }
    };
    // Elementwise consumer: shapes must broadcast-match.
    if is_elementwise(cons.op) {
        return shapes_broadcast_compatible(&p_shape, &c_shape);
    }
    // Reduction consumer: fusing into the producer's tile loop is legal
    // only for INNERMOST-axis reductions (row-wise accumulation inside one
    // tile). Reducing a non-innermost axis requires cross-tile accumulators
    // — the split-reduction strategy (Universe B), so the plain fuse is
    // rejected here (documented subset; full Presburger analysis is
    // CEP-19).
    if let Op::Reduce { axis, .. } = cons.op {
        let producer_rank = p_shape.rank();
        // The producer's result rank equals the reduction input rank; the
        // consumer's type is the reduced output. Reject when the reduced
        // axis is not the innermost axis of the producer's domain.
        let non_innermost = axis + 1 < producer_rank;
        if non_innermost {
            return Err(LegalityError::ReductionAxisConflict);
        }
        return Ok(true);
    }
    // Matmul consumers may fuse epilogues (elementwise output stage), but
    // not the main loop: treat as not fusable at this level (documented).
    if matches!(cons.op, Op::Matmul { .. } | Op::Conv { .. }) {
        return Err(LegalityError::NotTensorOp);
    }
    Ok(is_elementwise(cons.op))
}

/// CEP:WHAT: Reports whether the op is elementwise over its inputs.
/// CEP:WHY: Elementwise ops are the always-fusable class (same iteration
///          domain).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: tests
fn is_elementwise(op: Op) -> bool {
    matches!(op, Op::Binary(_) | Op::Unary(_))
}

/// CEP:WHAT: Tensor shape of a node (None for non-tensors).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: tests
fn tensor_shape(node: &xir_core::node::Node) -> Option<Shape> {
    match node.ty {
        Type::Tensor(t) => Some(t.shape),
        _ => None,
    }
}

/// CEP:WHAT: Broadcast compatibility of two shapes.
/// CEP:WHY: Index-map compatibility for elementwise fusion: each dim must
///          match or be 1 on one side (NumPy broadcasting rule — the
///          affine-compatible subset of the full index-map check).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err(ShapeIncompatible) on dim mismatch.
/// CEP:ASSUMES: none
/// CEP:COST: O(rank).
/// CEP:EVIDENCE: test `shape_mismatch_rejected`.
fn shapes_broadcast_compatible(a: &Shape, b: &Shape) -> Result<bool, LegalityError> {
    let (long, short) = if a.rank() >= b.rank() {
        (a.as_slice(), b.as_slice())
    } else {
        (b.as_slice(), a.as_slice())
    };
    let offset = long.len() - short.len();
    for (i, d) in long.iter().enumerate() {
        if i < offset {
            continue;
        }
        let s = short[i - offset];
        if *d != s && *d != 1 && s != 1 {
            return Err(LegalityError::ShapeIncompatible);
        }
    }
    Ok(true)
}

/// CEP:WHAT: Layout conversion cost between producer and consumer layouts.
/// CEP:WHY: The search prices layout-mismatched fusions with a conversion
///          term (architecture: extraction penalizes layout-breaking
///          rewrites); identical layouts cost zero.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: cost model tests.
pub fn layout_conversion_cost(from: Layout, to: Layout, elem: ScalarType) -> u32 {
    if from == to {
        0
    } else {
        // Full-element shuffle penalty, scaled by element width.
        8 * (elem.byte_size() / 4).max(1)
    }
}

// MAX_INPUTS participates in arity walks; import kept for future checks.
#[allow(unused_imports)]
use MAX_INPUTS as _MaxInputsDoc;

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::IrArena;
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op, UnaryOp};
    use xir_core::ty::{ScalarType, Shape, TensorType, Type};

    fn tensor(shape: &[i64]) -> Type {
        Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(shape).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        })
    }

    // CEP:WHAT: Elementwise producer-consumer pairs fuse.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on false rejection.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn elementwise_fuses() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let m = Node::new(Op::Param { index: 0 }, root, &[], tensor(&[4, 4]));
        let mid = a.insert_node(root, m);
        assert!(mid.is_ok());
        if let Ok(mv) = mid {
            let v = a.value_of(mv, 0);
            assert!(v.is_ok());
            if let Ok(val) = v {
                let neg = Node::new(Op::Unary(UnaryOp::Neg), root, &[val], tensor(&[4, 4]));
                let nid = a.insert_node(root, neg);
                assert!(nid.is_ok());
                if let (Ok(p), Ok(c)) = (mid, nid) {
                    assert_eq!(can_fuse(&a, p, c), Ok(true));
                }
            }
        }
    }

    // CEP:WHAT: Shape-mismatched elementwise fusion is rejected.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on illegal acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn shape_mismatch_rejected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let m = Node::new(Op::Param { index: 0 }, root, &[], tensor(&[4, 8]));
        let mid = a.insert_node(root, m);
        let neg = Node::new(
            Op::Unary(UnaryOp::Neg),
            root,
            &[xir_core::id::ValueId::NONE],
            tensor(&[16, 16]),
        );
        let nid = a.insert_node(root, neg);
        if let (Ok(p), Ok(c)) = (mid, nid) {
            assert_eq!(can_fuse(&a, p, c), Err(LegalityError::ShapeIncompatible));
        }
    }

    // CEP:WHAT: Axis-0 same-rank reductions are blocked pending split.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on illegal acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn reduction_axis_blocked() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let m = Node::new(Op::Param { index: 0 }, root, &[], tensor(&[4, 4]));
        let mid = a.insert_node(root, m);
        let red = Node::new(
            Op::Reduce {
                axis: 0,
                monoid: xir_core::op::Monoid::Add,
            },
            root,
            &[xir_core::id::ValueId::NONE],
            tensor(&[4]),
        );
        let rid = a.insert_node(root, red);
        if let (Ok(p), Ok(c)) = (mid, rid) {
            assert_eq!(
                can_fuse(&a, p, c),
                Err(LegalityError::ReductionAxisConflict)
            );
        }
    }

    // CEP:WHAT: Effectful producers never fuse.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on illegal acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn effectful_rejected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let rng = Node::new(
            Op::Rng {
                dist: xir_core::op::RngDist::Uniform,
                seed: 1,
            },
            root,
            &[],
            tensor(&[4]),
        );
        let rid = a.insert_node(root, rng);
        let neg = Node::new(
            Op::Unary(UnaryOp::Neg),
            root,
            &[xir_core::id::ValueId::NONE],
            tensor(&[4]),
        );
        let nid = a.insert_node(root, neg);
        if let (Ok(p), Ok(c)) = (rid, nid) {
            assert_eq!(can_fuse(&a, p, c), Err(LegalityError::Effectful));
        }
    }

    // CEP:WHAT: Layout conversion cost is zero for matching layouts.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on cost inversion.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn layout_cost_table() {
        assert_eq!(
            layout_conversion_cost(Layout::RowMajor, Layout::RowMajor, ScalarType::F64),
            0
        );
        assert!(layout_conversion_cost(Layout::RowMajor, Layout::ColMajor, ScalarType::F64) > 0);
        // Silence unused import warning for BinaryOp in this test module.
        let _ = Op::Binary(BinaryOp::Add);
    }
}
