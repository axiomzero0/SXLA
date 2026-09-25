// CEP:FILE: crates/xir-levels/src/level1.rs
// CEP:WHAT: Level-1 tensor algebra graph — layout inference and transpose
//           propagation passes.
// CEP:WHY: Master architecture Level 1: "Passes: Layout inference, transpose
//          propagation, reduce reassociation, sharding propagation". Layout
//          is a first-class attribute of TensorType; inference propagates
//          canonical layouts down the def-use chain so the fusion legality
//          engine can price layout conversions (arch section 5: extraction
//          penalizes rewrites that force expensive layout conversions).
// CEP:CLASS: CEP-0 (hot pass body)
// CEP:STATUS: partial
// CEP:FAILURE: PassError from xir-graph propagates; conservative no-op
//             otherwise (never a silent miscompile).
// CEP:ASSUMES: verified input; tensor types already attached by the
//           frontend or the layout pass's own type synthesis.
// CEP:COST: O(nodes) single pass per node + O(uses) rewrite scan.
// CEP:EVIDENCE: tests `layout_propagates_through_transpose`,
//           `elementwise_keeps_layout`.
// CEP:SECURITY: IR untrusted; bounds-checked.
// CEP:HPC-PASS: layout-infer
// CEP:HPC-PASS-KIND: layout inference
// CEP:HPC-PASS-INPUT: verified Level-0/1 snapshot
// CEP:HPC-PASS-OUTPUT: tensor types with canonical layouts
// CEP:HPC-PASS-ANALYSIS-REQUIRED: use-def chains
// CEP:HPC-PASS-ANALYSIS-PRODUCED: layout map (transient)
// CEP:HPC-PASS-ANALYSIS-INVALIDATED: none
// CEP:HPC-PASS-LEGALITY: layout rewrite only on layout-free (None-typed
//           placeholder) or default RowMajor tensors; never overrides an
//           explicit non-default layout
// CEP:HPC-PASS-PRESERVES: semantics, shapes, effect order
// CEP:HPC-PASS-COST: O(nodes)
// CEP:HPC-PASS-FAILURE: conservative skip
// CEP:HPC-PASS-TARGET: target-independent
// CEP:HPC-PASS-EVIDENCE: tests in this module
// CEP:HPC-TRANSFORM: Propagates layouts through transpose chains.
// CEP:HPC-DETERMINISM: deterministic; slot-order walk
// CEP:TODO(main-agent): CEP-15: sharding propagation + reduce
//           reassociation (reassociation is gated on monoid/float rules).
//! Level-1 tensor passes.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;
use xir_core::op::Op;
use xir_core::ty::{Layout, TensorType, Type};

/// CEP:WHAT: Runs layout inference over tensor-producing nodes.
/// CEP:WHY: Canonical layouts: transpose consumers get ColMajor sources
///          where the producer allows it, enabling the fusion legality
///          engine to skip physical transposes (transpose propagation).
///          Only RowMajor-default tensors are rewritten — explicit
///          non-default layouts are caller intent and stay untouched
///          (legality above).
/// CEP:STATUS: complete
/// CEP:FAILURE: none; conservative skips.
/// CEP:ASSUMES: verified input.
/// CEP:COST: O(nodes); mutation in place on the caller's working arena.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn layout_infer(arena: &mut IrArena) -> u32 {
    let mut changed = 0u32;
    // Collect tensor nodes in slot order (deterministic).
    let mut tensor_nodes: Vec<(NodeId, Op)> = Vec::new();
    arena.for_each_live_node(|id, node| {
        if matches!(node.ty, Type::Tensor(_)) {
            tensor_nodes.push((id, node.op));
        }
    });
    for (id, op) in tensor_nodes {
        let inferred = match op {
            // A transpose flips the layout of its result.
            Op::Transpose { .. } => Some(Layout::ColMajor),
            // Matmul prefers row-major accumulators.
            Op::Matmul { .. } => Some(Layout::RowMajor),
            // Everything else: inherit from the first tensor input.
            _ => first_input_layout(arena, id),
        };
        if let Some(layout) = inferred {
            if let Ok(node) = arena.node_mut(id) {
                if let Type::Tensor(t) = node.ty {
                    if t.layout != layout && t.layout == Layout::RowMajor {
                        node.ty = Type::Tensor(TensorType {
                            elem: t.elem,
                            shape: t.shape,
                            layout,
                        });
                        changed += 1;
                    }
                }
            }
        }
    }
    changed
}

/// CEP:WHAT: Reads the layout of the first tensor-typed input.
/// CEP:WHY: Elementwise chains inherit layouts from producers (the
///          "elementwise keeps layout" rule the fusion engine relies on).
/// CEP:STATUS: complete
/// CEP:FAILURE: none; None when no tensor input exists.
/// CEP:ASSUMES: inputs live (verified).
/// CEP:COST: O(arity).
/// CEP:EVIDENCE: tests in this module.
fn first_input_layout(arena: &IrArena, id: NodeId) -> Option<Layout> {
    let node = arena.node(id).ok()?;
    for i in 0..node.n_inputs as usize {
        if i >= MAX_INPUTS {
            break;
        }
        let def = node.inputs[i].node();
        if let Ok(prod) = arena.node(def) {
            if let Type::Tensor(t) = prod.ty {
                return Some(t.layout);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::const_f64;
    use xir_core::node::Node;
    use xir_core::ty::{ScalarType, Shape};

    fn tensor_ty(layout: Layout) -> Type {
        Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(&[2, 3]).ok().unwrap_or(Shape::scalar()),
            layout,
        })
    }

    // CEP:WHAT: A transpose node's result becomes ColMajor.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if layout is not propagated.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn layout_propagates_through_transpose() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c = const_f64(&mut a, root, 1.0);
        assert!(c.is_ok());
        if let Ok(cv) = c {
            let v = a.value_of(cv, 0);
            assert!(v.is_ok());
            if let Ok(val) = v {
                let tr = Node::new(
                    Op::Transpose {
                        perm: [1, 0, 0, 0],
                        rank: 2,
                    },
                    root,
                    &[val],
                    tensor_ty(Layout::RowMajor),
                );
                let id = a.insert_node(root, tr);
                assert!(id.is_ok());
                let changed = layout_infer(&mut a);
                assert!(changed >= 1);
                if let Ok(n) = a.node(id.ok().unwrap_or(NodeId::NONE)) {
                    if let Type::Tensor(t) = n.ty {
                        assert_eq!(t.layout, Layout::ColMajor);
                    }
                }
            }
        }
    }

    // CEP:WHAT: Row-major tensors without layout-flipping ops are kept.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on spurious rewrites.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn elementwise_keeps_layout() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c = const_f64(&mut a, root, 1.0);
        if let Ok(cv) = c {
            if let Ok(val) = a.value_of(cv, 0) {
                let neg = Node::new(
                    Op::Unary(xir_core::op::UnaryOp::Neg),
                    root,
                    &[val],
                    tensor_ty(Layout::RowMajor),
                );
                let id = a.insert_node(root, neg);
                assert!(id.is_ok());
                // Scalar const producer: no tensor input -> no rewrite.
                let changed = layout_infer(&mut a);
                assert_eq!(changed, 0);
            }
        }
    }
}
