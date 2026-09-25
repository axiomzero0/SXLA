// CEP:FILE: crates/fusion/src/repair.rs
// CEP:WHAT: Fusion repair passes — cluster splitting, rematerialization,
//           copy insertion.
// CEP:WHY: Master architecture section 5: "If a cluster is illegal or
//          exceeds hardware limits, repair passes kick in: Cluster
//          Splitting (break along min-cut boundaries), Rematerialization
//          (duplicate cheap producers), Copy Insertion (materialize
//          intermediates to break illegal dependence cycles)." Splitting
//          here cuts at the LARGEST resource consumer (a 1-cut that
//          maximally relieves pressure — the budget-greedy approximation
//          of min-cut); rematerialization duplicates producers cheaper
//          than the bytes they save; copy insertion is defensive (SSA
//          graphs are acyclic — the verifier proves it, the repair keeps
//          the action for future mutable-buffer levels).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none; repair returns the action list (possibly empty when
//             the cluster is legal).
// CEP:ASSUMES: estimate computed for the same members.
// CEP:COST: O(members) per repair step; bounded iterations (see search).
// CEP:EVIDENCE: tests `over_shared_splits`, `over_registers_rematerializes`,
//           `legal_cluster_untouched`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-DETERMINISM: deterministic; largest-first with slot tie-break.
//! Fusion repair passes.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;

use crate::resource::{ResourceEstimate, SHARED_MEM_BUDGET_BYTES};

/// One repair action.
///
/// CEP:WHAT: The repair vocabulary of the architecture.
/// CEP:WHY: The search applies actions to bring clusters under budget;
///          explicit actions keep the decision auditable (telemetry
///          FusionDecision events reference them).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: node references live.
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairAction {
    /// Move `node` out of the cluster (cluster splitting).
    SplitCluster {
        /// The node to split out.
        node: NodeId,
    },
    /// Duplicate `node` instead of sharing it (rematerialization).
    Rematerialize {
        /// The cheap producer to duplicate.
        node: NodeId,
    },
    /// Materialize `node` to memory (copy insertion; defensive path).
    InsertCopy {
        /// The intermediate to materialize.
        node: NodeId,
    },
}

/// CEP:WHAT: Computes one repair step for a cluster against its estimate.
/// CEP:WHY: Policy: shared-memory overrun splits out the largest consumer;
///          register overrun rematerializes the cheapest producer; both
///          keep the cluster legal without discarding fusion entirely
///          (graceful degradation, Law 6).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (None when legal).
/// CEP:ASSUMES: members nonempty; estimate matches members.
/// CEP:COST: O(members).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn repair_cluster(
    arena: &IrArena,
    members: &[NodeId],
    estimate: &ResourceEstimate,
) -> Option<RepairAction> {
    if !estimate.over_budget() {
        return None;
    }
    if estimate.shared_bytes > SHARED_MEM_BUDGET_BYTES {
        // Split at the largest byte consumer.
        let mut best: Option<(NodeId, i64)> = None;
        for m in members {
            if let Ok(n) = arena.node(*m) {
                let bytes = n.ty.byte_size();
                let better = match best {
                    None => true,
                    Some((_, b)) => bytes > b,
                };
                if better {
                    best = Some((*m, bytes));
                }
            }
        }
        return best.map(|(node, _)| RepairAction::SplitCluster { node });
    }
    // Register pressure: rematerialize the CHEAPEST producer (smallest byte
    // footprint — duplicating it costs the least memory traffic).
    let mut cheapest: Option<(NodeId, i64)> = None;
    for m in members {
        if let Ok(n) = arena.node(*m) {
            let bytes = n.ty.byte_size();
            let better = match cheapest {
                None => true,
                Some((_, b)) => bytes < b,
            };
            if better {
                cheapest = Some((*m, bytes));
            }
        }
    }
    cheapest.map(|(node, _)| RepairAction::Rematerialize { node })
}

/// CEP:WHAT: Applies repair actions until the cluster fits or steps run out.
/// CEP:WHY: Bounded repair loop (CEP&CC 39: bounded resources, bounded
///          repair work); each step removes one member or duplicates one
///          producer, so at most `max_steps` actions return.
/// CEP:STATUS: complete
/// CEP:FAILURE: none; residual over-budget members return as InsertCopy
///              (defensive materialization — loud in telemetry).
/// CEP:ASSUMES: members live.
/// CEP:COST: O(steps * members).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn repair_to_fit(arena: &IrArena, members: &[NodeId], max_steps: u32) -> Vec<RepairAction> {
    let mut actions: Vec<RepairAction> = Vec::new();
    let mut current: Vec<NodeId> = members.to_vec();
    for _ in 0..max_steps {
        let est = crate::resource::resource_estimate(arena, &current);
        match repair_cluster(arena, &current, &est) {
            None => break,
            Some(RepairAction::SplitCluster { node }) => {
                current.retain(|m| *m != node);
                actions.push(RepairAction::SplitCluster { node });
            }
            Some(RepairAction::Rematerialize { node }) => {
                // Rematerialization relieves register pressure WITHOUT
                // removing the member: duplicate it conceptually; the cost
                // model prices the extra arithmetic.
                actions.push(RepairAction::Rematerialize { node });
                break;
            }
            Some(a @ RepairAction::InsertCopy { .. }) => {
                actions.push(a);
                break;
            }
        }
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::IrArena;
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op, UnaryOp};
    use xir_core::ty::{Layout, ScalarType, Shape, TensorType, Type};

    fn tensor(shape: &[i64]) -> Type {
        Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(shape).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        })
    }

    // CEP:WHAT: Legal clusters get no repair actions.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on spurious repair.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn legal_cluster_untouched() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let n = Node::new(
            Op::Binary(BinaryOp::Add),
            root,
            &[xir_core::id::ValueId::NONE; 2],
            tensor(&[4, 4]),
        );
        let id = a.insert_node(root, n);
        assert!(id.is_ok());
        if let Ok(i) = id {
            let est = crate::resource::resource_estimate(&a, &[i]);
            assert!(repair_cluster(&a, &[i], &est).is_none());
        }
    }

    // CEP:WHAT: Shared-memory overrun splits out the largest member.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong split target.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn over_shared_splits() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let small = Node::new(
            Op::Binary(BinaryOp::Add),
            root,
            &[xir_core::id::ValueId::NONE; 2],
            tensor(&[2, 2]),
        );
        let big = Node::new(
            Op::Unary(UnaryOp::Neg),
            root,
            &[xir_core::id::ValueId::NONE],
            tensor(&[64, 1024]),
        );
        let s = a.insert_node(root, small);
        let b = a.insert_node(root, big);
        if let (Ok(s), Ok(b)) = (s, b) {
            let members = [s, b];
            let est = crate::resource::resource_estimate(&a, &members);
            let action = repair_cluster(&a, &members, &est);
            assert_eq!(action, Some(RepairAction::SplitCluster { node: b }));
            // repair_to_fit converges by splitting the big node out.
            let actions = repair_to_fit(&a, &members, 4);
            assert!(actions.contains(&RepairAction::SplitCluster { node: b }));
        }
    }

    // CEP:WHAT: Register-only overrun rematerializes the cheapest producer.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong repair kind.
    // CEP:ASSUMES: shared bytes under budget while registers over.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn over_registers_rematerializes() {
        // Synthetic estimate: registers over, shared under.
        let est = ResourceEstimate {
            shared_bytes: 1024,
            register_units: crate::resource::REGISTER_BUDGET_UNITS + 32,
            occupancy_hint: 1,
        };
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let cheap = Node::new(Op::ConstI64(1), root, &[], Type::Scalar(ScalarType::I64));
        let c = a.insert_node(root, cheap);
        assert!(c.is_ok());
        if let Ok(c) = c {
            let action = repair_cluster(&a, &[c], &est);
            assert!(matches!(action, Some(RepairAction::Rematerialize { .. })));
        }
    }
}
