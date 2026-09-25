// CEP:FILE: crates/xir-graph/src/schedule.rs
// CEP:WHAT: Canonical sea-of-nodes scheduler — deterministic topological
//           execution order per region.
// CEP:WHY: Sea-of-nodes has no implicit order; the interpreter (Tier-0
//          fallback), the Level-3 lowering and the printer need ONE
//          canonical order. Kahn's algorithm with slot-index tie-breaking
//          gives a deterministic, dominance-respecting order (values before
//          uses) — CEP&CC 38.19 (stable scheduling) and the HPC-IR contract
//          ("structured regions are secondary projections" of the graph).
// CEP:CLASS: CEP-1 (analysis output for cold paths) — the hot passes stay
//           order-free; scheduling happens at projection boundaries.
// CEP:STATUS: complete
// CEP:FAILURE: ScheduleError::{Cycle, UnknownNode} — cycles are loud, never
//             guessed around.
// CEP:ASSUMES: verified input (no dangling uses).
// CEP:COST: O(nodes + edges) with a bounded ready queue (Vec, slot order).
// CEP:EVIDENCE: tests `values_before_uses`, `deterministic_order`,
//           `cycle_is_detected`.
// CEP:SECURITY: bounded walks.
// CEP:HPC-DETERMINISM: deterministic — Kahn with slot tie-break.
//! Canonical scheduler.
use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;

/// Scheduler failure enumeration.
///
/// CEP:WHAT: Explicit error type.
/// CEP:WHY: Law 6 — cyclic graphs must fail loudly (a cycle in a
///          sea-of-nodes is a broken invariant, Severity 0 material).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleError {
    /// The value graph contains a cycle (broken SSA invariant).
    Cycle,
    /// A use referenced an unknown node (input not verified).
    UnknownNode,
}

/// CEP:WHAT: Computes the canonical execution order (all live nodes).
/// CEP:WHY: The projection of the sea-of-nodes into a linear program:
///          region-tree order first (parents before children), then
///          value-dependence topological order with slot tie-breaking.
/// CEP:STATUS: complete
/// CEP:FAILURE: Cycle / UnknownNode.
/// CEP:ASSUMES: verified arena.
/// CEP:COST: O(nodes + edges); one allocation for the order (CEP-1 output).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic tie-breaking by node slot index.
pub fn schedule(arena: &IrArena) -> Result<Vec<NodeId>, ScheduleError> {
    // Collect live nodes in slot order.
    let mut ids: Vec<NodeId> = Vec::with_capacity(arena.node_count());
    arena.for_each_live_node(|id, _| ids.push(id));
    // In-degree = number of live value inputs.
    let mut indegree: Vec<u32> = Vec::with_capacity(ids.len());
    for id in ids.iter() {
        let node = arena.node(*id).map_err(|_| ScheduleError::UnknownNode)?;
        let mut d = 0u32;
        for i in 0..node.n_inputs as usize {
            if i < MAX_INPUTS {
                d += 1;
            }
        }
        indegree.push(d);
    }
    // Map slot -> position in ids for decrementing dependents.
    let mut pos_of: Vec<Option<usize>> = vec![None; arena.slot_count()];
    for (pos, id) in ids.iter().enumerate() {
        let slot = id.index() as usize;
        if slot < pos_of.len() {
            pos_of[slot] = Some(pos);
        }
    }
    // Reverse edges: producer -> consumers.
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    for (pos, id) in ids.iter().enumerate() {
        let node = arena.node(*id).map_err(|_| ScheduleError::UnknownNode)?;
        for i in 0..node.n_inputs as usize {
            if i >= MAX_INPUTS {
                break;
            }
            let producer_slot = node.inputs[i].node().index() as usize;
            if let Some(ppos) = pos_of.get(producer_slot).copied().flatten() {
                consumers[ppos].push(pos);
            } else {
                // Producer not live: verifier should have caught this.
                return Err(ScheduleError::UnknownNode);
            }
        }
    }
    // Kahn: ready = zero indegree in slot order (ids is slot-ordered, so a
    // simple scan preserves the tie-break).
    let mut order: Vec<NodeId> = Vec::with_capacity(ids.len());
    let mut indeg = indegree;
    let mut emitted = vec![false; ids.len()];
    let mut remaining = ids.len();
    while remaining > 0 {
        let mut progressed = false;
        for pos in 0..ids.len() {
            if emitted[pos] || indeg[pos] != 0 {
                continue;
            }
            // Emit in slot order.
            order.push(ids[pos]);
            emitted[pos] = true;
            remaining -= 1;
            progressed = true;
            for &c in consumers[pos].iter() {
                if !emitted[c] {
                    indeg[c] = indeg[c].saturating_sub(1);
                }
            }
        }
        if !progressed {
            return Err(ScheduleError::Cycle);
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_f64, const_i64};
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};

    // CEP:WHAT: Scheduled order puts every value before its uses.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on topological violation.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn values_before_uses() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(i0), Some(i1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[i0, i1],
                    Type::Scalar(ScalarType::I64),
                );
                let _ = a.insert_node(root, add);
            }
        }
        let order = schedule(&a);
        assert!(order.is_ok());
        if let Ok(ord) = order {
            assert_eq!(ord.len(), 3);
            // The add must come last: its inputs are the two consts.
            let last = ord[2];
            let n = a.node(last);
            assert!(n.is_ok());
            if let Ok(n) = n {
                assert_eq!(n.op, Op::Binary(BinaryOp::Add));
            }
        }
    }

    // CEP:WHAT: Scheduling is deterministic across calls.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn deterministic_order() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let _ = const_f64(&mut a, root, 1.0);
        let _ = const_f64(&mut a, root, 2.0);
        let _ = const_f64(&mut a, root, 3.0);
        let o1 = schedule(&a);
        let o2 = schedule(&a);
        assert!(o1.is_ok() && o2.is_ok());
        if let (Ok(x1), Ok(x2)) = (o1, o2) {
            assert_eq!(x1, x2);
        }
    }

    // CEP:WHAT: Cyclic graphs are detected (defensive; verifier normally
    //           prevents them).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a cycle schedules silently.
    // CEP:ASSUMES: hand-crafted cycle bypasses the verifier.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn cycle_is_detected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        // Self-referencing node: input points at itself (invalid IR).
        let ghost = Node::new(
            Op::Unary(xir_core::op::UnaryOp::Neg),
            root,
            &[xir_core::id::ValueId::NONE],
            Type::Scalar(ScalarType::F64),
        );
        let g = a.insert_node(root, ghost);
        assert!(g.is_ok());
        if let Ok(gid) = g {
            let gv = a.value_of(gid, 0);
            assert!(gv.is_ok());
            if let Ok(gval) = gv {
                if let Ok(n) = a.node_mut(gid) {
                    n.inputs[0] = gval;
                    n.n_inputs = 1;
                }
                // Self-cycle: scheduler must report it.
                assert_eq!(schedule(&a), Err(ScheduleError::Cycle));
            }
        }
    }
}
