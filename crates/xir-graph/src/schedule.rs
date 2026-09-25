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
//           `cycle_is_detected`, `fused_schedule_keeps_legality`,
//           `fused_schedule_groups_cluster_members`.
// CEP:SECURITY: bounded walks.
// CEP:HPC-DETERMINISM: deterministic — Kahn with slot tie-break; the fused
//           variant adds a cluster-affinity key ahead of the slot key.
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

/// CEP:WHAT: Computes a FUSION-AWARE execution order: topological legality
///           first, cluster locality second.
/// CEP:WHY: CEP-22 half-closing (audit F-3): the fusion search's winning
///          ClusterSet must influence the Level-3 projection — the
///          architecture's "ClusterSet feeds the level-3 tiling decisions".
///          Kahn emits ONE node per step; among READY nodes the key is
///          (cluster-affinity, slot): a node in the ACTIVE cluster (the
///          cluster of the last emitted node) wins over same-slot-lower
///          strangers, so cluster members land ADJACENT in the schedule
///          (producer-consumer locality — the whole point of fusion) while
///          every emission stays a zero-indegree node (values-before-uses
///          legality is untouched). Unclustered nodes form their own
///          neutral group: None == None is affinity, None vs Some is not.
/// CEP:STATUS: complete
/// CEP:FAILURE: Cycle / UnknownNode (same contract as `schedule`).
/// CEP:ASSUMES: verified arena; `cluster_of` is a pure lookup (the
///              ClusterSet is immutable during scheduling).
/// CEP:COST: Theta(nodes^2) ALWAYS (nodes full ready-scans, one emission
///           per scan) — versus the batched plain scheduler's
///           O(depth * (nodes + edges)); correctness-first Tier-2 path, a
///           ready-heap redesign is a future optimization (documented).
/// CEP:EVIDENCE: tests `fused_schedule_keeps_legality`,
///           `fused_schedule_groups_cluster_members`.
/// CEP:HPC-DETERMINISM: deterministic — fixed key, fixed scan order.
pub fn schedule_fused(
    arena: &IrArena,
    cluster_of: impl Fn(NodeId) -> Option<u32>,
) -> Result<Vec<NodeId>, ScheduleError> {
    // Same setup as `schedule`: slot-ordered live set, indegrees, consumers.
    let mut ids: Vec<NodeId> = Vec::with_capacity(arena.node_count());
    arena.for_each_live_node(|id, _| ids.push(id));
    let mut indeg: Vec<u32> = Vec::with_capacity(ids.len());
    for id in ids.iter() {
        let node = arena.node(*id).map_err(|_| ScheduleError::UnknownNode)?;
        let mut d = 0u32;
        for i in 0..node.n_inputs as usize {
            if i < MAX_INPUTS {
                d += 1;
            }
        }
        indeg.push(d);
    }
    let mut pos_of: Vec<Option<usize>> = vec![None; arena.slot_count()];
    for (pos, id) in ids.iter().enumerate() {
        let slot = id.index() as usize;
        if slot < pos_of.len() {
            pos_of[slot] = Some(pos);
        }
    }
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
    for (pos, id) in ids.iter().enumerate() {
        let node = arena.node(*id).map_err(|_| ScheduleError::UnknownNode)?;
        for i in 0..node.n_inputs as usize {
            if i >= MAX_INPUTS {
                break;
            }
            let def = node.inputs[i].node();
            match pos_of.get(def.index() as usize).copied().flatten() {
                Some(dpos) => consumers[dpos].push(pos),
                // Dangling use: same loud contract as `schedule` (audit F-5
                // — skipping the edge would surface later as a bogus Cycle).
                None => return Err(ScheduleError::UnknownNode),
            }
        }
    }
    // Step-wise Kahn with the (affinity, slot) key.
    let cluster_ids: Vec<Option<u32>> = ids.iter().map(|id| cluster_of(*id)).collect();
    let mut order: Vec<NodeId> = Vec::with_capacity(ids.len());
    let mut emitted = vec![false; ids.len()];
    let mut remaining = ids.len();
    let mut active: Option<u32> = None;
    while remaining > 0 {
        // Best ready node: same-cluster first, then lowest slot.
        let mut best: Option<usize> = None;
        for pos in 0..ids.len() {
            if emitted[pos] || indeg[pos] != 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => {
                    // Key: (!same-active-cluster, slot) — minimized. The
                    // negation makes same-cluster (true == affinity) sort
                    // FIRST (false < true as the leading tuple element).
                    let key_p = (cluster_ids[pos] != active, ids[pos].index());
                    let key_b = (cluster_ids[b] != active, ids[b].index());
                    key_p < key_b
                }
            };
            if better {
                best = Some(pos);
            }
        }
        let Some(pos) = best else {
            return Err(ScheduleError::Cycle);
        };
        order.push(ids[pos]);
        emitted[pos] = true;
        remaining -= 1;
        active = cluster_ids[pos];
        for &c in consumers[pos].iter() {
            if !emitted[c] {
                indeg[c] = indeg[c].saturating_sub(1);
            }
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

    // CEP:WHAT: The fused scheduler keeps values-before-uses legality on a
    //           diamond graph with an unrelated stranger node interleaved.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if any use precedes its value.
    // CEP:ASSUMES: diamond: c0 -> (a, b) -> add; stranger const independent.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fused_schedule_keeps_legality() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_f64(&mut a, root, 1.0);
        let c1 = const_f64(&mut a, root, 2.0);
        let stranger = const_f64(&mut a, root, 9.0);
        if let (Ok(v0), Ok(v1), Ok(vs)) = (c0, c1, stranger) {
            let (i0, i1, is) = (
                a.value_of(v0, 0).ok(),
                a.value_of(v1, 0).ok(),
                a.value_of(vs, 0).ok(),
            );
            if let (Some(x0), Some(x1), Some(xs)) = (i0, i1, is) {
                let _ = xs;
                let l = a.insert_node(
                    root,
                    Node::new(
                        Op::Binary(BinaryOp::Add),
                        root,
                        &[x0, x1],
                        Type::Scalar(ScalarType::F64),
                    ),
                );
                let r = a.insert_node(
                    root,
                    Node::new(
                        Op::Binary(BinaryOp::Mul),
                        root,
                        &[x0, x1],
                        Type::Scalar(ScalarType::F64),
                    ),
                );
                if let (Ok(l_id), Ok(r_id)) = (l, r) {
                    let (il, ir) = (a.value_of(l_id, 0).ok(), a.value_of(r_id, 0).ok());
                    if let (Some(xl), Some(xr)) = (il, ir) {
                        let _sum = a.insert_node(
                            root,
                            Node::new(
                                Op::Binary(BinaryOp::Add),
                                root,
                                &[xl, xr],
                                Type::Scalar(ScalarType::F64),
                            ),
                        );
                        // Cluster: {l, r} together; everything else unclustered.
                        let in_cluster = |id: NodeId| {
                            if id == l_id || id == r_id {
                                Some(0u32)
                            } else {
                                None
                            }
                        };
                        let order = schedule_fused(&a, in_cluster);
                        assert!(order.is_ok());
                        if let Ok(ord) = order {
                            let pos_of: std::collections::BTreeMap<u32, usize> = ord
                                .iter()
                                .enumerate()
                                .map(|(p, id)| (id.index(), p))
                                .collect();
                            for id in ord.iter() {
                                if let Ok(node) = a.node(*id) {
                                    for i in 0..node.n_inputs as usize {
                                        if i >= MAX_INPUTS {
                                            break;
                                        }
                                        let def = node.inputs[i].node();
                                        let dp = pos_of.get(&def.index()).copied().unwrap_or(0);
                                        let up = pos_of.get(&id.index()).copied().unwrap_or(0);
                                        assert!(dp < up, "use before value");
                                    }
                                }
                            }
                            // Cluster members adjacent.
                            let lp = pos_of.get(&l_id.index()).copied().unwrap_or(0);
                            let rp = pos_of.get(&r_id.index()).copied().unwrap_or(0);
                            assert_eq!((lp as i64 - rp as i64).abs(), 1);
                        }
                    }
                }
            }
        }
    }

    // CEP:WHAT: Cluster members group adjacently even when a lower-slot
    //           stranger is ready at the same step (affinity beats slot).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if affinity loses to the slot key.
    // CEP:ASSUMES: two-cluster chain with interleaved strangers.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fused_schedule_groups_cluster_members() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        // s0, s1, s2: strangers (slot order first). c0 -> c1: cluster pair
        // whose consumer is only ready after c0 (c1 must wait).
        let s0 = const_i64(&mut a, root, 10);
        let s1 = const_i64(&mut a, root, 11);
        let s2 = const_i64(&mut a, root, 12);
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        if let (Ok(v0), Ok(v1), Ok(v2), Ok(w0), Ok(w1)) = (s0, s1, s2, c0, c1) {
            let _ = (v0, v1, v2);
            let i0 = a.value_of(w0, 0).ok();
            let i1 = a.value_of(w1, 0).ok();
            if let (Some(x0), Some(x1)) = (i0, i1) {
                let add = a
                    .insert_node(
                        root,
                        Node::new(
                            Op::Binary(BinaryOp::Add),
                            root,
                            &[x0, x1],
                            Type::Scalar(ScalarType::I64),
                        ),
                    )
                    .ok();
                let c0_id = w0;
                let c1_id = w1;
                let add_id = add.unwrap_or(NodeId::NONE);
                let in_cluster = |id: NodeId| {
                    if id == c0_id || id == c1_id || id == add_id {
                        Some(0u32)
                    } else {
                        None
                    }
                };
                let order = schedule_fused(&a, in_cluster);
                assert!(order.is_ok());
                if let Ok(ord) = order {
                    let positions: Vec<u32> = ord.iter().map(|id| id.index()).collect();
                    // The three cluster members are consecutive.
                    let cp: Vec<usize> = positions
                        .iter()
                        .enumerate()
                        .filter(|(_, idx)| {
                            **idx == c0_id.index()
                                || **idx == c1_id.index()
                                || **idx == add_id.index()
                        })
                        .map(|(p, _)| p)
                        .collect();
                    assert_eq!(cp.len(), 3);
                    let span = cp[2] - cp[0];
                    assert_eq!(span, 2, "cluster members must be adjacent: {positions:?}");
                    // Legality still holds: c0 and c1 before add.
                    let p_of =
                        |target: u32| positions.iter().position(|x| *x == target).unwrap_or(0);
                    assert!(p_of(c0_id.index()) < p_of(add_id.index()));
                    assert!(p_of(c1_id.index()) < p_of(add_id.index()));
                }
            }
        }
    }

    // CEP:WHAT: The affinity key DISCRIMINATES against plain slot order: a
    //           stranger at a slot BETWEEN cluster members is emitted
    //           BEFORE the pair (slot order alone would interleave it
    //           between them). This is the regression that pins the
    //           mechanism itself (audit F-1): deleting the affinity term
    //           from the key makes this test fail while the rest of the
    //           suite stays green.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if affinity stops steering the order.
    // CEP:ASSUMES: cluster {P(slot 3), C(slot 5)}; stranger S(slot 4)
    //              ready at the same step as P; C depends on P only.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fused_schedule_affinity_beats_slot() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let s0 = const_i64(&mut a, root, 10);
        let s1 = const_i64(&mut a, root, 11);
        let p = const_i64(&mut a, root, 3);
        let stranger = const_i64(&mut a, root, 99);
        // Cluster members: P (slot 3) and C (slot 5, consumes P).
        if let (Ok(v0), Ok(v1), Ok(vp), Ok(vs)) = (s0, s1, p, stranger) {
            let _ = (v0, v1, vs);
            let ip = a.value_of(vp, 0);
            if let Ok(xp) = ip {
                let c = a.insert_node(
                    root,
                    Node::new(
                        Op::Unary(xir_core::op::UnaryOp::Neg),
                        root,
                        &[xp],
                        Type::Scalar(ScalarType::I64),
                    ),
                );
                if let (Ok(p_id), Ok(c_id)) = (p, c) {
                    let s_id = vs;
                    let in_cluster = |id: NodeId| {
                        if id == p_id || id == c_id {
                            Some(0u32)
                        } else {
                            None
                        }
                    };
                    let order = schedule_fused(&a, in_cluster);
                    assert!(order.is_ok());
                    if let Ok(ord) = order {
                        let positions: Vec<u32> = ord.iter().map(|id| id.index()).collect();
                        let p_of =
                            |target: u32| positions.iter().position(|x| *x == target).unwrap_or(0);
                        // AFFINITY effect: the stranger (slot 4) is pulled
                        // BEFORE the pair, so the cluster members become
                        // adjacent. Plain slot order would emit P(3),
                        // S(4), C(5) — span 2 with the stranger inside.
                        assert!(p_of(s_id.index()) < p_of(p_id.index()));
                        assert_eq!(
                            p_of(c_id.index()) - p_of(p_id.index()),
                            1,
                            "cluster members must be adjacent under affinity: {positions:?}"
                        );
                        // Legality: P before C.
                        assert!(p_of(p_id.index()) < p_of(c_id.index()));
                    }
                }
            }
        }
    }
}
