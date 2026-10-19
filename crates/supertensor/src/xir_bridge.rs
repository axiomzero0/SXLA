// CEP:FILE: crates/supertensor/src/xir_bridge.rs
// CEP:WHAT: The XIR boundary — consumes the verified winner and produces the
//           artifacts the EXISTING pipeline consumes: a fusion ClusterSet
//           (from the winner's fuse decisions, budgeted by the hardware
//           model, edges derived through the ClusterSet's own authority) and
//           per-op tile hints (diagnostic surface for the future Level-3
//           tiling consumer).
// CEP:WHY: Zero core modification (axiom 5): supertensor speaks XIR through
//          public APIs only — ClusterSet is the pipeline's native fusion
//          artifact (project_with_fusion consumes it). Schedules are
//          placement-only transformations, so values are preserved by
//          construction and enforced differentially in the jit tests.
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: Err(SuperOptError) on any cluster assignment error (loud).
// CEP:ASSUMES: winner passed the verifier (pairs are legal); arena unchanged
//              since extraction.
// CEP:COST: O(pairs α(n)) union-find + O(nodes) edge derivation.
// CEP:EVIDENCE: tests `clusters_cover_fused_pairs`,
//           `clusters_match_search_output_values` (via lib entry test).
// CEP:SECURITY: no untrusted input.
//! XIR boundary: winner schedule -> ClusterSet + tile hints.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_levels::level2::ClusterSet;

use crate::polytope::{HardwareModel, PackedOpSched, SchedulePolytope};
use crate::verifier::SuperOptError;

/// Per-op tile hint (diagnostic; the Level-3 tiling consumer is CEP-29
/// follow-up — emitted now so the search's decisions are observable).
///
/// CEP:WHAT: (op, tile dims, mem level, vec width) record.
/// CEP:WHY: Observability of the winner's placement decisions without
///          inventing XIR text syntax for schedules (the honest surface:
///          the pipeline consumes ClusterSet; tiling lands when level-3
///          grows a schedule-parameter input).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: none
/// CEP:COST: 24 bytes each
/// CEP:EVIDENCE: lib entry test asserts hints mirror the winner
/// CEP:SECURITY: no untrusted input
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileHint {
    /// The scheduled op.
    pub op: NodeId,
    /// Chosen tile dims.
    pub tile: [u32; 3],
    /// Chosen memory level discriminant.
    pub mem_level: u8,
    /// Chosen vector width.
    pub vec_width: u8,
}

/// CEP:WHAT: Builds the ClusterSet from the winner's fuse decisions.
/// CEP:WHY: Union-find over (producer -> fuse_target) pairs; every cluster's
///          budgets derive from the hardware model and member output sizes
///          (shared = min(SRAM cap, Σ member outputs), registers = a fixed
///          abstract allowance); nodes not participating in any fusion stay
///          unclustered (the projection's owner table reports None).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err on ClusterSet assignment failure (DuplicateNode /
///              UnknownNode — both indicate state corruption; loud).
/// CEP:ASSUMES: winner verified; arena is the extraction arena.
/// CEP:COST: O(pairs α + nodes).
/// CEP:EVIDENCE: `clusters_cover_fused_pairs`
/// CEP:SECURITY: no untrusted input.
pub fn build_clusters(
    arena: &IrArena,
    poly: &SchedulePolytope,
    winner: &[PackedOpSched],
    hw: HardwareModel,
) -> Result<ClusterSet, SuperOptError> {
    // Union-find over node slots (deterministic slot order).
    let n = arena.slot_count();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let find = |parent: &mut Vec<u32>, mut x: u32| -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    };
    let mut fused_pairs: Vec<(NodeId, NodeId)> = Vec::new();
    for s in winner.iter() {
        if s.fuse_target != PackedOpSched::NO_FUSE {
            fused_pairs.push((NodeId(s.op_id), NodeId(s.fuse_target)));
        }
    }
    for (p, c) in fused_pairs.iter() {
        let rp = find(&mut parent, p.index());
        let rc = find(&mut parent, c.index());
        if rp != rc {
            parent[rc.max(rp) as usize] = rp.min(rc);
        }
    }
    // Group members by root (only groups of size >= 2 become clusters).
    let mut groups: Vec<(u32, Vec<NodeId>)> = Vec::new();
    for s in poly.op_bounds.iter() {
        let root = find(&mut parent, s.op_id.index());
        if let Some(group) = groups.iter_mut().find(|(r, _)| *r == root) {
            group.1.push(s.op_id);
        } else {
            groups.push((root, vec![s.op_id]));
        }
    }
    for (p, c) in fused_pairs.iter() {
        let root = find(&mut parent, p.index());
        // Fuse targets may be non-scheduleable nodes — attach them to their
        // producer's group so the cluster covers both sides of every pair.
        if let Some(group) = groups.iter_mut().find(|(r, _)| *r == root) {
            if !group.1.contains(c) {
                group.1.push(*c);
            }
        }
    }
    let mut clusters = ClusterSet::new(arena);
    for (_, members) in groups.iter() {
        if members.len() < 2 {
            continue;
        }
        let shared = members
            .iter()
            .filter_map(|m| arena.node(*m).ok())
            .filter_map(|node| node.ty.as_tensor())
            .map(|t| {
                u64::from(t.elem.byte_size()).saturating_mul(t.shape.num_elements().max(0) as u64)
            })
            .fold(0u64, |a, b| a.saturating_add(b))
            .min(hw.sram_cap_bytes) as u32;
        let idx = clusters.add_cluster(shared, 1024);
        for m in members.iter() {
            clusters.assign(idx, *m).map_err(|_| {
                SuperOptError::ConstraintViolation(crate::polytope::ConstraintViolation::UnknownOp(
                    idx as u16,
                ))
            })?;
        }
    }
    clusters.derive_edges(arena);
    Ok(clusters)
}

/// CEP:WHAT: Tile hints from the winner (observable placement decisions).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (diagnostic).
/// CEP:ASSUMES: winner verified.
/// CEP:COST: O(ops)
/// CEP:EVIDENCE: lib entry test
/// CEP:SECURITY: no untrusted input
pub fn tile_hints(winner: &[PackedOpSched]) -> Vec<TileHint> {
    winner
        .iter()
        .map(|s| TileHint {
            op: NodeId(s.op_id),
            tile: s.tile,
            mem_level: s.mem_level,
            vec_width: s.vec_width,
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use crate::polytope::HardwareModel;
    use crate::verifier::verify_candidate;
    use crate::wpa::extract_polytope;
    use xir_core::text::parse_arena;

    const SRC: &str = concat!(
        "xir v1 func @main {\n",
        "  %0 = param 0 : tensor<f64>[8,8]\n",
        "  %1 = param 1 : tensor<f64>[8,8]\n",
        "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[8,8]\n",
        "  %3 = param 2 : tensor<f64>[8,8]\n",
        "  %4 = binary.add %2, %3 : tensor<f64>[8,8]\n",
        "}\n"
    );

    // CEP:WHAT: Clusters cover exactly the fused pairs (winner with one
    //           fusion -> one cluster containing matmul + add).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a missing/extra cluster member.
    // CEP:ASSUMES: matmul->add fusion is legal (can_fuse authority).
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn clusters_cover_fused_pairs() {
        let arena = parse_arena(SRC, 64).expect("parse");
        let poly = extract_polytope(&arena, HardwareModel::reference());
        // Winner: fuse matmul into add; everything else unfused.
        let add_id = poly.op_bounds[1].op_id;
        let winner: Vec<PackedOpSched> = poly
            .op_bounds
            .iter()
            .map(|b| {
                let mut s = PackedOpSched {
                    op_id: b.op_id.0,
                    tile: b.tile_max,
                    fuse_target: PackedOpSched::NO_FUSE,
                    mem_level: b.mem_max as u8,
                    vec_width: 1,
                };
                if b.op_id.0 == poly.op_bounds[0].op_id.0 {
                    s.fuse_target = add_id.0;
                }
                s
            })
            .collect();
        assert!(verify_candidate(&arena, &poly, &winner).is_ok());
        let clusters =
            build_clusters(&arena, &poly, &winner, HardwareModel::reference()).expect("build");
        assert_eq!(clusters.len(), 1, "one fused cluster");
        let members: Vec<NodeId> = clusters.cluster(0).expect("cluster 0").members.clone();
        assert!(members.contains(&poly.op_bounds[0].op_id));
        assert!(members.contains(&add_id));
        // Tile hints mirror the winner (per-op placements).
        let hints = tile_hints(&winner);
        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].op, poly.op_bounds[0].op_id);
        assert_eq!(hints[0].tile, poly.op_bounds[0].tile_max);
    }
}
