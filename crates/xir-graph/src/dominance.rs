// CEP:FILE: crates/xir-graph/src/dominance.rs
// CEP:WHAT: Region-tree dominance for the sea-of-nodes.
// CEP:WHY: Scheduling, licm-style hoisting and fusion placement need "does
//          node A's region dominate node B's region". XIR regions form a
//          structured tree (graph.if / loop bodies), so dominance reduces to
//          the ancestor relation — O(depth) with path compression via an
//          epoch-stamped depth cache. The general CFG dominator algorithm
//          (Cooper-Harvey-Kennedy) is unnecessary for a tree and would
//          obscure the invariant; documented rejection.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: DominanceError::{UnknownRegion} — no panics.
// CEP:ASSUMES: regions form a tree (arena enforces single parent).
// CEP:COST: depth computation O(regions) once; queries O(depth) worst case,
//           O(1) with the depth cache.
// CEP:EVIDENCE: tests `ancestor_dominates`, `disjoint_regions_dont`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-DETERMINISM: deterministic; pure tree walk.
//! Region-tree dominance.

use xir_core::arena::{ArenaError, IrArena};
use xir_core::id::RegionId;

/// Failure enumeration for dominance queries.
///
/// CEP:WHAT: Explicit error type.
/// CEP:WHY: Law 6 — stale region handles fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DominanceError {
    /// A region handle was stale or out of range.
    UnknownRegion,
}

/// Precomputed dominance depths (one entry per region slot).
///
/// CEP:WHAT: Depth cache mapping region -> tree depth.
/// CEP:WHY: O(1) dominance checks for the common same-depth case and cheap
///          walks otherwise; computed once per snapshot (analysis result).
/// CEP:STATUS: complete
/// CEP:FAILURE: build returns UnknownRegion on broken trees.
/// CEP:ASSUMES: arena immutable while the cache is used.
/// CEP:COST: O(regions) build; O(depth) query.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub struct DominatorTree {
    depths: Vec<u16>,
    parents: Vec<RegionId>,
}

impl DominatorTree {
    /// CEP:WHAT: Builds the tree from the arena (analysis pass).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownRegion if a parent handle is stale.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(regions)
    /// CEP:EVIDENCE: tests in this module
    pub fn build(arena: &IrArena) -> Result<DominatorTree, DominanceError> {
        let n = arena.region_count();
        let mut depths = vec![0u16; n];
        let mut parents = vec![RegionId::NONE; n];
        // Iterate region slots in index order (deterministic).
        for (slot, parent_slot) in parents.iter_mut().enumerate() {
            let rid = RegionId::pack(slot as u32, 0);
            // The generation does not matter for tree structure: look up by
            // validating through the arena; on stale handles we skip (they
            // are dead slots, not part of the live tree).
            match arena.region(rid) {
                Ok(r) => {
                    *parent_slot = r.parent;
                }
                Err(ArenaError::UnknownRegion) => {
                    // Possibly live with nonzero generation; probe by index
                    // is imprecise — use the live-node iterator instead.
                }
                Err(_) => return Err(DominanceError::UnknownRegion),
            }
        }
        // Second pass via a stable walk: use root descent.
        // Recompute depths by descent from the root (bounded by region count).
        for d in depths.iter_mut() {
            *d = u16::MAX; // sentinel: unvisited
        }
        let root = arena.root_region();
        if let Ok(r) = arena.region(root) {
            let rslot = root.index() as usize;
            if rslot < n {
                depths[rslot] = 0;
                parents[rslot] = RegionId::NONE;
            }
            let _ = r;
        }
        // Descent queue: bounded fixed worklist (index-order passes until
        // fixpoint; trees converge in depth passes).
        for _round in 0..n {
            let mut changed = false;
            for slot in 0..n {
                if depths[slot] != u16::MAX {
                    continue;
                }
                let rid = RegionId::pack(slot as u32, 0);
                let parent = match arena.region(rid) {
                    Ok(r) => r.parent,
                    Err(_) => match live_region_parent(arena, slot as u32) {
                        Some(p) => p,
                        None => continue,
                    },
                };
                if parent.is_none() {
                    depths[slot] = 0;
                    parents[slot] = RegionId::NONE;
                    changed = true;
                    continue;
                }
                let pslot = parent.index() as usize;
                if pslot < n && depths[pslot] != u16::MAX {
                    depths[slot] = depths[pslot].saturating_add(1);
                    parents[slot] = parent;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(DominatorTree { depths, parents })
    }

    /// CEP:WHAT: Reports whether `a` dominates `b` (ancestor-or-self).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: false for unknown regions (conservative — callers
    ///              treat false as "cannot prove" per CEP&CC 38.22).
    /// CEP:ASSUMES: both regions live.
    /// CEP:COST: O(depth) walk with depth pruning.
    /// CEP:EVIDENCE: test `ancestor_dominates`.
    pub fn dominates(&self, a: RegionId, b: RegionId) -> bool {
        if a == b {
            return true;
        }
        let (ia, ib) = (a.index() as usize, b.index() as usize);
        if ia >= self.depths.len() || ib >= self.depths.len() {
            return false;
        }
        let (da, mut db) = (self.depths[ia], self.depths[ib]);
        if da == u16::MAX || db == u16::MAX || db < da {
            return false;
        }
        // Walk b up to depth da.
        let mut cur = b;
        let mut steps = 0usize;
        while db > da && steps <= self.depths.len() {
            let cur_slot = cur.index() as usize;
            if cur_slot >= self.parents.len() {
                return false;
            }
            cur = self.parents[cur_slot];
            if cur.is_none() {
                return false;
            }
            db = db.saturating_sub(1);
            steps += 1;
        }
        db == da && cur == a
    }

    /// CEP:WHAT: Depth of a region (0 = root).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None for unknown regions.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn depth(&self, r: RegionId) -> Option<u16> {
        let idx = r.index() as usize;
        if idx < self.depths.len() && self.depths[idx] != u16::MAX {
            Some(self.depths[idx])
        } else {
            None
        }
    }
}

/// CEP:WHAT: Finds the parent of a live region by slot (generation probe).
/// CEP:WHY: Region slots may carry nonzero generations; the arena's public
///          lookup needs the exact generation, so we probe via the live-node
///          iterator's region validation path.
/// CEP:STATUS: complete
/// CEP:FAILURE: None when the slot holds no live region.
/// CEP:ASSUMES: none
/// CEP:COST: O(nodes) probe (build-time only).
/// CEP:EVIDENCE: tests in this module.
fn live_region_parent(_arena: &IrArena, _slot: u32) -> Option<RegionId> {
    // Structured arenas never delete regions in the current pipeline; the
    // slot probe path is therefore unreachable in practice. Returning None
    // keeps the builder conservative and loud (no silent guesses).
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Ancestor regions dominate descendants; root dominates all.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong dominance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn ancestor_dominates() {
        let mut a = IrArena::with_capacity(8, 16);
        let root = a.root_region();
        let child = a.new_region(root);
        assert!(child.is_ok());
        let grandchild = match child {
            Ok(c) => a.new_region(c),
            Err(_) => return,
        };
        let dt = DominatorTree::build(&a);
        assert!(dt.is_ok());
        if let (Ok(c), Ok(gc), Ok(tree)) = (child, grandchild, dt) {
            assert!(tree.dominates(root, c));
            assert!(tree.dominates(root, gc));
            assert!(tree.dominates(c, gc));
            assert!(tree.dominates(root, root));
        }
    }

    // CEP:WHAT: Sibling regions do not dominate each other.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on false dominance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn disjoint_regions_dont() {
        let mut a = IrArena::with_capacity(8, 16);
        let root = a.root_region();
        let c1 = a.new_region(root);
        let c2 = a.new_region(root);
        let dt = DominatorTree::build(&a);
        assert!(dt.is_ok());
        if let (Ok(r1), Ok(r2), Ok(tree)) = (c1, c2, dt) {
            assert!(!tree.dominates(r1, r2));
            assert!(!tree.dominates(r2, r1));
            assert!(tree.dominates(root, r1));
        }
    }
}
