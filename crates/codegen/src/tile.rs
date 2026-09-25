// CEP:FILE: crates/codegen/src/tile.rs
// CEP:WHAT: Tiling — splits tensor domains into cache-friendly tiles.
// CEP:WHY: Master architecture Level 3 pass list; tiles are the loop
//          transformation the resource model budgets (SHARED_MEM_BUDGET).
//          Tile edges derive from the cache-line constant (Law 7: no
//          magic numbers).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: TileError::{RankTooLarge} — loud.
// CEP:ASSUMES: dims positive (Shape guarantees).
// CEP:COST: O(rank) per tensor.
// CEP:EVIDENCE: tests `tiles_fit_budget`, `tile_edges_from_cache_line`.
// CEP:SECURITY: internal only.
// CEP:HPC-DETERMINISM: deterministic.
//! Tiling pass.

use xir_core::ty::{Shape, MAX_RANK};

/// Tile edge in elements, derived from the cache-line floor.
///
/// CEP:WHAT: Tile edge constant.
/// CEP:WHY: A 64-byte line holds 8 f64 elements: tile edges of 64 elements
///          cover 8 lines — the smallest edge that amortizes line fetches
///          while keeping shared-memory tiles small (calibration in
///          docs/cost_model.md; Law 7).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: f64 reference element (documented).
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: test `tile_edges_from_cache_line`
pub const TILE_CACHE_LINE: i64 = 64;

/// Tiling failure enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileError {
    /// Rank exceeded the bounded shape representation.
    RankTooLarge,
}

/// The tiling plan for one tensor.
///
/// CEP:WHAT: Per-axis tile edges.
/// CEP:WHY: Loop lowering nests per tile with interior point loops; the
///          resource model checks tile bytes against the budget.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: none
/// CEP:COST: plain data
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePlan {
    /// Tile edges per axis (same rank as the tensor).
    pub edges: [i64; MAX_RANK],
    /// Live rank.
    pub rank: u8,
}

/// CEP:WHAT: Computes tile edges for a shape under a byte budget.
/// CEP:WHY: Greedy axis-major tiling: start from TILE_CACHE_LINE edges and
///          halve until the tile fits the budget — deterministic and
///          budget-respecting (CEP&CC 39).
/// CEP:STATUS: complete
/// CEP:FAILURE: RankTooLarge never here (Shape bounds rank) — reserved for
///              future strided forms.
/// CEP:ASSUMES: budget in bytes; element width 8 (f64 reference).
/// CEP:COST: O(rank * log(edges)).
/// CEP:EVIDENCE: test `tiles_fit_budget`.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn tile_shapes(shape: &Shape, budget_bytes: u32) -> Result<TilePlan, TileError> {
    let rank = shape.rank();
    let mut edges = [TILE_CACHE_LINE; MAX_RANK];
    loop {
        let mut tile_elems: i64 = 1;
        for (a, edge) in edges.iter().enumerate().take(rank as usize) {
            tile_elems = tile_elems.saturating_mul((*edge).min(shape.dim(a as u8).unwrap_or(1)));
        }
        let bytes = tile_elems.saturating_mul(8);
        if bytes <= budget_bytes as i64 || edges == [1; MAX_RANK] {
            break;
        }
        // Halve the LARGEST edge (deterministic: first largest).
        let mut largest = 0usize;
        for a in 1..rank as usize {
            if edges[a] > edges[largest] {
                largest = a;
            }
        }
        edges[largest] = (edges[largest] / 2).max(1);
    }
    Ok(TilePlan { edges, rank })
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Generated tiles respect the byte budget.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on budget violation.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tiles_fit_budget() {
        let s = Shape::from_dims(&[1024, 1024]);
        assert!(s.is_ok());
        if let Ok(shape) = s {
            let plan = tile_shapes(&shape, 16 * 1024);
            assert!(plan.is_ok());
            if let Ok(p) = plan {
                let mut elems: i64 = 1;
                for a in 0..p.rank as usize {
                    let d = shape.dim(a as u8).unwrap_or(1);
                    elems = elems.saturating_mul(p.edges[a].min(d));
                }
                assert!(elems * 8 <= 16 * 1024);
            }
        }
    }

    // CEP:WHAT: Tile edges derive from the cache-line constant.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on calibration drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tile_edges_from_cache_line() {
        let s = Shape::from_dims(&[8, 8]);
        if let Ok(shape) = s {
            // Small tensor, huge budget: edges stay at the constant.
            if let Ok(p) = tile_shapes(&shape, u32::MAX / 8) {
                assert_eq!(p.edges[0], TILE_CACHE_LINE);
            }
        }
    }
}
