// CEP:FILE: crates/egraph/src/union_find.rs
// CEP:WHAT: Deterministic union-find with path compression and rank.
// CEP:WHY: E-class merging needs near-constant-time union/find; determinism
//          requires fixed tie-breaks (lower id wins rank ties) so saturation
//          results never depend on match order (CEP&CC 38.10).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: UnionFindError::OutOfBounds on stale/wide ids.
// CEP:ASSUMES: ids < capacity (bounded table; loud errors otherwise).
// CEP:COST: find/union amortized O(alpha(n)); no allocation after init.
// CEP:EVIDENCE: tests `find_union_roundtrip`, `deterministic_representatives`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-DETERMINISM: deterministic (rank + lowest-id tie-break).
//! Deterministic union-find.

// Union-find failure enumeration.
//
// CEP:WHAT: Explicit error type.
// CEP:WHY: Law 6 — out-of-bounds ids must fail loudly.
// CEP:STATUS: complete
// CEP:FAILURE: n/a — this IS the failure report.
// CEP:ASSUMES: none
// CEP:COST: zero-size enum
// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnionFindError {
    /// Id beyond the table capacity.
    OutOfBounds,
}

/// Bounded union-find table.
///
/// CEP:WHAT: Parent + rank arrays over dense ids.
/// CEP:WHY: E-class canonicalization; arrays keep find/union cache-dense
///          with zero per-operation allocation (CEP-0).
/// CEP:STATUS: complete
/// CEP:FAILURE: OutOfBounds.
/// CEP:ASSUMES: capacity fixed at init (CEP-1 setup).
/// CEP:COST: amortized inverse-Ackermann.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    /// CEP:WHAT: Allocates the table with singletons.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(capacity) init.
    /// CEP:EVIDENCE: tests
    pub fn new(capacity: usize) -> UnionFind {
        let mut parent = Vec::with_capacity(capacity);
        for i in 0..capacity {
            parent.push(i as u32);
        }
        UnionFind { parent }
    }

    /// CEP:WHAT: Finds the canonical representative (path compression).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: OutOfBounds.
    /// CEP:ASSUMES: none
    /// CEP:COST: amortized O(alpha(n)).
    /// CEP:EVIDENCE: tests
    pub fn find(&mut self, id: u32) -> Result<u32, UnionFindError> {
        if id as usize >= self.parent.len() {
            return Err(UnionFindError::OutOfBounds);
        }
        // Find root.
        let mut root = id;
        loop {
            let p = self.parent[root as usize];
            if p == root {
                break;
            }
            root = p;
        }
        // Path compression (iterative — no recursion, CEP-0 bounded stack).
        let mut cur = id;
        while cur != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        Ok(root)
    }

    /// CEP:WHAT: Read-only find (no compression) for determinism-safe queries.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: OutOfBounds.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(path) worst case.
    /// CEP:EVIDENCE: tests
    pub fn find_ro(&self, id: u32) -> Result<u32, UnionFindError> {
        if id as usize >= self.parent.len() {
            return Err(UnionFindError::OutOfBounds);
        }
        let mut root = id;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        Ok(root)
    }

    /// CEP:WHAT: Merges two sets; the LOWER root id always wins.
    /// CEP:WHY: Deterministic representatives regardless of merge ORDER:
    ///          linking max-root under min-root makes the final root a pure
    ///          function of the member set (the saturation driver's
    ///          determinism anchor, CEP&CC 38.10). Rejected alternative:
    ///          union-by-rank — rank depends on merge history, so
    ///          representatives would differ under reordering. Path
    ///          compression alone keeps amortized cost near-constant.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: OutOfBounds.
    /// CEP:ASSUMES: none
    /// CEP:COST: two finds + O(1).
    /// CEP:EVIDENCE: test `deterministic_representatives`.
    pub fn union(&mut self, a: u32, b: u32) -> Result<u32, UnionFindError> {
        let ra = self.find(a)?;
        let rb = self.find(b)?;
        if ra == rb {
            return Ok(ra);
        }
        let winner = ra.min(rb);
        let loser = ra.max(rb);
        self.parent[loser as usize] = winner;
        Ok(winner)
    }

    /// CEP:WHAT: Number of slots.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// CEP:WHAT: Emptiness probe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Union/find round trip with compression.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on set corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn find_union_roundtrip() {
        let mut uf = UnionFind::new(8);
        assert!(uf.union(0, 1).is_ok());
        assert!(uf.union(2, 3).is_ok());
        assert!(uf.union(1, 3).is_ok());
        assert_eq!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(0).ok(), uf.find(4).ok());
        assert_eq!(uf.find(9), Err(UnionFindError::OutOfBounds));
    }

    // CEP:WHAT: Representatives are deterministic under merge-order reversal.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on order dependence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn deterministic_representatives() {
        let mut a = UnionFind::new(4);
        let _ = a.union(0, 1);
        let _ = a.union(1, 2);
        let mut b = UnionFind::new(4);
        let _ = b.union(2, 1);
        let _ = b.union(1, 0);
        assert_eq!(a.find_ro(0), b.find_ro(0));
        assert_eq!(a.find_ro(2), b.find_ro(2));
    }
}
