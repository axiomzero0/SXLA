// CEP:FILE: crates/egraph/src/egraph.rs
// CEP:WHAT: E-class storage — e-nodes, e-classes, the union-find-backed
//           congruence structure.
// CEP:WHY: Equality saturation core (master architecture section 4): e-nodes
//          are ops with e-class children; classes merge via the deterministic
//          union-find; the rebuild step restores congruence after merges.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: EgraphError::{Full, BadClass} — bounded storage, loud errors.
// CEP:ASSUMES: only pure ops inserted (driver contract; Rng/custom excluded).
// CEP:COST: add O(children); merge O(1) + rebuild O(nodes).
// CEP:EVIDENCE: tests `add_and_lookup`, `merge_dedups_nodes`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-IR: The e-graph is an analysis overlay, not the durable IR.
// CEP:HPC-DETERMINISM: deterministic; sorted class member lists.
//! E-class storage.

use std::collections::BTreeMap;

use xir_core::op::Op;

use crate::union_find::{UnionFind, UnionFindError};

/// E-graph failure enumeration.
///
/// CEP:WHAT: Explicit error type for e-graph construction.
/// CEP:WHY: Law 6 — bounded node capacity and stale ids must fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgraphError {
    /// E-node capacity exhausted (bounded by the driver's budget).
    Full,
    /// A class id is unknown.
    BadClass,
    /// Union-find failure (out of bounds).
    Union(UnionFindError),
    /// The application driver's closing DCE sweep failed (apply path).
    Dce,
    /// add() was called with more than 4 children (audit round 4, F-8: the
    /// silent truncation was a data-loss trap; the Op vocabulary caps
    /// e-node children at 4).
    TooManyChildren,
}

/// One e-node: an op whose children are e-class ids.
///
/// CEP:WHAT: The e-graph node form.
/// CEP:WHY: Children are CLASS ids (not node ids) so merges automatically
///          re-key lookup — the standard e-graph representation.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: children reference live classes.
/// CEP:COST: 32 bytes.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ENode {
    /// Opcode + immediates.
    pub op: Op,
    /// Child e-class ids.
    pub children: [u32; 4],
    /// Live child count.
    pub n_children: u8,
    /// Monotonic insertion index (deterministic tie-break in extraction).
    pub seq: u32,
}

/// The e-graph.
///
/// CEP:WHAT: Nodes + classes + union-find + lookup map.
/// CEP:WHY: The saturation container; BTreeMap keyed by structural hash
///          keeps class canonicalization seed-independent (38.19) while
///          providing deterministic iteration.
/// CEP:STATUS: complete
/// CEP:FAILURE: see EgraphError.
/// CEP:ASSUMES: bounded node capacity (driver budget; CEP&CC 39).
/// CEP:COST: see module header.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub struct EGraph {
    nodes: Vec<ENode>,
    /// Node id -> owning class (the class returned by add()).
    node_class: Vec<u32>,
    /// Class -> sorted member node list (canonical class = union-find root).
    classes: BTreeMap<u32, Vec<u32>>,
    uf: UnionFind,
    /// Structural hash -> node id (congruence lookup).
    lookup: BTreeMap<u64, u32>,
    next_class: u32,
    capacity: usize,
}

/// CEP:WHAT: Op congruence with IEEE-exact float comparison.
/// CEP:WHY: Op derives PartialEq over an f64 payload, and NaN != NaN
///          under it — while the structural hash compares BIT PATTERNS
///          (identical NaNs hash equal). Derived equality therefore
///          rejects congruent NaN constants, systematically routing them
///          through the (conservative) hash-collision path and minting
///          duplicates every round (audit round 4, F-4). This helper
///          compares ConstF64 payloads by bits, matching the hash's
///          discipline; every other op compares by derived equality.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (total relation over Op).
/// CEP:ASSUMES: none.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: test `nan_consts_are_congruent`.
fn op_congruent(a: Op, b: Op) -> bool {
    if let (Op::ConstF64(x), Op::ConstF64(y)) = (a, b) {
        x.to_bits() == y.to_bits()
    } else {
        a == b
    }
}

impl EGraph {
    /// CEP:WHAT: Allocates an empty e-graph with a node budget.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: budget chosen by the saturation driver (Tier-2 config).
    /// CEP:COST: O(budget) init.
    /// CEP:EVIDENCE: tests
    pub fn new(node_budget: usize) -> EGraph {
        EGraph {
            nodes: Vec::with_capacity(node_budget),
            node_class: Vec::with_capacity(node_budget),
            classes: BTreeMap::new(),
            uf: UnionFind::new(node_budget * 2),
            lookup: BTreeMap::new(),
            next_class: 0,
            capacity: node_budget,
        }
    }

    /// CEP:WHAT: Inserts an e-node; returns its class.
    /// CEP:WHY: Saturation adds rewrites as new e-nodes; identical structure
    ///          deduplicates to the existing node's class (congruence).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Full when the budget is exhausted (bounded, loud).
    /// CEP:ASSUMES: children are canonical class ids.
    /// CEP:COST: O(log nodes) lookup + O(1) insert.
    /// CEP:EVIDENCE: tests `add_and_lookup`.
    pub fn add(&mut self, op: Op, children: &[u32]) -> Result<u32, EgraphError> {
        if self.nodes.len() >= self.capacity {
            return Err(EgraphError::Full);
        }
        // Loud arity contract (audit round 4, F-8): e-nodes carry at most
        // 4 children; truncating silently would corrupt congruence keys.
        if children.len() > 4 {
            return Err(EgraphError::TooManyChildren);
        }
        let mut h = xir_core::hash::Fnv64::new();
        op.hash_into(&mut h);
        for c in children.iter().take(4) {
            h.write_u32(*c);
        }
        let key = h.finish();
        if let Some(&existing) = self.lookup.get(&key) {
            // Full congruence proof (CEP-18; the GVN discipline of audit
            // F-4): the hash is only a filter — op, arity AND the full
            // child list must match before two nodes may share a class.
            // A hash collision alone can never dedup non-congruent nodes.
            let ex = self
                .nodes
                .get(existing as usize)
                .ok_or(EgraphError::BadClass)?;
            let live = children.len();
            let congruent = op_congruent(ex.op, op)
                && ex.n_children as usize == live
                && (0..live).all(|k| children[k] == ex.children[k]);
            if congruent {
                let cls = *self
                    .node_class
                    .get(existing as usize)
                    .ok_or(EgraphError::BadClass)?;
                return self.uf.find_ro(cls).map_err(EgraphError::Union);
            }
            // Hash collision with a NON-congruent node: fall through and
            // insert (the new node displaces the lookup entry; the
            // displaced node remains reachable through its class — lookup
            // is an accelerator, not the membership record).
        }
        let seq = self.nodes.len() as u32;
        let mut padded = [0u32; 4];
        for (i, c) in children.iter().take(4).enumerate() {
            padded[i] = *c;
        }
        // Each distinct node gets a fresh class initially.
        let class = self.next_class;
        self.next_class += 1;
        let node = ENode {
            op,
            children: padded,
            n_children: children.len().min(4) as u8,
            seq,
        };
        let node_id = seq;
        self.nodes.push(node);
        self.node_class.push(class);
        self.lookup.insert(key, node_id);
        self.classes.entry(class).or_default().push(node_id);
        Ok(class)
    }

    /// CEP:WHAT: Merges two classes (dedup + union-find).
    /// CEP:WHY: Rewrite rules equate classes; the merge moves membership to
    ///          the canonical (lowest) class.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Union error propagation.
    /// CEP:ASSUMES: both classes live.
    /// CEP:COST: O(members) move.
    /// CEP:EVIDENCE: test `merge_dedups_nodes`.
    pub fn merge(&mut self, a: u32, b: u32) -> Result<u32, EgraphError> {
        // Canonicalize BOTH arguments first (audit F-5): comparing the
        // union-find root against a raw (possibly stale) argument corrupted
        // the classes membership map when a stale id's root was the loser.
        let ca = self.uf.find(a).map_err(EgraphError::Union)?;
        let cb = self.uf.find(b).map_err(EgraphError::Union)?;
        let winner = self.uf.union(ca, cb).map_err(EgraphError::Union)?;
        // Move members of the loser class into the winner's list.
        let loser = if winner == ca { cb } else { ca };
        if let Some(members) = self.classes.remove(&loser) {
            let entry = self.classes.entry(winner).or_default();
            for m in members {
                entry.push(m);
            }
            entry.sort();
        }
        Ok(winner)
    }

    /// CEP:WHAT: Congruence rebuild — re-canonicalizes every e-node's
    ///           children, rebuilds the lookup map, and merges classes of
    ///           nodes that became congruent through prior merges.
    /// CEP:WHY: The e-graph rebuild step (arch section 4): merges invalidate
    ///          child class ids, so two nodes that differ ONLY in merged-
    ///          away child ids are congruent but invisible to `add`'s
    ///          lookup. rebuild() restores the congruence closure: each
    ///          e-node's children are rewritten to their canonical
    ///          (union-find root) classes; the lookup map is rebuilt from
    ///          scratch on the canonical keys; duplicates found under one
    ///          key are proven congruent (op + arity + full child list)
    ///          and merged union-by-min-id. Deterministic: nodes processed
    ///          in id order; winners are the lower class id.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: EgraphError propagation (Union / BadClass).
    /// CEP:ASSUMES: no concurrent mutation (driver calls between rounds).
    /// CEP:COST: O(nodes * (children + log nodes)) per call; bounded by
    ///           MAX_ROUNDS calls per saturation.
    /// CEP:EVIDENCE: test `rebuild_merges_congruent`; saturation tests
    ///           (folded constants make parents congruent).
    /// CEP:HPC-DETERMINISM: deterministic.
    pub fn rebuild(&mut self) -> Result<u32, EgraphError> {
        let n = self.nodes.len();
        // 1. Canonicalize every e-node's children in place.
        for i in 0..n {
            let live = self.nodes[i].n_children as usize;
            for j in 0..live.min(4) {
                let c = self.nodes[i].children[j];
                self.nodes[i].children[j] = self.uf.find(c).map_err(EgraphError::Union)?;
            }
        }
        // 2. Rebuild the lookup map on canonical keys; merge duplicates.
        let mut new_lookup: BTreeMap<u64, u32> = BTreeMap::new();
        let mut merged = 0u32;
        for i in 0..n {
            let (op, children, live) = {
                let node = self.nodes.get(i).ok_or(EgraphError::BadClass)?;
                (node.op, node.children, node.n_children as usize)
            };
            let mut h = xir_core::hash::Fnv64::new();
            op.hash_into(&mut h);
            for c in children.iter().take(live.min(4)) {
                h.write_u32(*c);
            }
            let key = h.finish();
            match new_lookup.get(&key).copied() {
                None => {
                    new_lookup.insert(key, i as u32);
                }
                Some(existing) => {
                    // Full congruence proof before merging.
                    let ex = self
                        .nodes
                        .get(existing as usize)
                        .ok_or(EgraphError::BadClass)?;
                    let congruent = op_congruent(ex.op, op)
                        && ex.n_children as usize == live.min(4)
                        && (0..live.min(4)).all(|k| children[k] == ex.children[k]);
                    if congruent {
                        let a = *self
                            .node_class
                            .get(existing as usize)
                            .ok_or(EgraphError::BadClass)?;
                        let b = *self.node_class.get(i).ok_or(EgraphError::BadClass)?;
                        let ca = self.uf.find_ro(a).map_err(EgraphError::Union)?;
                        let cb = self.uf.find_ro(b).map_err(EgraphError::Union)?;
                        if ca != cb {
                            self.merge(ca, cb)?;
                            merged += 1;
                        }
                    }
                    // Non-congruent hash collision: keep the earlier
                    // node's entry (deterministic).
                }
            }
        }
        self.lookup = new_lookup;
        Ok(merged)
    }

    /// CEP:WHAT: Canonical class id of a class.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Union error.
    /// CEP:ASSUMES: none
    /// CEP:COST: amortized O(alpha).
    /// CEP:EVIDENCE: tests
    pub fn canon(&mut self, class: u32) -> Result<u32, EgraphError> {
        self.uf.find(class).map_err(EgraphError::Union)
    }

    /// CEP:WHAT: Members (node ids) of a class, sorted.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: BadClass for unknown classes.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1) borrow.
    /// CEP:EVIDENCE: tests
    pub fn members(&self, class: u32) -> Result<&[u32], EgraphError> {
        match self.classes.get(&class) {
            Some(v) => Ok(v),
            None => Err(EgraphError::BadClass),
        }
    }

    /// CEP:WHAT: Borrows an e-node by id.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: BadClass (reused error kind) for unknown node ids.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn node(&self, id: u32) -> Result<&ENode, EgraphError> {
        self.nodes.get(id as usize).ok_or(EgraphError::BadClass)
    }

    /// CEP:WHAT: Live node count.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// CEP:WHAT: Canonical class list in ascending order (deterministic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(classes)
    /// CEP:EVIDENCE: saturation driver tests.
    pub fn class_ids(&self) -> Vec<u32> {
        self.classes.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::op::BinaryOp;

    // CEP:WHAT: Nodes add, look up and deduplicate structurally.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on dedup failure.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn add_and_lookup() {
        let mut g = EGraph::new(32);
        let c0 = g.add(Op::ConstI64(1), &[]);
        assert!(c0.is_ok());
        let c1 = g.add(Op::ConstI64(2), &[]);
        assert!(c1.is_ok());
        if let (Ok(k0), Ok(k1)) = (c0, c1) {
            let add = g.add(Op::Binary(BinaryOp::Add), &[k0, k1]);
            assert!(add.is_ok());
            // Same structure again: dedups to the same class.
            let add2 = g.add(Op::Binary(BinaryOp::Add), &[k0, k1]);
            assert!(add2.is_ok());
            assert_eq!(add.ok(), add2.ok());
        }
        assert_eq!(g.node_count(), 3);
    }

    // CEP:WHAT: Merging classes moves membership to the canonical class.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on membership loss.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn merge_dedups_nodes() {
        let mut g = EGraph::new(32);
        let a = g.add(Op::ConstI64(1), &[]);
        let b = g.add(Op::ConstI64(1), &[]);
        // Distinct adds of the SAME const dedup already; simulate a merge of
        // two different classes instead.
        let c = g.add(Op::ConstI64(3), &[]);
        if let (Ok(ka), Ok(_kb), Ok(kc)) = (a, b, c) {
            let winner = g.merge(ka, kc);
            assert!(winner.is_ok());
            if let Ok(w) = winner {
                assert_eq!(w, ka.min(kc));
                let members = g.members(w);
                assert!(members.is_ok());
                if let Ok(m) = members {
                    assert_eq!(m.len(), 2);
                }
            }
        }
    }

    // CEP:WHAT: rebuild() merges nodes that became congruent through a
    //           prior merge: two adds differing only in merged-away child
    //           classes become one class (the congruence closure).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if rebuild misses the congruence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rebuild_merges_congruent() {
        let mut g = EGraph::new(32);
        // The real scenario: a RULE merges two structurally-distinct but
        // equal classes (here: const 5 and add(2,3) — what a fold does);
        // then two PARENT adds are inserted, one with the canonical id and
        // one with the stale id. They carry different lookup keys, so add()
        // cannot dedup them; rebuild's canonicalization must.
        let x = g.add(Op::ConstI64(9), &[]);
        let k5 = g.add(Op::ConstI64(5), &[]);
        let c2 = g.add(Op::ConstI64(2), &[]);
        let c3 = g.add(Op::ConstI64(3), &[]);
        assert!(x.is_ok() && k5.is_ok() && c2.is_ok() && c3.is_ok());
        if let (Ok(kx), Ok(a), Ok(b), Ok(c)) = (x, k5, c2, c3) {
            // sum = add(2, 3): structurally distinct from const 5.
            let sum = g.add(Op::Binary(BinaryOp::Add), &[b, c]);
            assert!(sum.is_ok());
            if let Ok(s) = sum {
                // The fold-equivalent merge: const5 class == sum class.
                let merged = g.merge(a, s);
                assert!(merged.is_ok());
                // Parents: one with the canonical id, one with the stale.
                let p1 = g.add(Op::Binary(BinaryOp::Mul), &[a, kx]);
                let p2 = g.add(Op::Binary(BinaryOp::Mul), &[s, kx]);
                assert!(p1.is_ok() && p2.is_ok());
                if let (Ok(q1), Ok(q2)) = (p1, p2) {
                    // Pre-rebuild: distinct classes (the stale child id
                    // made different lookup keys).
                    assert_ne!(g.canon(q1).ok(), g.canon(q2).ok());
                    let m = g.rebuild();
                    assert!(m.is_ok());
                    if let Ok(merged_count) = m {
                        assert!(merged_count >= 1, "rebuild must find the congruence");
                        // Post-rebuild: same class, children canonicalized.
                        assert_eq!(g.canon(q1).ok(), g.canon(q2).ok());
                    }
                }
            }
        }
    }

    // CEP:WHAT: rebuild() is idempotent (a second call merges nothing).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on non-idempotence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn rebuild_is_idempotent() {
        let mut g = EGraph::new(32);
        let x = g.add(Op::ConstI64(9), &[]);
        let k5 = g.add(Op::ConstI64(5), &[]);
        let c2 = g.add(Op::ConstI64(2), &[]);
        let c3 = g.add(Op::ConstI64(3), &[]);
        if let (Ok(kx), Ok(a), Ok(b), Ok(c)) = (x, k5, c2, c3) {
            let sum = g.add(Op::Binary(BinaryOp::Add), &[b, c]);
            if let Ok(s) = sum {
                let _ = g.merge(a, s);
                let _p1 = g.add(Op::Binary(BinaryOp::Mul), &[a, kx]);
                let _p2 = g.add(Op::Binary(BinaryOp::Mul), &[s, kx]);
                let first = g.rebuild();
                let second = g.rebuild();
                assert!(first.is_ok() && second.is_ok());
                if let (Ok(f), Ok(sd)) = (first, second) {
                    assert!(f >= 1);
                    assert_eq!(sd, 0, "second rebuild must be a no-op");
                }
            }
        }
    }

    // CEP:WHAT: Identical NaN constants are CONGRUENT (bit comparison),
    //           not hash collisions (audit round 4, F-4): derived PartialEq
    //           would reject NaN == NaN and mint duplicates forever.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if NaN consts fail to dedup.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn nan_consts_are_congruent() {
        let mut g = EGraph::new(32);
        let nan = f64::from_bits(0x7FF8_0000_0000_0001);
        let a = g.add(Op::ConstF64(nan), &[]);
        let b = g.add(Op::ConstF64(nan), &[]);
        assert!(a.is_ok() && b.is_ok());
        if let (Ok(ka), Ok(kb)) = (a, b) {
            // Same bits: dedup to ONE class, ONE node.
            assert_eq!(ka, kb);
            assert_eq!(g.node_count(), 1);
        }
        // Different NaN payloads stay distinct.
        let nan2 = f64::from_bits(0x7FF8_0000_0000_0002);
        let c = g.add(Op::ConstF64(nan2), &[]);
        assert!(c.is_ok());
        if let (Ok(kc), Ok(ka)) = (c, a) {
            assert_ne!(kc, ka);
            assert_eq!(g.node_count(), 2);
        }
    }
}
