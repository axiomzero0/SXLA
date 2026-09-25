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
        let mut h = xir_core::hash::Fnv64::new();
        op.hash_into(&mut h);
        for c in children.iter().take(4) {
            h.write_u32(*c);
        }
        let key = h.finish();
        if let Some(&existing) = self.lookup.get(&key) {
            // Congruent node: return its (canonical) class.
            let cls = *self
                .node_class
                .get(existing as usize)
                .ok_or(EgraphError::BadClass)?;
            return self.uf.find_ro(cls).map_err(EgraphError::Union);
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
}
