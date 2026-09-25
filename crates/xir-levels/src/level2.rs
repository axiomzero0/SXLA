// CEP:FILE: crates/xir-levels/src/level2.rs
// CEP:WHAT: Level-2 fusion cluster hypergraph — cluster sets, edges,
//           materialization decisions.
// CEP:WHY: Master architecture Level 2: "Hypergraph (Clusters containing
//          subgraphs)... Explicitly represents fusion decisions, resource
//          budgets, and schedule hints." The fusion crate's search produces
//           these structures; codegen consumes them to shape loop nests.
// CEP:CLASS: CEP-0 (data structures)
// CEP:STATUS: complete
// CEP:FAILURE: ClusterError::{DuplicateNode, UnknownNode} — loud, no
//             silent double-assignment of a node to two clusters.
// CEP:ASSUMES: clusters partition the node set (verifier-enforced at the
//           fusion commit boundary).
// CEP:COST: O(nodes) build; O(1) cluster-of queries.
// CEP:EVIDENCE: tests `clusters_partition_nodes`, `edges_follow_uses`.
// CEP:SECURITY: internal ids only.
// CEP:HPC-IR: Level-2 form: hypergraph of clusters.
// CEP:HPC-DETERMINISM: deterministic; sorted cluster membership.
//! Level-2 fusion cluster hypergraph.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;

/// Cluster structure failure enumeration.
///
/// CEP:WHAT: Explicit error type for cluster construction.
/// CEP:WHY: Law 6 — partition violations must be loud (a node in two
///          clusters is an ambiguous lowering contract, Severity 0
///          material).
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterError {
    /// The same node was assigned to two clusters.
    DuplicateNode,
    /// A referenced node is unknown.
    UnknownNode,
}

/// One fusion cluster: a set of nodes scheduled into one kernel.
///
/// CEP:WHAT: Cluster membership + budget metadata.
/// CEP:WHY: The hypergraph node (arch: fusion.cluster). Members are sorted
///          NodeIds (deterministic iteration, CEP&CC 38.19).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: membership is exclusive across clusters (enforced by
///           ClusterSet::add).
/// CEP:COST: O(members) memory.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Sorted member node ids.
    pub members: Vec<NodeId>,
    /// Shared-memory budget in bytes assigned by the resource model.
    pub shared_budget: u32,
    /// Register budget (abstract units).
    pub register_budget: u32,
    /// True when the cluster must write its outputs to global memory
    /// (fusion.materialize decision).
    pub materialize: bool,
}

/// The cluster hypergraph.
///
/// CEP:WHAT: Cluster set + producer/consumer edges between clusters.
/// CEP:WHY: Edges carry the dataflow that crosses cluster boundaries —
///          exactly the edges that cost memory traffic in the cost model.
/// CEP:STATUS: complete
/// CEP:FAILURE: see ClusterError.
/// CEP:ASSUMES: nodes belong to the referenced arena.
/// CEP:COST: O(nodes + edges).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic; sorted structures.
pub struct ClusterSet {
    /// Clusters in creation (search) order — the search is deterministic.
    clusters: Vec<Cluster>,
    /// Node slot -> cluster index.
    owner: Vec<Option<u32>>,
    /// Directed edges: (from_cluster, to_cluster) deduplicated + sorted.
    edges: Vec<(u32, u32)>,
}

impl ClusterSet {
    /// CEP:WHAT: Builds an empty cluster set sized to the arena.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: arena immutable while the set is used.
    /// CEP:COST: O(nodes) allocation (CEP-1 setup).
    /// CEP:EVIDENCE: tests
    pub fn new(arena: &IrArena) -> ClusterSet {
        ClusterSet {
            clusters: Vec::new(),
            owner: vec![None; arena.slot_count()],
            edges: Vec::new(),
        }
    }

    /// CEP:WHAT: Builds a truly empty cluster set (no clusters, no owner
    ///           table — every node reports unclustered).
    /// CEP:WHY: The cluster-free projection path (`project`) and tests
    ///          need a zero-sized sentinel without an arena reference;
    ///          `cluster_of` returns None for every node by construction.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: never assigned (assign() would reject the zero table).
    /// CEP:COST: O(1).
    /// CEP:EVIDENCE: level3 fused-projection tests use it as the baseline.
    pub fn empty() -> ClusterSet {
        ClusterSet {
            clusters: Vec::new(),
            owner: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// CEP:WHAT: Adds an empty cluster and returns its index.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1) amortized
    /// CEP:EVIDENCE: tests
    pub fn add_cluster(&mut self, shared_budget: u32, register_budget: u32) -> u32 {
        self.clusters.push(Cluster {
            members: Vec::new(),
            shared_budget,
            register_budget,
            materialize: false,
        });
        (self.clusters.len() - 1) as u32
    }

    /// CEP:WHAT: Assigns a node to a cluster.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: DuplicateNode when the node is already owned.
    /// CEP:ASSUMES: node live.
    /// CEP:COST: O(1).
    /// CEP:EVIDENCE: test `clusters_partition_nodes`.
    pub fn assign(&mut self, cluster: u32, node: NodeId) -> Result<(), ClusterError> {
        let slot = node.index() as usize;
        if slot >= self.owner.len() {
            return Err(ClusterError::UnknownNode);
        }
        if self.owner[slot].is_some() {
            return Err(ClusterError::DuplicateNode);
        }
        self.owner[slot] = Some(cluster);
        if let Some(c) = self.clusters.get_mut(cluster as usize) {
            c.members.push(node);
            // Keep membership sorted for deterministic iteration.
            c.members.sort_by_key(|n| n.index());
        }
        Ok(())
    }

    /// CEP:WHAT: Derives inter-cluster edges from value uses.
    /// CEP:WHY: Edges are the materialized dataflow — the cost model's
    ///          memory-traffic term; derived deterministically from uses.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none; uses of unclustered nodes are ignored (they are
    ///              roots/params).
    /// CEP:ASSUMES: all relevant nodes assigned.
    /// CEP:COST: O(nodes + edges log edges) (sort).
    /// CEP:EVIDENCE: test `edges_follow_uses`.
    pub fn derive_edges(&mut self, arena: &IrArena) {
        let mut edges: Vec<(u32, u32)> = Vec::new();
        arena.for_each_live_node(|id, node| {
            let consumer = match self.cluster_of(id) {
                Some(c) => c,
                None => return,
            };
            for i in 0..node.n_inputs as usize {
                if i >= xir_core::node::MAX_INPUTS {
                    break;
                }
                let producer = node.inputs[i].node();
                if let Some(p) = self.cluster_of(producer) {
                    if p != consumer {
                        edges.push((p, consumer));
                    }
                }
            }
        });
        edges.sort();
        edges.dedup();
        self.edges = edges;
    }

    /// CEP:WHAT: Cluster index owning a node (None if unclustered).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn cluster_of(&self, node: NodeId) -> Option<u32> {
        self.owner.get(node.index() as usize).copied().flatten()
    }

    /// CEP:WHAT: Number of clusters.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// CEP:WHAT: Emptiness probe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }

    /// CEP:WHAT: Sets a cluster's force-materialization flag.
    /// CEP:WHY: The resource model / repair pass marks over-budget or
    ///          halo-heavy clusters: their intermediates MUST write to
    ///          global memory even when consumed in-cluster (the
    ///          bufferization escape hatch). Exposed as an API so the
    ///          search driver and tests set it through the same contract.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (out-of-range index is a silent no-op — the
    ///              caller's cluster handle came from add_cluster).
    /// CEP:ASSUMES: cluster index from add_cluster.
    /// CEP:COST: O(1).
    /// CEP:EVIDENCE: level3 `materialized_cluster_forces_global`.
    pub fn set_materialize(&mut self, cluster: u32, materialize: bool) {
        if let Some(c) = self.clusters.get_mut(cluster as usize) {
            c.materialize = materialize;
        }
    }

    /// CEP:WHAT: Borrows a cluster by index.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: None when out of range.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn cluster(&self, index: u32) -> Option<&Cluster> {
        self.clusters.get(index as usize)
    }

    /// CEP:WHAT: The derived inter-cluster edges (sorted, deduplicated).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: derive_edges ran.
    /// CEP:COST: O(1) borrow
    /// CEP:EVIDENCE: tests
    pub fn edges(&self) -> &[(u32, u32)] {
        &self.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::{const_f64, const_i64};
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};

    // CEP:WHAT: Nodes land in exactly one cluster; double-assign fails.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on partition violation.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn clusters_partition_nodes() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 1);
        let c1 = const_i64(&mut a, root, 2);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let mut cs = ClusterSet::new(&a);
            let k0 = cs.add_cluster(0, 0);
            let k1 = cs.add_cluster(0, 0);
            assert!(cs.assign(k0, v0).is_ok());
            assert!(cs.assign(k1, v1).is_ok());
            assert_eq!(cs.cluster_of(v0), Some(k0));
            assert_eq!(cs.cluster_of(v1), Some(k1));
            // Double assignment is loud.
            assert_eq!(cs.assign(k1, v0), Err(ClusterError::DuplicateNode));
            assert_eq!(cs.len(), 2);
            assert!(!cs.is_empty());
        }
    }

    // CEP:WHAT: Edges follow producer/consumer uses across clusters.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on missing or spurious edges.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn edges_follow_uses() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let c0 = const_f64(&mut a, root, 1.0);
        let c1 = const_f64(&mut a, root, 2.0);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::F64),
                );
                let add_id = a.insert_node(root, add);
                assert!(add_id.is_ok());
                let mut cs = ClusterSet::new(&a);
                let kprod = cs.add_cluster(0, 0);
                let kcons = cs.add_cluster(0, 0);
                assert!(cs.assign(kprod, v0).is_ok());
                assert!(cs.assign(kprod, v1).is_ok());
                if let Ok(av) = add_id {
                    assert!(cs.assign(kcons, av).is_ok());
                }
                cs.derive_edges(&a);
                assert_eq!(cs.edges(), &[(kprod, kcons)]);
            }
        }
    }
}
