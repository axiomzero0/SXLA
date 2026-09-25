// CEP:FILE: crates/xir-core/src/arena.rs
// CEP:WHAT: IrArena — capacity-bounded, generation-guarded node/region storage.
// CEP:WHY: The master architecture mandates arena-allocated IR with zero
//          Box/Arc per node. This arena preallocates node and region capacity
//          at construction (the single allocation event, CEP&CC 25.5) and
//          every subsequent insertion is a bounds-checked slot write —
//          loud `ArenaError::Exhausted` instead of hidden reallocation
//          (Law 1). Generation bumping on free-slot reuse makes stale
//          NodeIds detectable instead of silently aliased (Law 2).
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: ArenaError::{Exhausted, InputOverflow, OutputOverflow,
//              UnknownNode, UnknownRegion} — explicit, no panics.
// CEP:ASSUMES: single-threaded mutation per compilation context; snapshots
//              (immutable views) cross threads via Arc in snapshot.rs (CEP-1).
// CEP:COST: insert = free-list pop or append + slot write, O(1); lookup =
//           2 compares + 1 load; no allocation after `with_capacity`.
// CEP:EVIDENCE: tests `insert_lookup_roundtrip`, `generation_reuse_detected`,
//           `capacity_is_bounded`, `region_tree`.
// CEP:SECURITY: indices are internal; parser-validated data only.
// CEP:HPC-DETERMINISM: deterministic; ids derive from insertion order.
//! Bounded IR arena.

use crate::id::{NodeId, RegionId, ValueId};
use crate::node::{Node, MAX_INPUTS, MAX_OUTPUTS};
use crate::op::Op;

/// Failure enumeration for arena operations.
///
/// CEP:WHAT: Explicit error type for arena mutation/lookup.
/// CEP:WHY: Law 6 — bounded capacity and stale handles must fail loudly.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    /// Node or region capacity exhausted (bounded storage).
    Exhausted,
    /// More than MAX_INPUTS value inputs requested.
    InputOverflow,
    /// More than MAX_OUTPUTS result slots requested.
    OutputOverflow,
    /// NodeId lookup failed (stale generation or out of range).
    UnknownNode,
    /// RegionId lookup failed.
    UnknownRegion,
}

/// Control region record.
///
/// CEP:WHAT: One node of the control tree (body / then / else / loop).
/// CEP:WHY: graph.if region nodes (arch Level 0) and Level-3 loop nests both
///          structure control via this tree; children nest by id.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (arena validates).
/// CEP:ASSUMES: parent NONE marks the root region.
/// CEP:COST: 24 bytes.
/// CEP:EVIDENCE: test `region_tree`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    /// Parent region (NONE for root).
    pub parent: RegionId,
    /// First node in this region's intrusive list.
    pub first_node: NodeId,
    /// Monotonic slot generation for id reuse detection.
    pub generation: u16,
}

/// The node/region arena.
///
/// CEP:WHAT: Bounded slot storage with free lists and generation guarding.
/// CEP:WHY: See module header. Free lists reuse deleted slots deterministically
///          (LIFO) with a generation bump so old handles become invalid.
/// CEP:STATUS: complete
/// CEP:FAILURE: see ArenaError.
/// CEP:ASSUMES: see module header.
/// CEP:COST: see module header.
/// CEP:EVIDENCE: module tests.
/// CEP:SECURITY: no pointers stored.
/// CEP:HPC-DETERMINISM: deterministic; insertion order defines ids.
pub struct IrArena {
    nodes: Vec<Node>,
    node_gen: Vec<u16>,
    node_free: Vec<u32>,
    regions: Vec<Region>,
    region_free: Vec<u32>,
    next_region_gen: Vec<u16>,
    root: RegionId,
}

impl IrArena {
    /// CEP:WHAT: Allocates the arena with node/region capacity (init boundary).
    /// CEP:WHY: The one allocation event; all later inserts are slot writes.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: panics on zero capacity (programmer error at init, CEP-1
    ///              boundary — documented).
    /// CEP:ASSUMES: capacities sized by the stage runner against expected IR
    ///              volume (jit crate policy).
    /// CEP:COST: two allocations; O(capacity) zeroing.
    /// CEP:EVIDENCE: tests in this module.
    /// CEP:SECURITY: capacity caller-controlled, bounded by driver policy.
    pub fn with_capacity(node_capacity: usize, region_capacity: usize) -> IrArena {
        debug_assert!(node_capacity > 0 && region_capacity > 0);
        let nodes = Vec::with_capacity(node_capacity);
        let regions = Vec::with_capacity(region_capacity);
        let mut arena = IrArena {
            nodes,
            node_gen: Vec::with_capacity(node_capacity),
            node_free: Vec::with_capacity(node_capacity / 4 + 1),
            regions,
            region_free: Vec::with_capacity(region_capacity / 4 + 1),
            next_region_gen: Vec::with_capacity(region_capacity),
            root: RegionId::NONE,
        };
        // Root region at slot 0, generation 0.
        if let Ok(root) = arena.new_region(RegionId::NONE) {
            arena.root = root;
        }
        arena
    }

    /// CEP:WHAT: The root region id.
    /// CEP:WHY: Builders attach top-level nodes to the root.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests in this module
    pub fn root_region(&self) -> RegionId {
        self.root
    }

    /// CEP:WHAT: Creates a child region.
    /// CEP:WHY: graph.if then/else bodies, loop bodies (arch regions).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Exhausted when region capacity is full.
    /// CEP:ASSUMES: parent exists or is NONE (root's parent).
    /// CEP:COST: append or free-pop; O(1).
    /// CEP:EVIDENCE: test `region_tree`.
    pub fn new_region(&mut self, parent: RegionId) -> Result<RegionId, ArenaError> {
        if self.regions.len() >= self.regions.capacity() {
            return Err(ArenaError::Exhausted);
        }
        let (index, generation) = if let Some(idx) = self.region_free.pop() {
            let gen = self.next_region_gen[idx as usize].wrapping_add(1);
            self.next_region_gen[idx as usize] = gen;
            (idx, gen)
        } else {
            let idx = self.regions.len() as u32;
            self.next_region_gen.push(0);
            (idx, 0)
        };
        if self.regions.len() as u32 <= index {
            // grow up to index (only when index == len)
            if self.regions.len() as u32 != index || self.regions.len() >= self.regions.capacity() {
                return Err(ArenaError::Exhausted);
            }
            self.regions.push(Region {
                parent,
                first_node: NodeId::NONE,
                generation,
            });
        } else {
            self.regions[index as usize] = Region {
                parent,
                first_node: NodeId::NONE,
                generation,
            };
        }
        Ok(RegionId::pack(index, generation))
    }

    /// CEP:WHAT: Inserts a node into a region.
    /// CEP:WHY: Node construction; links the node at the head of the region's
    ///          intrusive list (deterministic reverse insertion order — the
    ///          scheduler reorders canonically before printing).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Exhausted / InputOverflow / UnknownRegion.
    /// CEP:ASSUMES: node inputs already validated to <= MAX_INPUTS by
    ///              Node::new's padding (arity re-checked here for defense).
    /// CEP:COST: O(1).
    /// CEP:EVIDENCE: tests `insert_lookup_roundtrip`, `capacity_is_bounded`.
    pub fn insert_node(&mut self, region: RegionId, node: Node) -> Result<NodeId, ArenaError> {
        if node.n_inputs as usize > MAX_INPUTS {
            return Err(ArenaError::InputOverflow);
        }
        let (region_slot, region_gen) = self.validate_region(region)?;
        if self.nodes.len() >= self.nodes.capacity() {
            return Err(ArenaError::Exhausted);
        }
        let mut node = node;
        // Region must exist and match generation.
        let r = &self.regions[region_slot as usize];
        if r.generation != region_gen {
            return Err(ArenaError::UnknownRegion);
        }
        let (index, generation) = if let Some(idx) = self.node_free.pop() {
            let gen = self.node_gen[idx as usize].wrapping_add(1);
            self.node_gen[idx as usize] = gen;
            (idx, gen)
        } else {
            let idx = self.nodes.len() as u32;
            self.node_gen.push(0);
            (idx, 0)
        };
        if index as usize == self.nodes.len() {
            // Head insertion into the region's list.
            node.next_in_region = r.first_node;
            node.region = region;
            let id = NodeId::pack(index, generation, node.op.level());
            self.nodes.push(node);
            self.regions[region_slot as usize].first_node = id;
            Ok(id)
        } else if (index as usize) < self.nodes.len() {
            // Reused slot: same linking discipline.
            let first = r.first_node;
            node.next_in_region = first;
            node.region = region;
            let id = NodeId::pack(index, generation, node.op.level());
            self.nodes[index as usize] = node;
            self.regions[region_slot as usize].first_node = id;
            Ok(id)
        } else {
            Err(ArenaError::Exhausted)
        }
    }

    /// CEP:WHAT: Looks a node up by handle (generation-checked).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownNode on stale/out-of-range handles.
    /// CEP:ASSUMES: id came from insert_node.
    /// CEP:COST: 2 compares + 1 load.
    /// CEP:EVIDENCE: test `generation_reuse_detected`.
    pub fn node(&self, id: NodeId) -> Result<&Node, ArenaError> {
        let idx = id.index() as usize;
        if idx >= self.nodes.len() {
            return Err(ArenaError::UnknownNode);
        }
        if self.node_gen[idx] != id.generation() {
            return Err(ArenaError::UnknownNode);
        }
        Ok(&self.nodes[idx])
    }

    /// CEP:WHAT: Mutable node lookup (pass editing).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownNode (as node()).
    /// CEP:ASSUMES: single-threaded mutation (module contract).
    /// CEP:COST: as node().
    /// CEP:EVIDENCE: pass tests in xir-graph.
    pub fn node_mut(&mut self, id: NodeId) -> Result<&mut Node, ArenaError> {
        let idx = id.index() as usize;
        if idx >= self.nodes.len() {
            return Err(ArenaError::UnknownNode);
        }
        if self.node_gen[idx] != id.generation() {
            return Err(ArenaError::UnknownNode);
        }
        Ok(&mut self.nodes[idx])
    }

    /// CEP:WHAT: Removes a node and bumps its slot generation.
    /// CEP:WHY: DCE and rewrites free slots; the generation bump invalidates
    ///          every outstanding handle to the deleted node (Law 2).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownNode; does not repair the region list (the caller's
    ///              edit log must unlink first — verifier checks).
    /// CEP:ASSUMES: caller unlinked the node from its region list.
    /// CEP:COST: O(1) + free-list push.
    /// CEP:EVIDENCE: test `generation_reuse_detected`.
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), ArenaError> {
        let idx = id.index() as usize;
        if idx >= self.nodes.len() || self.node_gen[idx] != id.generation() {
            return Err(ArenaError::UnknownNode);
        }
        // Unlink from the region list if it is the head or in the chain.
        let region = self.nodes[idx].region;
        if region != RegionId::NONE {
            if let Ok((rslot, _gen)) = self.validate_region(region) {
                let head = self.regions[rslot as usize].first_node;
                if head == id {
                    self.regions[rslot as usize].first_node = self.nodes[idx].next_in_region;
                } else {
                    // Walk the chain to unlink (bounded by node count).
                    let mut cur = head;
                    let mut steps = 0usize;
                    while cur != NodeId::NONE && steps <= self.nodes.len() {
                        let nxt = match self.node(cur) {
                            Ok(n) => n.next_in_region,
                            Err(_) => NodeId::NONE,
                        };
                        if nxt == id {
                            let removed_next = self.nodes[idx].next_in_region;
                            if let Ok(nmut) = self.node_mut(cur) {
                                nmut.next_in_region = removed_next;
                            }
                            break;
                        }
                        cur = nxt;
                        steps += 1;
                    }
                }
            }
        }
        self.node_gen[idx] = self.node_gen[idx].wrapping_add(1);
        self.node_free.push(id.index());
        Ok(())
    }

    /// CEP:WHAT: Validates a region handle into (slot, generation).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownRegion on stale/out-of-range.
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 compares.
    /// CEP:EVIDENCE: region tests.
    fn validate_region(&self, id: RegionId) -> Result<(u32, u16), ArenaError> {
        if id.is_none() {
            return Err(ArenaError::UnknownRegion);
        }
        let idx = id.index() as usize;
        if idx >= self.regions.len() {
            return Err(ArenaError::UnknownRegion);
        }
        if self.next_region_gen[idx] != id.generation() {
            return Err(ArenaError::UnknownRegion);
        }
        Ok((id.index(), id.generation()))
    }

    /// CEP:WHAT: Region record lookup.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownRegion.
    /// CEP:ASSUMES: none
    /// CEP:COST: as validate_region.
    /// CEP:EVIDENCE: region tests.
    pub fn region(&self, id: RegionId) -> Result<&Region, ArenaError> {
        let (slot, _gen) = self.validate_region(id)?;
        Ok(&self.regions[slot as usize])
    }

    /// CEP:WHAT: Number of live node slots (diagnostic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn node_count(&self) -> usize {
        self.nodes.len() - self.node_free.len()
    }

    /// CEP:WHAT: Upper bound on ANY live node's slot index.
    /// CEP:WHY: Slots are indices into the backing array; deletions leave
    ///          holes, so live count UNDERCOUNTS the index space. Every
    ///          index-keyed table MUST size by slot_count, never by
    ///          node_count (audit F-3 follow-up: out-of-bounds panic when
    ///          passes run after canonicalization removed nodes).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: driver tier tests (folded programs with holes).
    pub fn slot_count(&self) -> usize {
        self.nodes.len()
    }

    /// CEP:WHAT: Number of live regions.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn region_count(&self) -> usize {
        self.regions.len() - self.region_free.len()
    }

    /// CEP:WHAT: Iterates live node ids in slot order (deterministic).
    /// CEP:WHY: Passes and the printer need stable iteration (CEP&CC 38.19:
    ///          no allocation-order dependence — slot order is structural).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(capacity) with generation filtering; caller bounds drains.
    /// CEP:EVIDENCE: printer determinism tests.
    pub fn for_each_live_node<F>(&self, mut f: F)
    where
        F: FnMut(NodeId, &Node),
    {
        for idx in 0..self.nodes.len() {
            // A slot is live iff it is not on the free list (freed slots keep
            // their bumped generation until reuse).
            let live = self.node_free.iter().all(|free| *free != idx as u32);
            if live {
                let id = NodeId::pack(idx as u32, self.node_gen[idx], self.nodes[idx].op.level());
                f(id, &self.nodes[idx]);
            }
        }
    }

    /// CEP:WHAT: Clones the arena with capacity at least equal to the source.
    /// CEP:WHY: The snapshot commit path (CEP-1) clones storage before
    ///          applying edits; preserving capacity guarantees no hidden
    ///          reallocation later (Law 1 — bounded storage stays bounded).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (allocation happens in the CEP-1 commit path only).
    /// CEP:ASSUMES: single-threaded call from the commit boundary.
    /// CEP:COST: O(nodes + regions) copy.
    /// CEP:EVIDENCE: snapshot commit tests.
    pub fn deep_clone(&self) -> IrArena {
        let cap_n = self.nodes.capacity().max(self.nodes.len());
        let cap_r = self.regions.capacity().max(self.regions.len());
        let mut out = IrArena::with_capacity(cap_n, cap_r);
        out.nodes = Vec::with_capacity(cap_n);
        out.nodes.extend_from_slice(&self.nodes);
        out.node_gen = Vec::with_capacity(cap_n);
        out.node_gen.extend_from_slice(&self.node_gen);
        out.node_free = Vec::with_capacity(self.node_free.len());
        out.node_free.extend_from_slice(&self.node_free);
        out.regions = Vec::with_capacity(cap_r);
        out.regions.extend_from_slice(&self.regions);
        out.region_free = Vec::with_capacity(self.region_free.len());
        out.region_free.extend_from_slice(&self.region_free);
        out.next_region_gen = Vec::with_capacity(cap_r);
        out.next_region_gen.extend_from_slice(&self.next_region_gen);
        out.root = self.root;
        out
    }

    /// CEP:WHAT: Produces the value handle for a node's output slot.
    /// CEP:WHY: Builders thread SSA values after insertion.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: OutputOverflow when slot >= MAX_OUTPUTS; UnknownNode.
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 validate + pack.
    /// CEP:EVIDENCE: builder tests.
    pub fn value_of(&self, node: NodeId, slot: u8) -> Result<ValueId, ArenaError> {
        if slot as usize >= MAX_OUTPUTS {
            return Err(ArenaError::OutputOverflow);
        }
        let n = self.node(node)?;
        if slot == 1 && n.ty1 == Type::None {
            // Only multi-output ops expose slot 1.
            return Err(ArenaError::OutputOverflow);
        }
        Ok(ValueId::from_node(node, slot))
    }
}

// Type import used by value_of's slot-1 check.
use crate::ty::Type;

/// CEP:WHAT: Convenience constructor for a constant node.
/// CEP:WHY: Frontends and tests build constants constantly; centralizing the
///          type inference (Scalar(F64/I64)) avoids per-site guesses.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: as insert_node
/// CEP:EVIDENCE: builder tests
pub fn const_i64(arena: &mut IrArena, region: RegionId, v: i64) -> Result<NodeId, ArenaError> {
    let node = Node::new(
        Op::ConstI64(v),
        region,
        &[],
        Type::Scalar(crate::ty::ScalarType::I64),
    );
    arena.insert_node(region, node)
}

/// CEP:WHAT: Convenience constructor for an f64 constant node.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: as insert_node
/// CEP:EVIDENCE: builder tests
pub fn const_f64(arena: &mut IrArena, region: RegionId, v: f64) -> Result<NodeId, ArenaError> {
    let node = Node::new(
        Op::ConstF64(v),
        region,
        &[],
        Type::Scalar(crate::ty::ScalarType::F64),
    );
    arena.insert_node(region, node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::BinaryOp;

    // CEP:WHAT: Insert then lookup round trip.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn insert_lookup_roundtrip() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let id = const_i64(&mut a, root, 42);
        assert!(id.is_ok());
        let id = match id {
            Ok(v) => v,
            Err(_) => return,
        };
        let n = a.node(id);
        assert!(n.is_ok());
        let n = match n {
            Ok(v) => v,
            Err(_) => return,
        };
        assert_eq!(n.op, Op::ConstI64(42));
        assert_eq!(n.region, root);
        let v = a.value_of(id, 0);
        assert!(v.is_ok());
        // Slot 1 does not exist for single-output nodes.
        assert_eq!(a.value_of(id, 1), Err(ArenaError::OutputOverflow));
    }

    // CEP:WHAT: Deletion bumps generation; stale handles fail.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if stale handles succeed.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn generation_reuse_detected() {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let id = const_f64(&mut a, root, 1.5);
        assert!(id.is_ok());
        let id = match id {
            Ok(v) => v,
            Err(_) => return,
        };
        assert!(a.remove_node(id).is_ok());
        assert_eq!(a.node(id), Err(ArenaError::UnknownNode));
        // Reuse the slot with a different op; the old handle stays invalid.
        let nid = const_f64(&mut a, root, 2.5);
        assert!(nid.is_ok());
        assert_eq!(a.node(id), Err(ArenaError::UnknownNode));
        if let Ok(n2) = nid {
            let got = a.node(n2);
            assert!(got.is_ok());
        }
    }

    // CEP:WHAT: Node capacity is enforced loudly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if hidden growth occurs.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn capacity_is_bounded() {
        let mut a = IrArena::with_capacity(4, 2);
        let root = a.root_region();
        for i in 0..4i64 {
            let r = const_i64(&mut a, root, i);
            assert!(r.is_ok(), "insert {} must fit", i);
        }
        assert_eq!(const_i64(&mut a, root, 99), Err(ArenaError::Exhausted));
    }

    // CEP:WHAT: Region tree creation and parent links.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on broken links.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn region_tree() {
        let mut a = IrArena::with_capacity(8, 8);
        let root = a.root_region();
        let then_r = a.new_region(root);
        assert!(then_r.is_ok());
        let then_r = match then_r {
            Ok(v) => v,
            Err(_) => return,
        };
        let got = a.region(then_r);
        assert!(got.is_ok());
        if let Ok(r) = got {
            assert_eq!(r.parent, root);
        }
        // Stale region handles fail.
        let bad = RegionId::pack(99, 0);
        assert_eq!(a.region(bad), Err(ArenaError::UnknownRegion));
    }

    // CEP:WHAT: Live-node iteration respects deletions.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if removed nodes leak into iteration.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn iteration_respects_deletion() {
        let mut a = IrArena::with_capacity(8, 2);
        let root = a.root_region();
        let id0 = const_i64(&mut a, root, 1);
        let id1 = const_i64(&mut a, root, 2);
        assert!(id0.is_ok() && id1.is_ok());
        if let (Ok(i0), Ok(i1)) = (id0, id1) {
            assert!(a.remove_node(i0).is_ok());
            let mut seen = 0;
            a.for_each_live_node(|id, _n| {
                assert_ne!(id, i0);
                seen += 1;
            });
            assert_eq!(seen, 1);
            // Region list must skip the removed head.
            let r = a.region(root);
            assert!(r.is_ok());
            if let Ok(reg) = r {
                assert_eq!(reg.first_node, i1);
            }
        }
    }

    // CEP:WHAT: Binary node inputs round trip through padding.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on input corruption.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn inputs_roundtrip() {
        let mut a = IrArena::with_capacity(8, 2);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        assert!(c0.is_ok() && c1.is_ok());
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let in0 = a.value_of(v0, 0);
            let in1 = a.value_of(v1, 0);
            assert!(in0.is_ok() && in1.is_ok());
            if let (Ok(i0), Ok(i1)) = (in0, in1) {
                let node = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[i0, i1],
                    Type::Scalar(crate::ty::ScalarType::I64),
                );
                let add = a.insert_node(root, node);
                assert!(add.is_ok());
                if let Ok(add_id) = add {
                    let got = a.node(add_id);
                    assert!(got.is_ok());
                    if let Ok(n) = got {
                        assert_eq!(n.n_inputs, 2);
                        assert_eq!(n.input(0), Some(i0));
                        assert_eq!(n.input(1), Some(i1));
                        assert_eq!(n.input(2), None);
                    }
                }
            }
        }
    }
}
