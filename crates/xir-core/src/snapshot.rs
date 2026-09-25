// CEP:FILE: crates/xir-core/src/snapshot.rs
// CEP:WHAT: IrSnapshot — immutable, hashable IR versions with transactional
//           edit-log commits.
// CEP:WHY: Master architecture "Transactional Rewrites": passes generate edit
//          logs; edits are validated and committed atomically to a new
//          Arc<IrSnapshot>, so concurrent passes read the old snapshot
//          without locking. Snapshot hashing follows CEP&CC 38.19 (no
//          allocation-order dependence): the fingerprint covers ops,
//          immediates, SSA edges and region structure in canonical slot order.
// CEP:CLASS: CEP-1 (commit path) / CEP-0 (hash/read path)
// CEP:STATUS: complete
// CEP:FAILURE: CommitError::{VerifierFailed, ArenaExhausted, UnknownTarget}
//              — commits never produce a half-applied snapshot (all-or-nothing
//              apply on a cloned arena).
// CEP:ASSUMES: Arc publication is the ONLY Arc use (CEP-1 boundary); read
//              paths on worker threads go through the immutable snapshot with
//              no reference counting traffic (Gear 4 reads pin epochs instead).
// CEP:COST: commit = O(nodes) arena clone + edit application + O(nodes) hash
//           (documented honestly; a future incremental-hash redesign is
//           CEP-11). Reads are O(1) handle lookups.
// CEP:EVIDENCE: tests `commit_produces_new_version`, `hash_is_deterministic`,
//           `edits_are_atomic`.
// CEP:SECURITY: snapshots are process-local; hashes are fingerprints, not
//           secrets (CEP&CC 22.9).
// CEP:HPC-DETERMINISM: deterministic — the hash covers structure, not
//           addresses; slot order is structural.
//! Immutable IR snapshots with transactional commits.

use std::sync::Arc;

use crate::arena::{ArenaError, IrArena};
use crate::id::{NodeId, ValueId};
use crate::op::Op;

/// Failure enumeration for commits.
///
/// CEP:WHAT: Explicit error type for the transactional commit boundary.
/// CEP:WHY: Law 6 — a failed verifier or exhausted arena must abort the
///          commit loudly, never publish a broken snapshot.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitError {
    /// The pre-commit verifier rejected the edit log.
    VerifierFailed,
    /// Applying edits exhausted the arena's bounded capacity.
    ArenaExhausted,
    /// An edit referenced an unknown node.
    UnknownTarget,
}

/// One transactional edit.
///
/// CEP:WHAT: The edit vocabulary of transactional rewrites.
/// CEP:WHY: Passes describe mutations declaratively; the commit applies them
///          in order on a clone so the published snapshot is all-or-nothing.
/// CEP:STATUS: complete
/// CEP:FAILURE: application errors surface as CommitError.
/// CEP:ASSUMES: edits refer to nodes live in the base snapshot.
/// CEP:COST: O(1) each; application is O(edits).
/// CEP:EVIDENCE: test `edits_are_atomic`.
#[derive(Debug, Clone, Copy)]
pub enum Edit {
    /// Rewrites every input use of `from` to `to`.
    ReplaceAllUses {
        /// Old value.
        from: ValueId,
        /// New value.
        to: ValueId,
    },
    /// Removes a node (DCE and rewrites).
    RemoveNode {
        /// Target node.
        node: NodeId,
    },
    /// Replaces a node's op (e.g. e-graph extraction swaps).
    SetOp {
        /// Target node.
        node: NodeId,
        /// New opcode.
        op: Op,
    },
    /// Sets a node's effect-token input (token threading repairs).
    SetEffectIn {
        /// Target node.
        node: NodeId,
        /// New token input.
        effect: ValueId,
    },
}

/// An edit log collected by a pass.
///
/// CEP:WHAT: Ordered edit list plus a monotonically increasing pass id.
/// CEP:WHY: Pass provenance is part of the pipeline manifest (CEP&CC 38.20);
///          the id feeds telemetry, not translation semantics.
/// CEP:STATUS: complete
/// CEP:FAILURE: appends cannot fail (Vec growth in the CEP-1 pass path).
/// CEP:ASSUMES: collected single-threaded per pass.
/// CEP:COST: O(1) amortized append.
/// CEP:EVIDENCE: tests in this module.
#[derive(Debug, Clone)]
pub struct EditLog {
    /// Pass identifier for telemetry (not translation semantics).
    pub pass_id: u16,
    edits: Vec<Edit>,
}

impl EditLog {
    /// CEP:WHAT: Creates an empty log for a pass.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: one allocation (pass path)
    /// CEP:EVIDENCE: tests in this module
    pub fn new(pass_id: u16) -> EditLog {
        EditLog {
            pass_id,
            edits: Vec::new(),
        }
    }

    /// CEP:WHAT: Appends one edit.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: amortized O(1)
    /// CEP:EVIDENCE: tests in this module
    pub fn push(&mut self, e: Edit) {
        self.edits.push(e);
    }

    /// CEP:WHAT: Number of edits (diagnostic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn len(&self) -> usize {
        self.edits.len()
    }

    /// CEP:WHAT: Emptiness probe.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    /// CEP:WHAT: Borrow the edit list (commit + telemetry).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero
    /// CEP:EVIDENCE: commit path
    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    /// CEP:WHAT: Merges another log (Disjoint-concurrency passes).
    /// CEP:WHY: Gear-1 workers edit disjoint subgraphs; the manager merges
    ///          their logs before one atomic commit.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: logs are disjoint (manager's contract; verifier catches
    ///              violations at commit).
    /// CEP:COST: O(other.len())
    /// CEP:EVIDENCE: pass-manager tests.
    pub fn merge(&mut self, other: &EditLog) {
        self.edits.extend_from_slice(&other.edits);
    }
}

/// An immutable IR version.
///
/// CEP:WHAT: Arena + version + structural fingerprint behind an Arc.
/// CEP:WHY: The transactional publication unit (arch section 2): readers get
///          a consistent view; writers commit new versions.
/// CEP:STATUS: complete
/// CEP:FAILURE: lookups delegate to the arena (ArenaError).
/// CEP:ASSUMES: frozen after construction (private arena, no mutation API).
/// CEP:COST: O(nodes) memory; O(1) reads.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: hash is structural (see module header).
pub struct IrSnapshot {
    arena: IrArena,
    version: u64,
    fingerprint: u64,
}

impl IrSnapshot {
    /// CEP:WHAT: Builds version 0 from an arena and computes the fingerprint.
    /// CEP:WHY: Frontend entry: the parsed IR becomes the root snapshot.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (arena is moved in, frozen by ownership).
    /// CEP:ASSUMES: arena passed a verifier run (caller contract).
    /// CEP:COST: O(nodes) hash.
    /// CEP:EVIDENCE: test `hash_is_deterministic`.
    pub fn new(arena: IrArena) -> IrSnapshot {
        let fingerprint = compute_fingerprint(&arena);
        IrSnapshot {
            arena,
            version: 0,
            fingerprint,
        }
    }

    /// CEP:WHAT: Immutable arena view.
    /// CEP:WHY: Passes and the printer read without mutation handles.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: zero
    /// CEP:EVIDENCE: whole stack
    pub fn arena(&self) -> &IrArena {
        &self.arena
    }

    /// CEP:WHAT: Version number (0-based, increments per commit).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn version(&self) -> u64 {
        self.version
    }

    /// CEP:WHAT: Structural fingerprint.
    /// CEP:WHY: JIT cache keys and pipeline manifests (CEP&CC 38.19).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// CEP:WHAT: Applies an edit log atomically, producing the next version.
    /// CEP:WHY: Transactional rewrite boundary: clone → apply → verify →
    ///          publish. A failed apply discards the clone; the source
    ///          snapshot is untouched (all-or-nothing).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: VerifierFailed / ArenaExhausted / UnknownTarget.
    /// CEP:ASSUMES: `verify` runs BEFORE the snapshot is constructed from the
    ///              clone; the caller supplies the verifier closure
    ///              (xir-graph owns the actual verifier).
    /// CEP:COST: O(nodes) clone + O(edits) apply + O(nodes) rehash.
    /// CEP:EVIDENCE: tests `commit_produces_new_version`, `edits_are_atomic`.
    pub fn commit(
        &self,
        log: &EditLog,
        verify: &dyn Fn(&IrArena) -> bool,
    ) -> Result<Arc<IrSnapshot>, CommitError> {
        let mut clone = self.arena.deep_clone();
        for e in log.edits() {
            apply_edit(&mut clone, e)?;
        }
        if !verify(&clone) {
            return Err(CommitError::VerifierFailed);
        }
        let fingerprint = compute_fingerprint(&clone);
        Ok(Arc::new(IrSnapshot {
            arena: clone,
            version: self.version.wrapping_add(1),
            fingerprint,
        }))
    }

    /// CEP:WHAT: Direct mutation escape hatch for builder phases.
    /// CEP:WHY: Frontends build the arena BEFORE the first snapshot; this
    ///          accessor is intentionally absent for published snapshots —
    ///          use `commit` instead (transactional contract).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: only used pre-publication by the owning builder.
    /// CEP:COST: zero
    /// CEP:EVIDENCE: builder tests
    pub fn into_arena(self) -> IrArena {
        self.arena
    }
}

/// CEP:WHAT: Applies one edit to a mutable arena.
/// CEP:STATUS: complete
/// CEP:FAILURE: UnknownTarget for stale handles; ArenaExhausted never
///              (RemoveNode/SetOp/SetEffectIn do not allocate).
/// CEP:ASSUMES: edit targets exist in the clone.
/// CEP:COST: ReplaceAllUses is O(nodes) (scan); others O(1).
/// CEP:EVIDENCE: test `edits_are_atomic`.
fn apply_edit(arena: &mut IrArena, e: &Edit) -> Result<(), CommitError> {
    match e {
        Edit::ReplaceAllUses { from, to } => {
            let mut ids: Vec<NodeId> = Vec::new();
            arena.for_each_live_node(|id, _| ids.push(id));
            for id in ids {
                if let Ok(node) = arena.node_mut(id) {
                    for i in 0..crate::node::MAX_INPUTS {
                        if node.inputs[i] == *from {
                            node.inputs[i] = *to;
                        }
                    }
                    if node.effect_in == *from {
                        node.effect_in = *to;
                    }
                }
            }
            Ok(())
        }
        Edit::RemoveNode { node } => match arena.remove_node(*node) {
            Ok(()) => Ok(()),
            Err(ArenaError::UnknownNode) => Err(CommitError::UnknownTarget),
            Err(_) => Err(CommitError::ArenaExhausted),
        },
        Edit::SetOp { node, op } => match arena.node_mut(*node) {
            Ok(n) => {
                n.op = *op;
                Ok(())
            }
            Err(_) => Err(CommitError::UnknownTarget),
        },
        Edit::SetEffectIn { node, effect } => match arena.node_mut(*node) {
            Ok(n) => {
                n.effect_in = *effect;
                Ok(())
            }
            Err(_) => Err(CommitError::UnknownTarget),
        },
    }
}

// Arena cloning is IrArena::deep_clone (arena.rs): capacity-preserving copy.

/// CEP:WHAT: Computes the structural fingerprint in canonical slot order.
/// CEP:WHY: CEP&CC 38.19: hashing must not depend on allocation order or map
///          iteration; we walk slots in index order and hash node structure
///          (op + immediates + inputs + region + types).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(nodes)
/// CEP:EVIDENCE: test `hash_is_deterministic`.
fn compute_fingerprint(arena: &IrArena) -> u64 {
    let mut h = crate::hash::Fnv64::new();
    arena.for_each_live_node(|_id, node| {
        node.hash_into(&mut h);
    });
    // Region count participates (control structure changes are semantic).
    h.write_u64(arena.region_count() as u64);
    h.finish()
}

// Unit-type helper re-exported for doc clarity in dependents.
pub type SnapshotRef = Arc<IrSnapshot>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::const_i64;
    use crate::node::Node;
    use crate::op::BinaryOp;

    fn build() -> IrSnapshot {
        let mut a = IrArena::with_capacity(64, 8);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let i0 = a.value_of(v0, 0).ok();
            let i1 = a.value_of(v1, 0).ok();
            if let (Some(a0), Some(a1)) = (i0, i1) {
                let node = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    crate::ty::Type::Scalar(crate::ty::ScalarType::I64),
                );
                let _ = a.insert_node(root, node);
            }
        }
        IrSnapshot::new(a)
    }

    fn always_ok(_a: &IrArena) -> bool {
        true
    }

    // CEP:WHAT: Commit bumps the version and keeps the old snapshot intact.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on version or isolation breakage.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn commit_produces_new_version() {
        let s = build();
        let v0 = s.version();
        let log = EditLog::new(1);
        let s2 = s.commit(&log, &always_ok);
        assert!(s2.is_ok());
        if let Ok(s2) = s2 {
            assert_eq!(s2.version(), v0 + 1);
            assert_eq!(s.version(), v0);
        }
    }

    // CEP:WHAT: Fingerprints are structural and repeatable.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on nondeterminism.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn hash_is_deterministic() {
        let a = build();
        let b = build();
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    // CEP:WHAT: Edits apply all-or-nothing; verifier failure aborts.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if partial state leaks.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn edits_are_atomic() {
        let s = build();
        let mut log = EditLog::new(2);
        // Remove a node that does not exist -> whole commit fails.
        log.push(Edit::RemoveNode {
            node: NodeId::pack(999, 9, crate::id::IrLevel::Graph),
        });
        let failed = s.commit(&log, &always_ok);
        assert_eq!(failed.err(), Some(CommitError::UnknownTarget));

        // Verifier rejection also aborts.
        let mut log2 = EditLog::new(3);
        log2.push(Edit::SetOp {
            node: NodeId::pack(0, 0, crate::id::IrLevel::Graph),
            op: Op::Dot,
        });
        let rejected = s.commit(&log2, &|_a| false);
        assert_eq!(rejected.err(), Some(CommitError::VerifierFailed));
        // Source unchanged.
        assert_eq!(s.version(), 0);
    }
}
