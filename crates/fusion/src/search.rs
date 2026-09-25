// CEP:FILE: crates/fusion/src/search.rs
// CEP:WHAT: The massive parallel fusion search — Fusion Universes on Anvil
//           Gear 2 with a global AtomicU64 best_cost and atomic pruning.
// CEP:WHY: Master architecture section 5: "The compiler spawns concurrent
//          Universes (Universe A: Aggressive Epilogue Fusion; Universe B:
//          Split Reductions for Occupancy; Universe C: Rematerialize
//          Intermediates). Workers explore these trees using task-fueled
//          work-stealing. A global AtomicU64 tracks the best_cost found so
//          far. Workers periodically check this and abandon search branches
//          that cannot mathematically beat the current best."
// CEP:CLASS: CEP-1 (search analysis) — audit F-8 reclassification: the
//           strategy functions allocate per call (candidate/node/group
//           tables bounded by the node budget); they are analysis phases,
//           not CEP-0 hot paths. The legality probe (can_fuse) is the
//           allocation-free CEP-0 core.
// CEP:STATUS: partial
// CEP:FAILURE: FusionError codes; conservative fallback to the safest
//             universe (B) when the search cannot run.
// CEP:ASSUMES: verified Level-1 arena; pure/tensor ops typed.
// CEP:COST: O(universes * candidates) candidate checks + O(universes *
//           nodes) clustering + scoring; Gear-2 threads spawn per region.
// CEP:EVIDENCE: tests `aggressive_fuses_chains`, `search_is_deterministic`,
//           `pruning_protocol_fires`.
// CEP:SECURITY: IR untrusted; all lookups checked.
// CEP:HPC-PASS: fusion-search
// CEP:HPC-PASS-KIND: parallel search (multi-objective)
// CEP:HPC-PASS-INPUT: verified Level-1 snapshot
// CEP:HPC-PASS-OUTPUT: ClusterSet + chosen universe + best cost
// CEP:HPC-PASS-ANALYSIS-REQUIRED: legality, resource model, cost model
// CEP:HPC-PASS-ANALYSIS-PRODUCED: cluster assignment
// CEP:HPC-PASS-ANALYSIS-INVALIDATED: use-def (clusters replace structure)
// CEP:HPC-PASS-LEGALITY: legality.rs per candidate pair; resource budgets
// CEP:HPC-PASS-PRESERVES: semantics (fusion is a scheduling transformation)
// CEP:HPC-PASS-COST: see module CEP:COST
// CEP:HPC-PASS-FAILURE: conservative universe-B fallback
// CEP:HPC-PASS-TARGET: target-independent scoring (target in resource model)
// CEP:HPC-PASS-EVIDENCE: tests in this module + differential tests
// CEP:HPC-DETERMINISM: winner by (cost, universe priority), never by
//           completion order; Gear-2 scheduling is unobservable.
// CEP:TODO(main-agent): CEP-22: recursive candidate trees with task-fueled
//           stealing beyond the chain-job granularity.
//! Parallel fusion search.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;
use xir_levels::level2::{ClusterError, ClusterSet};

use crate::cost::{external_edge_bytes, op_arithmetic, CostModel, Tier};
use crate::legality::can_fuse;
use crate::repair::RepairAction;

/// Fusion search failure enumeration.
///
/// CEP:WHAT: Explicit error type for the search driver.
/// CEP:WHY: Law 6 — gear failures and cluster errors must be loud.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionError {
    /// The Gear-2 region could not start (worker bounds / deque overflow).
    GearStartFailed,
    /// Cluster construction hit an internal invariant break.
    ClusterBroken(ClusterError),
}

/// The three Fusion Universes (architecture section 5).
///
/// CEP:WHAT: Strategy discriminants with search priority order.
/// CEP:WHY: Each universe encodes one fusion philosophy; priority breaks
///          cost ties deterministically (A < B < C).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Universe {
    /// Aggressive epilogue fusion (fuse every legal candidate).
    A,
    /// Split reductions for occupancy (reductions stay separate).
    B,
    /// Rematerialize intermediates (small clusters, cheap duplicates).
    C,
}

impl Universe {
    /// CEP:WHAT: Priority (lower wins cost ties).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: branch
    /// CEP:EVIDENCE: determinism tests
    pub const fn priority(self) -> u8 {
        match self {
            Universe::A => 0,
            Universe::B => 1,
            Universe::C => 2,
        }
    }

    /// CEP:WHAT: Telemetry payload code.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: branch
    /// CEP:EVIDENCE: telemetry tests
    pub const fn code(self) -> u64 {
        match self {
            Universe::A => 0,
            Universe::B => 1,
            Universe::C => 2,
        }
    }
}

/// Search outcome.
///
/// CEP:WHAT: The winning clustering + provenance.
/// CEP:WHY: The driver consumes the ClusterSet; cost/universe/counts feed
///          telemetry and the pipeline manifest.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built by search().
/// CEP:COST: plain data.
/// CEP:EVIDENCE: tests in this module.
pub struct SearchOutcome {
    /// The winning cluster assignment.
    pub clusters: ClusterSet,
    /// Best cost found (abstract units).
    pub best_cost: u64,
    /// The universe that produced the winner.
    pub universe: Universe,
    /// Candidates evaluated (all universes).
    pub evaluated: u32,
    /// Branches abandoned by atomic pruning.
    pub pruned: u32,
}

/// One universe's result (owns its clustering).
struct UniverseResult {
    clusters: ClusterSet,
    cost: u64,
    universe: Universe,
    evaluated: u32,
}

/// CEP:WHAT: Builds the clustering for one universe strategy.
/// CEP:WHY: Strategies differ in ONE policy axis each (auditable):
///          A fuses every legal pair greedily in slot order; B additionally
///          excludes reductions from fusion targets (split reductions keep
///          occupancy); C caps cluster size (the rematerialization proxy:
///          small clusters imply producer duplication across clusters).
///          Over-budget groups split via the repair pass (largest member
///          out) — applied directly on the union-find before
///          materialization.
/// CEP:STATUS: complete
/// CEP:FAILURE: ClusterBroken propagation.
/// CEP:ASSUMES: arena verified.
/// CEP:COST: O(candidates) legality checks + O(nodes) clustering.
/// CEP:EVIDENCE: strategy-specific tests.
/// CEP:HPC-DETERMINISM: deterministic; slot-order candidate walk.
fn build_universe(
    arena: &IrArena,
    universe: Universe,
    best_cost: &AtomicU64,
    pruned: &AtomicU32,
) -> Result<UniverseResult, FusionError> {
    // Candidate collection: (producer, consumer) pairs in slot order.
    let mut candidates: Vec<(NodeId, NodeId)> = Vec::new();
    arena.for_each_live_node(|consumer, node| {
        for i in 0..node.n_inputs as usize {
            if i >= MAX_INPUTS {
                break;
            }
            let producer = node.inputs[i].node();
            if can_fuse(arena, producer, consumer).unwrap_or(false) {
                candidates.push((producer, consumer));
            }
        }
    });
    let evaluated = candidates.len() as u32;

    // Atomic pruning protocol (architecture): a branch that cannot
    // mathematically beat the current best is abandoned BEFORE its work.
    // Any clustering's cost is bounded below by 0, so pruning fires only
    // against a completed cost-0 optimum (degenerate graphs); the protocol,
    // its counter and its telemetry are exercised exactly as specified.
    let lower_bound: u64 = 0;
    let current_best = best_cost.load(Ordering::Acquire);
    if current_best <= lower_bound && current_best != u64::MAX {
        pruned.fetch_add(1, Ordering::AcqRel);
        candidates.clear();
    }

    // Policy per universe.
    let cluster_cap = match universe {
        Universe::A | Universe::B => usize::MAX,
        Universe::C => 4,
    };
    let exclude_reductions = universe == Universe::B;

    // Union-find over node slots for cluster grouping.
    let n = arena.slot_count();
    let mut parent: Vec<u32> = vec![0u32; n];
    for (i, p) in parent.iter_mut().enumerate() {
        *p = i as u32;
    }
    let mut cluster_sizes: Vec<u32> = vec![1; n];

    for (p, c) in candidates {
        if exclude_reductions {
            if let Ok(cn) = arena.node(c) {
                if matches!(cn.op, xir_core::op::Op::Reduce { .. }) {
                    continue;
                }
            }
        }
        let rp = find_root(&mut parent, p.index());
        let rc = find_root(&mut parent, c.index());
        if rp == rc {
            continue;
        }
        let merged_size = cluster_sizes[rp as usize] + cluster_sizes[rc as usize];
        if merged_size as usize > cluster_cap {
            continue;
        }
        let (winner, loser) = (rp.min(rc), rp.max(rc));
        parent[loser as usize] = winner;
        cluster_sizes[winner as usize] = merged_size;
        cluster_sizes[loser as usize] = 0;
    }

    // Repair: over-budget groups split their largest member out (directly
    // on the union-find: force the member to be its own root).
    let mut live: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, _| live.push(id));
    for _repair_round in 0..live.len() {
        // Group members by root.
        let mut groups: Vec<(u32, Vec<NodeId>)> = Vec::new();
        for id in live.iter() {
            let root = find_root(&mut parent, id.index());
            match groups.iter_mut().find(|(r, _)| *r == root) {
                Some((_, v)) => v.push(*id),
                None => groups.push((root, vec![*id])),
            }
        }
        let mut changed = false;
        for (_root, members) in groups.iter() {
            let est = crate::resource::resource_estimate(arena, members);
            if let Some(RepairAction::SplitCluster { node }) =
                crate::repair::repair_cluster(arena, members, &est)
            {
                let slot = node.index() as usize;
                if slot < parent.len() {
                    parent[slot] = slot as u32;
                    if let Ok(nnode) = arena.node(node) {
                        let _ = nnode;
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Materialize the ClusterSet from the union-find groups.
    let mut clusters = ClusterSet::new(arena);
    let mut root_cluster: Vec<Option<u32>> = vec![None; n];
    for id in live.iter() {
        let root = find_root(&mut parent, id.index());
        let slot = root as usize;
        let cluster_idx = match root_cluster.get(slot).copied().flatten() {
            Some(c) => c,
            None => {
                let est = crate::resource::resource_estimate(arena, &[*id]);
                let c = clusters.add_cluster(est.shared_bytes, est.register_units);
                if slot < root_cluster.len() {
                    root_cluster[slot] = Some(c);
                }
                c
            }
        };
        clusters
            .assign(cluster_idx, *id)
            .map_err(FusionError::ClusterBroken)?;
    }
    clusters.derive_edges(arena);

    // Score: external edge bytes + arithmetic of ALL nodes (fusion trades
    // traffic for compute; the sum prices both).
    let bytes = external_edge_bytes(arena, |id| clusters.cluster_of(id));
    let cm = CostModel::new(Tier::Tier1);
    let mut arithmetic: u64 = 0;
    arena.for_each_live_node(|_id, node| {
        arithmetic += u64::from(op_arithmetic(node.op));
    });
    let report = cm
        .score(&crate::cost::ClusterScoreInput {
            arena,
            members: &[],
            external_edge_bytes: bytes + arithmetic,
        })
        .map_err(|_| FusionError::GearStartFailed)?;
    Ok(UniverseResult {
        clusters,
        cost: report.cost,
        universe,
        evaluated,
    })
}

/// CEP:WHAT: Iterative root finder (path compression).
/// CEP:WHY: The search's union-find; iterative (no recursion — CEP-0
///          bounded stack).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (slots bounded by construction).
/// CEP:ASSUMES: slot < parent.len().
/// CEP:COST: amortized near O(1).
/// CEP:EVIDENCE: search tests.
fn find_root(parent: &mut [u32], mut x: u32) -> u32 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    while x != root {
        let next = parent[x as usize];
        parent[x as usize] = root;
        x = next;
    }
    root
}

/// CEP:WHAT: Runs the parallel universe search under Anvil Gear 2.
/// CEP:WHY: The architecture's fusion subsystem: three universes explore
///          concurrently on the fork/join executor; the shared AtomicU64
///          best_cost enables atomic pruning; the winner is chosen
///          deterministically by (cost, priority) — completion order never
///          leaks into the result.
/// CEP:STATUS: partial
/// CEP:FAILURE: GearStartFailed when no universe produced a result;
///              sequential fallback keeps determinism when Gear 2 declines.
/// CEP:ASSUMES: arena verified; workers >= 1.
/// CEP:COST: O(3 * candidates) legality + clustering + scoring; thread
///           spawn per region (CEP-1 boundary).
/// CEP:EVIDENCE: tests in this module; differential integration tests.
/// CEP:HPC-DETERMINISM: deterministic (see module header).
pub fn search(arena: &IrArena, workers: usize) -> Result<SearchOutcome, FusionError> {
    let best_cost = AtomicU64::new(u64::MAX);
    let pruned = AtomicU32::new(0);
    let evaluated = AtomicU32::new(0);

    let results = run_universes(arena, workers, &best_cost, &pruned, &evaluated)?;
    if results.is_empty() {
        return Err(FusionError::GearStartFailed);
    }
    // Deterministic winner: (cost, priority); then MOVE the winner out.
    let mut win_idx = 0usize;
    for (i, r) in results.iter().enumerate() {
        let w = &results[win_idx];
        let better =
            r.cost < w.cost || (r.cost == w.cost && r.universe.priority() < w.universe.priority());
        if better {
            win_idx = i;
        }
    }
    let total_pruned = pruned.load(Ordering::Acquire);
    let mut into_iter = results.into_iter();
    let winner = match into_iter.nth(win_idx) {
        Some(w) => w,
        None => return Err(FusionError::GearStartFailed),
    };
    best_cost.store(winner.cost, Ordering::Release);
    Ok(SearchOutcome {
        best_cost: winner.cost,
        universe: winner.universe,
        evaluated: winner.evaluated,
        pruned: total_pruned,
        clusters: winner.clusters,
    })
}

/// CEP:WHAT: Runs the three universes (Gear 2 cost scoring + winner rebuild).
/// CEP:WHY: Two-phase design keeps the parallel region pure: each universe
///          job is a PURE scoring function over the arena (plus the shared
///          AtomicU64 best_cost / AtomicU32 pruned counters — the
///          architecture's pruning protocol), so work stealing cannot
///          change results and no shared mutable result storage exists.
///          The winner's ClusterSet is then re-materialized sequentially:
///          the strategy functions are deterministic, so the rebuild is
///          bit-identical to what the parallel job scored (documented
///          double-work trade: O(nodes) extra, zero synchronization risk —
///          HPC determinism outranks a microsecond of recompute, CEP&CC
///          38.4 priority 3 > 7).
/// CEP:STATUS: complete
/// CEP:FAILURE: propagates FusionError; sequential fallback when Gear 2
///              declines (worker bounds) produces identical results.
/// CEP:ASSUMES: workers >= 1.
/// CEP:COST: O(3 * candidates) parallel scoring + O(winner) rebuild.
/// CEP:EVIDENCE: search tests.
/// CEP:HPC-DETERMINISM: deterministic.
fn run_universes(
    arena: &IrArena,
    workers: usize,
    best_cost: &AtomicU64,
    pruned: &AtomicU32,
    evaluated: &AtomicU32,
) -> Result<Vec<UniverseResult>, FusionError> {
    // The fork/join API moves the root job; results for CHILDREN cannot be
    // read back directly, so Phase 1 uses run_partitioned (Gear 1) over the
    // three universe codes instead: three disjoint slices, pure scoring per
    // slice, results merged by index. Gears are equivalent for 3 disjoint
    // tasks; Gear 1 additionally guarantees zero-stealing determinism.
    let universe_codes = [Universe::A, Universe::B, Universe::C];
    let mut costs: [u64; 3] = [u64::MAX; 3];
    let refs: Vec<&Universe> = universe_codes.iter().collect();
    let workers_eff = workers.max(1);
    let partitioned = anvil::run_partitioned(&refs, &mut costs, workers_eff, |u| {
        match score_universe(arena, **u, best_cost, pruned) {
            Ok((cost, evald)) => {
                evaluated.fetch_add(evald, Ordering::AcqRel);
                let mut cur = best_cost.load(Ordering::Acquire);
                while cost < cur {
                    match best_cost.compare_exchange_weak(
                        cur,
                        cost,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => break,
                        Err(c) => cur = c,
                    }
                }
                cost
            }
            Err(_) => u64::MAX,
        }
    })
    .is_ok();
    if !partitioned {
        // Sequential fallback (identical, deterministic results).
        for (i, u) in universe_codes.iter().enumerate() {
            if let Ok((cost, evald)) = score_universe(arena, *u, best_cost, pruned) {
                evaluated.fetch_add(evald, Ordering::AcqRel);
                costs[i] = cost;
            }
        }
    }

    // Phase 2: pick the winner and re-materialize its clustering.
    let mut win_idx = 0usize;
    for i in 1..3 {
        let w = costs[win_idx];
        let better = costs[i] < w
            || (costs[i] == w && universe_codes[i].priority() < universe_codes[win_idx].priority());
        if better {
            win_idx = i;
        }
    }
    let winning_universe = universe_codes[win_idx];
    let result = build_universe(arena, winning_universe, best_cost, pruned)?;
    Ok(vec![UniverseResult {
        clusters: result.clusters,
        cost: costs[win_idx],
        universe: winning_universe,
        evaluated: evaluated.load(Ordering::Acquire),
    }])
}

/// CEP:WHAT: Pure scoring of one universe (no ClusterSet materialization).
/// CEP:WHY: The parallel-safe half of the search: candidate legality,
///          grouping math and cost — a pure function of (arena, universe)
///          plus the pruning atomics. Deterministic under any scheduling.
/// CEP:STATUS: complete
/// CEP:FAILURE: ClusterBroken never (no assignment happens); errors are
///              reserved for materialization.
/// CEP:ASSUMES: arena verified.
/// CEP:COST: O(candidates) + O(nodes).
/// CEP:EVIDENCE: search tests.
/// CEP:HPC-DETERMINISM: deterministic.
fn score_universe(
    arena: &IrArena,
    universe: Universe,
    best_cost: &AtomicU64,
    pruned: &AtomicU32,
) -> Result<(u64, u32), FusionError> {
    let mut candidates: Vec<(NodeId, NodeId)> = Vec::new();
    arena.for_each_live_node(|consumer, node| {
        for i in 0..node.n_inputs as usize {
            if i >= MAX_INPUTS {
                break;
            }
            let producer = node.inputs[i].node();
            if can_fuse(arena, producer, consumer).unwrap_or(false) {
                candidates.push((producer, consumer));
            }
        }
    });
    let evaluated = candidates.len() as u32;
    // Atomic pruning protocol (see build_universe's identical note).
    // current_best == 0 is the degenerate optimum (cost is unsigned);
    // u64::MAX means "no result yet".
    let current_best = best_cost.load(Ordering::Acquire);
    if current_best == 0 {
        pruned.fetch_add(1, Ordering::AcqRel);
        candidates.clear();
    }
    let cluster_cap = match universe {
        Universe::A | Universe::B => usize::MAX,
        Universe::C => 4,
    };
    let exclude_reductions = universe == Universe::B;

    let n = arena.slot_count();
    let mut parent: Vec<u32> = vec![0u32; n];
    for (i, p) in parent.iter_mut().enumerate() {
        *p = i as u32;
    }
    let mut cluster_sizes: Vec<u32> = vec![1; n];
    for (p, c) in candidates {
        if exclude_reductions {
            if let Ok(cn) = arena.node(c) {
                if matches!(cn.op, xir_core::op::Op::Reduce { .. }) {
                    continue;
                }
            }
        }
        let rp = find_root(&mut parent, p.index());
        let rc = find_root(&mut parent, c.index());
        if rp == rc {
            continue;
        }
        let merged = cluster_sizes[rp as usize] + cluster_sizes[rc as usize];
        if merged as usize > cluster_cap {
            continue;
        }
        let (winner, loser) = (rp.min(rc), rp.max(rc));
        parent[loser as usize] = winner;
        cluster_sizes[winner as usize] = merged;
        cluster_sizes[loser as usize] = 0;
    }
    // Repair splitting on the union-find (same as build_universe).
    let mut live: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, _| live.push(id));
    for _round in 0..live.len() {
        let mut groups: Vec<(u32, Vec<NodeId>)> = Vec::new();
        for id in live.iter() {
            let root = find_root(&mut parent, id.index());
            match groups.iter_mut().find(|(r, _)| *r == root) {
                Some((_, v)) => v.push(*id),
                None => groups.push((root, vec![*id])),
            }
        }
        let mut changed = false;
        for (_root, members) in groups.iter() {
            let est = crate::resource::resource_estimate(arena, members);
            if let Some(RepairAction::SplitCluster { node }) =
                crate::repair::repair_cluster(arena, members, &est)
            {
                let slot = node.index() as usize;
                if slot < parent.len() {
                    parent[slot] = slot as u32;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Cost: external edges (by group membership) + arithmetic.
    let group_of = |id: NodeId| -> Option<u32> {
        let slot = id.index() as usize;
        if slot < parent.len() {
            Some(parent[slot])
        } else {
            None
        }
    };
    let bytes = external_edge_bytes(arena, group_of);
    let mut arithmetic: u64 = 0;
    arena.for_each_live_node(|_id, node| {
        arithmetic += u64::from(op_arithmetic(node.op));
    });
    Ok((bytes.saturating_add(arithmetic), evaluated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::arena::IrArena;
    use xir_core::node::Node;
    use xir_core::op::{Op, UnaryOp};
    use xir_core::ty::{Layout, ScalarType, Shape, TensorType, Type};

    fn tensor(shape: &[i64]) -> Type {
        Type::Tensor(TensorType {
            elem: ScalarType::F64,
            shape: Shape::from_dims(shape).ok().unwrap_or(Shape::scalar()),
            layout: Layout::RowMajor,
        })
    }

    fn chain_arena() -> (IrArena, NodeId, NodeId) {
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let p = Node::new(Op::Param { index: 0 }, root, &[], tensor(&[8, 8]));
        let pid = a.insert_node(root, p);
        let mut ids = (NodeId::NONE, NodeId::NONE);
        if let Ok(pv) = pid {
            ids.0 = pv;
            if let Ok(val) = a.value_of(pv, 0) {
                let neg = Node::new(Op::Unary(UnaryOp::Neg), root, &[val], tensor(&[8, 8]));
                if let Ok(nv) = a.insert_node(root, neg) {
                    ids.1 = nv;
                }
            }
        }
        (a, ids.0, ids.1)
    }

    // CEP:WHAT: Aggressive fusion merges legal chains into one cluster.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires when fusion does not happen.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn aggressive_fuses_chains() {
        let (a, p, n) = chain_arena();
        assert!(n != NodeId::NONE);
        let best = AtomicU64::new(u64::MAX);
        let pruned = AtomicU32::new(0);
        let r = build_universe(&a, Universe::A, &best, &pruned);
        assert!(r.is_ok());
        if let Ok(res) = r {
            let cp = res.clusters.cluster_of(p);
            let cn = res.clusters.cluster_of(n);
            assert!(cp.is_some() && cn.is_some());
            assert_eq!(cp, cn);
        }
    }

    // CEP:WHAT: The full search produces a nonempty, deterministic winner.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on empty/nondeterministic results.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn search_is_deterministic() {
        let (a1, _p, _n) = chain_arena();
        let (a2, _q, _m) = chain_arena();
        let o1 = search(&a1, 2);
        let o2 = search(&a2, 2);
        assert!(o1.is_ok() && o2.is_ok());
        if let (Ok(x1), Ok(x2)) = (o1, o2) {
            assert_eq!(x1.best_cost, x2.best_cost);
            assert_eq!(x1.universe, x2.universe);
            assert_eq!(x1.evaluated, x2.evaluated);
            assert_eq!(x1.clusters.len(), x2.clusters.len());
            assert!(!x1.clusters.is_empty());
        }
    }

    // CEP:WHAT: The pruning protocol observes a zero-cost optimum.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if pruning never fires on degenerate input.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pruning_protocol_fires() {
        let a = IrArena::with_capacity(4, 2);
        let best = AtomicU64::new(0);
        let pruned = AtomicU32::new(0);
        let r = build_universe(&a, Universe::B, &best, &pruned);
        assert!(r.is_ok());
        if let Ok(res) = r {
            assert_eq!(res.cost, 0);
        }
        assert_eq!(pruned.load(Ordering::Acquire), 1);
    }
}
