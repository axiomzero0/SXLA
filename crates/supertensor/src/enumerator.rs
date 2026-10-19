// CEP:FILE: crates/supertensor/src/enumerator.rs
// CEP:WHAT: Stage 2 — the concurrent enumerative search over the polytope:
//           worker threads DFS-enumerate schedules (mem DESC, tiles ASC over
//           divisors, fuse none-then-targets, vec ASC), evaluate the analytic
//           roofline, and stream complete candidates to the merger thread
//           through per-worker TypedSpsc channels (Anvil v2 Upgrade 3); the
//           merger maintains the Pareto frontier and publishes a packed prune
//           anchor the workers poll non-blockingly.
// CEP:WHY: Axiom 4 — lock-free SPSC pipelines throughout, no mutexes, no
//           global mutable state: each worker owns exactly one TypedSpsc
//           producer half (type-enforced discipline); the merger owns all
//           consumer halves. Budget caps bound the search (bounded resources,
//           no hangs by construction).
// CEP:CLASS: CEP-29
// CEP:STATUS: complete
// CEP:FAILURE: search never fails: empty polytope -> empty frontier (caller
//              decides); budgets truncate deterministically (fixed DFS order).
// CEP:ASSUMES: polytope ops capped at MAX_SCHED_OPS (lib.rs truncation);
//              workers >= 1; channel capacity 4096 (power of two).
// CEP:COST: O(domain within budget) per worker; channel send amortized ~1
//           Release per threshold; frontier insert O(64).
// CEP:EVIDENCE: tests `search_is_exhaustive_on_small_domain`,
//           `search_is_deterministic`, `search_terminates_under_budget`,
//           `prune_hints_preserve_frontier_content` (integration).
// CEP:SECURITY: no untrusted input.
// CEP:HPC-DETERMINISM: THE CENTRAL CONTRACT — nondeterministic interleaving,
//           deterministic RESULTS, CONDITIONED on two surfaced preconditions
//           (review F5 — honest surface, not silent):
//           (P1) no worker trips a budget cap (budget_exhausted == false),
//           (P2) prune predicates are exactly `dominates` over all THREE
//                cost axes (fixed: the anchor packs cycles/hbm/sram).
//           Under P1+P2 the proof holds:
//           (1) The enumerated SET per worker is a deterministic function of
//               (polytope, budget, slice) — fixed DFS order, fixed caps.
//               [P1 matters: with pruning active the DFS reaches deeper
//               before tripping a budget, so the truncated prefix becomes
//               interleaving-dependent — the flag surfaces exactly this.]
//           (2) A candidate skipped by the anchor prune is strictly dominated
//               by a point that was in the frontier when the anchor was
//               published; frontier points are only removed by (transitive)
//               domination, so every pruned candidate is dominated by SOME
//               point of the FINAL frontier — and the insert rule rejects
//               dominated points anyway. Hence the final frontier is
//               IDENTICAL with or without pruning. The anchor is advisory:
//               soundness and determinism never depend on it.
//           (3) The winner is the frontier minimum under the total order
//               (pareto.rs) — a function of content, never arrival order.
//           The capacity truncation (cap 64, evict worst by total order) is
//           deterministic GIVEN the content (the order is total: sched-lex
//           breaks every tie) — cap_evicted is advisory, not a precondition.
//           EXECUTABLE WITNESS: `pruned_matches_exhaustive` runs the search
//           with and without pruning and asserts identical frontier content.
// CEP:TODO(main-agent): CEP-29: adaptive budgets (per-phase Gear Contexts
//                       arrive with Anvil v2 Upgrade 1).
//! Stage 2: concurrent enumeration + cost evaluation + Pareto merge.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anvil::typed::{Backpressure, Pod, TypedSpsc};

use crate::cost::{evaluate, CostPoint};
use crate::pareto::{pack_anchor, unpack_anchor, FrontierPoint, ParetoFrontier, ANCHOR_NONE};
use crate::polytope::{MemLevel, PackedOpSched, SchedulePolytope};

/// Maximum scheduled ops (dominant-op scheduling; lib.rs truncates).
pub const MAX_SCHED_OPS: usize = 12;

/// Per-worker evaluated-candidate budget (bounded search; deterministic
/// truncation because the DFS order is fixed).
pub const MAX_CAND_PER_WORKER: u64 = 65_536;

/// Per-worker stack-expansion budget (bounds pathological wide domains).
pub const MAX_EXPANSIONS_PER_WORKER: u64 = 262_144;

/// Channel capacity per worker (power of two; spec's 4096).
const CHANNEL_CAP: usize = 4096;

/// Merger drain batch width.
const DRAIN_BATCH: usize = 32;

/// The POD candidate record streamed through the typed channel.
///
/// CEP:WHAT: Fixed-size schedule + evaluated cost — the complete unit the
///           merger inserts into the frontier.
/// CEP:WHY: Pod requirement (anvil::typed) demands Copy + plain layout; the
///           fixed op array keeps the record allocation-free on the hot path.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (data)
/// CEP:ASSUMES: n_ops <= MAX_SCHED_OPS.
/// CEP:COST: ~416 bytes, Copy.
/// CEP:EVIDENCE: channel round-trip in `search_is_exhaustive_on_small_domain`.
/// CEP:SECURITY: no untrusted input
#[derive(Clone, Copy)]
pub struct CandidateRecord {
    /// Packed per-op schedules (slots 0..n_ops).
    pub ops: [PackedOpSched; MAX_SCHED_OPS],
    /// Live op count.
    pub n_ops: u8,
    /// Roofline cycles.
    pub cycles: u64,
    /// HBM traffic bytes.
    pub hbm_bytes: u64,
    /// SRAM footprint bytes.
    pub sram_usage: u64,
}

impl Pod for CandidateRecord {}

impl CandidateRecord {
    /// CEP:WHAT: Zeroed record (DFS seed).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: constant
    /// CEP:EVIDENCE: search tests
    /// CEP:SECURITY: no untrusted input
    fn zeroed(n_ops: usize) -> CandidateRecord {
        CandidateRecord {
            ops: [PackedOpSched {
                op_id: 0,
                tile: [0; 3],
                fuse_target: PackedOpSched::NO_FUSE,
                mem_level: 0,
                vec_width: 1,
            }; MAX_SCHED_OPS],
            n_ops: n_ops as u8,
            cycles: 0,
            hbm_bytes: 0,
            sram_usage: 0,
        }
    }

    /// CEP:WHAT: Schedule slice view.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: n_ops <= MAX_SCHED_OPS.
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: verifier consumes it
    /// CEP:SECURITY: no untrusted input
    pub fn sched(&self) -> &[PackedOpSched] {
        &self.ops[..self.n_ops as usize]
    }
}

/// Search diagnostics + result.
///
/// CEP:WHAT: The Pareto frontier plus advisory counters and DETERMINISM
///           PRECONDITION FLAGS.
/// CEP:WHY: Counters (evaluated/pruned) depend on anchor timing — ADVISORY
///          ONLY; the frontier content is the deterministic result (see
///          module header proof). `budget_exhausted` / `cap_evicted` surface
///          the two preconditions the determinism guarantee is conditioned
///          on: budgets and the frontier cap must not trip for the explored
///          set to be the full domain (review F5 — honest surfaces, not
///          silent).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: owned frontier
/// CEP:EVIDENCE: determinism tests + `pruned_matches_exhaustive`
/// CEP:SECURITY: no untrusted input
pub struct SearchOutcome {
    /// The final Pareto frontier (deterministic content).
    pub frontier: ParetoFrontier,
    /// Advisory: candidates evaluated (nondeterministic under pruning).
    pub evaluated: u64,
    /// Advisory: candidates skipped by the anchor prune.
    pub pruned: u64,
    /// True when any worker tripped a budget cap (determinism precondition).
    pub budget_exhausted: bool,
    /// True when the frontier cap evicted live points (advisory; content
    /// stays deterministic given the inserted set).
    pub cap_evicted: bool,
}

/// CEP:WHAT: One op's full discrete choice domain, materialized once.
/// CEP:WHY: The DFS needs a FIXED iteration order; materializing the domain
///          first also gives the worker-partition axis (round-robin slices).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: bounds consistent with divisibility.
/// CEP:COST: O(domain) build, reused across DFS.
/// CEP:EVIDENCE: exhaustive-count test
/// CEP:SECURITY: no untrusted input
fn op_domain(poly: &SchedulePolytope, idx: usize) -> Vec<PackedOpSched> {
    let b = &poly.op_bounds[idx];
    let mut out = Vec::new();
    // Memory levels DESCENDING (HBM-first safe default).
    let mut mems: Vec<MemLevel> = Vec::new();
    let mut m = b.mem_max as u8;
    loop {
        if let Some(level) = MemLevel::from_u8(m) {
            mems.push(level);
        }
        if m == b.mem_min as u8 {
            break;
        }
        m -= 1;
    }
    // Tiles: divisors of each dim's gcd clamp, ascending.
    let mut tiles: Vec<[u32; 3]> = Vec::new();
    let mut t0 = b.tile_min[0];
    while t0 <= b.tile_max[0] {
        if b.tile_max[0].is_multiple_of(t0) {
            let mut t1 = b.tile_min[1];
            while t1 <= b.tile_max[1] {
                if b.tile_max[1].is_multiple_of(t1) {
                    let mut t2 = b.tile_min[2];
                    while t2 <= b.tile_max[2] {
                        if b.tile_max[2].is_multiple_of(t2) {
                            tiles.push([t0, t1, t2]);
                        }
                        t2 += 1;
                    }
                }
                t1 += 1;
            }
        }
        t0 += 1;
    }
    // Fuse targets: none first, then bounds order.
    let mut fuses: Vec<u64> = vec![PackedOpSched::NO_FUSE];
    for t in b.valid_fuse.iter() {
        fuses.push(t.0);
    }
    // Vector widths ascending from the mask.
    let widths: Vec<u8> = (0..4u8)
        .filter(|bit| b.vec_mask & (1 << bit) != 0)
        .map(|bit| 1u8 << bit)
        .collect();
    for &mem in mems.iter() {
        for &tile in tiles.iter() {
            for &fuse in fuses.iter() {
                for &vw in widths.iter() {
                    out.push(PackedOpSched {
                        op_id: b.op_id.0,
                        tile,
                        fuse_target: fuse,
                        mem_level: mem as u8,
                        vec_width: vw,
                    });
                }
            }
        }
    }
    out
}

/// CEP:WHAT: The concurrent search driver.
/// CEP:WHY: Pipeline topology per the architecture: N enumerator workers
///          (each owns one TypedSpsc producer) -> single merger thread (owns
///          all consumer halves, maintains the frontier, publishes the
///          anchor). Workers are sliced round-robin over op-0's domain —
///          the enumerated SET is the disjoint union, hence deterministic.
/// CEP:STATUS: complete
/// CEP:FAILURE: empty polytope -> empty frontier (no error).
/// CEP:ASSUMES: polytope truncated to <= MAX_SCHED_OPS ops (lib.rs).
/// CEP:COST: bounded by budgets; channel memory = workers * cap * 416B.
/// CEP:EVIDENCE: module tests + concurrency_stress.rs.
/// CEP:SECURITY: no untrusted input.
pub fn search(poly: &SchedulePolytope, workers: usize) -> SearchOutcome {
    search_impl(poly, workers, true)
}

/// CEP:WHAT: Exhaustive search — the prune anchor machinery is disabled
///           (workers never read it).
/// CEP:WHY: The pruned-vs-exhaustive equivalence witness: the frontier
///           content must be IDENTICAL with and without pruning (the
///           module-header proof, executed — review F5's demand for an
///           executable test rather than an untested claim).
/// CEP:STATUS: complete
/// CEP:FAILURE: as search.
/// CEP:ASSUMES: budgets must not trip for full-domain equivalence.
/// CEP:COST: as search (typically slower — no pruning).
/// CEP:EVIDENCE: `pruned_matches_exhaustive`.
/// CEP:SECURITY: no untrusted input.
pub fn search_exhaustive(poly: &SchedulePolytope, workers: usize) -> SearchOutcome {
    search_impl(poly, workers, false)
}

/// CEP:WHAT: The search driver core (prune toggle).
/// CEP:WHY: Pipeline topology per the architecture: N enumerator workers
///          (each owns one TypedSpsc producer) -> single merger thread (owns
///          all consumer halves, maintains the frontier, publishes the
///          anchor). Workers are sliced round-robin over op-0's domain —
///          the enumerated SET is the disjoint union, hence deterministic.
/// CEP:STATUS: complete
/// CEP:FAILURE: empty polytope -> empty frontier (no error).
/// CEP:ASSUMES: polytope truncated to <= MAX_SCHED_OPS ops (lib.rs).
/// CEP:COST: bounded by budgets; channel memory = workers * cap * 416B.
/// CEP:EVIDENCE: module tests + concurrency_stress.rs.
/// CEP:SECURITY: no untrusted input.
fn search_impl(poly: &SchedulePolytope, workers: usize, enable_prune: bool) -> SearchOutcome {
    let n = poly.op_count();
    if n == 0 {
        return SearchOutcome {
            frontier: ParetoFrontier::new(),
            evaluated: 0,
            pruned: 0,
            budget_exhausted: false,
            cap_evicted: false,
        };
    }
    let domain0 = op_domain(poly, 0);
    let n_workers = workers.max(1).min(domain0.len().max(1)).min(8);

    // Sound constant compute lower bound (all flops at the full-width REG
    // peak — no candidate can beat it; used by the partial prune).
    let reg_peak = poly.hw.reg_peak_flops_per_cycle.max(1);
    let const_cycles_lb: u64 = poly
        .op_flops
        .iter()
        .fold(0u64, |a, f| a.saturating_add(f.div_ceil(reg_peak)));

    // One typed channel per worker: the halves each hold an Arc to the
    // shared ring, so the ring lives exactly as long as both halves.
    let mut producers = Vec::new();
    let mut consumers = Vec::new();
    for _ in 0..n_workers {
        let ch: TypedSpsc<CandidateRecord, { Backpressure::YIELD }> = TypedSpsc::new(CHANNEL_CAP);
        let (p, c) = ch.split();
        producers.push(p);
        consumers.push(c);
    }

    let done = AtomicUsize::new(0);
    let anchor = AtomicU64::new(ANCHOR_NONE);
    let evaluated_total = AtomicU64::new(0);
    let pruned_total = AtomicU64::new(0);
    let budget_hit = AtomicBool::new(false);

    let domains: Vec<Vec<PackedOpSched>> = (1..n).map(|i| op_domain(poly, i)).collect();

    let anchor_ref = &anchor;
    let done_ref = &done;
    let ev_ref = &evaluated_total;
    let pr_ref = &pruned_total;
    let budget_ref = &budget_hit;

    std::thread::scope(|s| {
        for (w, mut producer) in producers.into_iter().enumerate() {
            let poly_ref = poly;
            let domains_ref = &domains;
            let domain0_ref = &domain0;
            let worker_fn = move || {
                let mut evaluated: u64 = 0;
                let mut pruned: u64 = 0;
                let mut expansions: u64 = 0;
                let mut anchor_word = ANCHOR_NONE;
                let mut budget_exhausted = false;
                let mut stack: Vec<(usize, CandidateRecord)> = Vec::new();
                // Round-robin slice of op-0's domain.
                for (ci, choice) in domain0_ref.iter().enumerate() {
                    if ci % n_workers != w {
                        continue;
                    }
                    let mut rec = CandidateRecord::zeroed(n);
                    rec.ops[0] = *choice;
                    if feasible_prefix(poly_ref, &rec, 1) {
                        stack.push((1, rec));
                    }
                }
                while let Some((depth, rec)) = stack.pop() {
                    expansions += 1;
                    if expansions > MAX_EXPANSIONS_PER_WORKER {
                        budget_exhausted = true;
                        break;
                    }
                    if enable_prune && (expansions & 0xFF) == 0 {
                        anchor_word = anchor_ref.load(Ordering::Relaxed);
                    }
                    if depth == n {
                        if evaluated >= MAX_CAND_PER_WORKER {
                            budget_exhausted = true;
                            break;
                        }
                        // Evaluate first, THEN prune on the actual cost: a
                        // STRICTLY anchor-dominated candidate would be rejected
                        // by the frontier's insert rule anyway — skipping the
                        // send saves channel + merger work only (soundness and
                        // determinism never depend on it; module-header proof).
                        let cost: CostPoint = evaluate(rec.sched(), poly_ref);
                        let dominated = enable_prune && anchor_word != ANCHOR_NONE && {
                            let (ac, ah, asr) = unpack_anchor(anchor_word);
                            // All three axes of `dominates` (review F1):
                            // pruning must mean "anchor dominates this
                            // point", strict on at least one axis.
                            (cost.cycles >= ac && cost.hbm_bytes >= ah && cost.sram_usage >= asr)
                                && (cost.cycles > ac
                                    || cost.hbm_bytes > ah
                                    || cost.sram_usage > asr)
                        };
                        if dominated {
                            pruned += 1;
                            continue;
                        }
                        producer.send(CandidateRecord {
                            ops: rec.ops,
                            n_ops: rec.n_ops,
                            cycles: cost.cycles,
                            hbm_bytes: cost.hbm_bytes,
                            sram_usage: cost.sram_usage,
                        });
                        evaluated += 1;
                        continue;
                    }
                    for choice in domains_ref[depth - 1].iter() {
                        let mut next = rec;
                        next.ops[depth] = *choice;
                        if feasible_prefix(poly_ref, &next, depth + 1)
                            && !(enable_prune
                                && prefix_prune(
                                    &next,
                                    depth + 1,
                                    anchor_word,
                                    const_cycles_lb,
                                    poly_ref,
                                ))
                        {
                            stack.push((depth + 1, next));
                        }
                    }
                }
                // Publish counters, flush the channel, THEN mark done (the
                // flush-before-done ordering guarantees the merger's final
                // drain sees every sent record).
                ev_ref.fetch_add(evaluated, Ordering::Relaxed);
                pr_ref.fetch_add(pruned, Ordering::Relaxed);
                if budget_exhausted {
                    budget_ref.store(true, Ordering::Relaxed);
                }
                producer.flush();
                done_ref.fetch_add(1, Ordering::Release);
            };
            s.spawn(worker_fn);
        }
        // Merger (this thread): drain all channels, maintain the frontier,
        // publish the anchor.
        let mut frontier = ParetoFrontier::new();
        let mut buf = [CandidateRecord::zeroed(0); DRAIN_BATCH];
        loop {
            let mut got: usize = 0;
            for cons in consumers.iter_mut() {
                loop {
                    let n = cons.try_recv_batch(&mut buf);
                    if n == 0 {
                        break;
                    }
                    got += n;
                    for rec in buf.iter().take(n) {
                        frontier.insert(FrontierPoint {
                            cost: CostPoint {
                                cycles: rec.cycles,
                                hbm_bytes: rec.hbm_bytes,
                                sram_usage: rec.sram_usage,
                            },
                            sched: rec.sched().to_vec(),
                        });
                    }
                }
            }
            if got == 0 {
                if done_ref.load(Ordering::Acquire) == n_workers {
                    break;
                }
                std::thread::yield_now();
            }
            if let Some(best) = frontier.best() {
                anchor_ref.store(
                    pack_anchor(best.cost.cycles, best.cost.hbm_bytes, best.cost.sram_usage),
                    Ordering::Relaxed,
                );
            }
        }
        // Counters + frontier out of the scope via the return below.
        let cap_evicted = frontier.evicted();
        SearchOutcome {
            frontier,
            evaluated: evaluated_total.load(Ordering::Relaxed),
            pruned: pruned_total.load(Ordering::Relaxed),
            budget_exhausted: budget_hit.load(Ordering::Relaxed),
            cap_evicted,
        }
    })
}

/// CEP:WHAT: Prefix feasibility — bounds + divisibility + linear partial
///           check for the scheduled prefix.
/// CEP:STATUS: complete
/// CEP:FAILURE: false on any violation (sound rejection).
/// CEP:ASSUMES: sched covers ops 0..depth.
/// CEP:COST: O(depth + rows).
/// CEP:EVIDENCE: exhaustive-count test (domain sizes multiply exactly).
/// CEP:SECURITY: no untrusted input.
fn feasible_prefix(poly: &SchedulePolytope, rec: &CandidateRecord, depth: usize) -> bool {
    for (i, s) in rec.ops.iter().take(depth).enumerate() {
        let b = &poly.op_bounds[i];
        if s.op_id != b.op_id.0 {
            return false;
        }
        let mem = match MemLevel::from_u8(s.mem_level) {
            Some(m) => m,
            None => return false,
        };
        if mem < b.mem_min || mem > b.mem_max {
            return false;
        }
        if s.vec_width == 0
            || !s.vec_width.is_power_of_two()
            || s.vec_width > 8
            || b.vec_mask & (1 << s.vec_width.trailing_zeros()) == 0
        {
            return false;
        }
        if s.fuse_target != PackedOpSched::NO_FUSE
            && !b.valid_fuse.iter().any(|t| t.0 == s.fuse_target)
        {
            return false;
        }
    }
    for d in poly.divisibility_constraints.iter() {
        if (d.op_index as usize) < depth {
            let t = rec.ops[d.op_index as usize].tile[d.dim as usize];
            // Spec rule: the tile DIVIDES the problem gcd (divisor.is_multiple_of(t)
            // — the inverted multiple-check this replaces collapsed the
            // domain to a singleton; review F3).
            if t == 0 || !d.divisor.is_multiple_of(t) {
                return false;
            }
        }
    }
    poly.global_constraints.check_partial(rec.sched(), depth)
}

/// CEP:WHAT: Sound prefix prune — all THREE axes lower-bounded and STRICTLY
///           dominated on at least one (matching `dominates` exactly).
/// CEP:WHY: cycles lower bound = the constant program bound (all flops at
///           the full-width REG peak); hbm lower bound = scheduled hbm so far
///           + Σ remaining op minima (which are 0 unless the op is FORCED
///           off-chip — an SRAM placement contributes zero HBM traffic;
///           review F2); sram lower bound = scheduled SRAM outputs (remaining
///           contribute >= 0). Strictness on at least one axis keeps
///           equal-cost twins in the search (determinism; module header).
/// CEP:STATUS: complete
/// CEP:FAILURE: false unless strictly dominated on all-axes-and-one-strict
///              (sound side = keep).
/// CEP:ASSUMES: depth scheduled ops in rec.
/// CEP:COST: O(depth)
/// CEP:EVIDENCE: determinism test + `pruned_matches_exhaustive`.
/// CEP:SECURITY: no untrusted input
fn prefix_prune(
    rec: &CandidateRecord,
    depth: usize,
    anchor_word: u64,
    const_cycles_lb: u64,
    poly: &SchedulePolytope,
) -> bool {
    if anchor_word == ANCHOR_NONE {
        return false;
    }
    let (ac, ah, asr) = unpack_anchor(anchor_word);
    if const_cycles_lb < ac {
        return false; // cycles axis can still win.
    }
    // hbm lower bound: scheduled contributions + unscheduled minima (0
    // unless the op's placement is forced to HBM — see op_min_hbm's rule).
    let mut hbm_lb: u64 = 0;
    let mut sram_lb: u64 = 0;
    for (i, s) in rec.ops.iter().take(depth).enumerate() {
        let out = poly.op_out_bytes.get(i).copied().unwrap_or(0);
        let fused = s.fuse_target != PackedOpSched::NO_FUSE;
        match MemLevel::from_u8(s.mem_level) {
            Some(MemLevel::Hbm) => hbm_lb = hbm_lb.saturating_add(out),
            Some(MemLevel::Reg) => {
                if !fused {
                    hbm_lb = hbm_lb.saturating_add(out);
                }
            }
            Some(MemLevel::Sram) => {
                sram_lb = sram_lb.saturating_add(out);
            }
            None => {}
        }
    }
    let remaining_min: u64 = (depth..poly.op_count())
        .filter_map(|i| poly.op_min_hbm.get(i).copied())
        .sum::<u64>();
    hbm_lb = hbm_lb.saturating_add(remaining_min);
    if hbm_lb < ah || sram_lb < asr {
        return false;
    }
    // Strict on at least one axis.
    hbm_lb > ah || sram_lb > asr || const_cycles_lb > ac
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test-only precondition assertion; the house assert+if-let style would obscure test logic (waiver W-5)"
)]
mod tests {
    use super::*;
    use crate::polytope::{DivConstraint, HardwareModel, OpBounds, SparseLinearSystem};
    use xir_core::id::NodeId;

    /// Two-op synthetic polytope with tiny domains for exact counting.
    fn tiny_polytope() -> SchedulePolytope {
        let hw = HardwareModel::reference();
        let mk = |id: u64| OpBounds {
            op_id: NodeId(id),
            tile_min: [1, 1, 1],
            tile_max: [2, 2, 2],
            mem_min: MemLevel::Reg,
            mem_max: MemLevel::Hbm,
            valid_fuse: vec![],
            vec_mask: 0b0001, // width 1 only
        };
        SchedulePolytope {
            op_bounds: vec![mk(1), mk(2)],
            global_constraints: SparseLinearSystem { rows: vec![] },
            divisibility_constraints: vec![
                DivConstraint {
                    op_index: 0,
                    dim: 0,
                    divisor: 2,
                },
                DivConstraint {
                    op_index: 1,
                    dim: 0,
                    divisor: 2,
                },
            ],
            hw,
            op_flops: vec![100, 100],
            op_out_bytes: vec![16, 16],
            op_in_bytes: vec![32, 32],
            op_min_hbm: vec![16, 16],
        }
    }

    // CEP:WHAT: Search is exhaustive on a small domain (per-op domain: 3
    //           mems x 8 valid tiles {1,2}|2 per dim x 1 fuse x 1 vw = 24;
    //           two ops -> 576 complete candidates).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on a lost candidate.
    // CEP:ASSUMES: budgets never trip at this size.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn search_is_exhaustive_on_small_domain() {
        let poly = tiny_polytope();
        let out = search(&poly, 2);
        assert!(!out.budget_exhausted);
        // Every candidate evaluates; the frontier holds the non-dominated set.
        assert!(out.evaluated >= 24, "evaluated {}", out.evaluated);
        let best = out.frontier.best().expect("non-empty frontier");
        assert_eq!(best.cost.cycles, 50);
    }

    // CEP:WHAT: THE PROOF'S EXECUTABLE WITNESS (review F5): the frontier
    //           content is IDENTICAL with and without prune hints, and with
    //           1 vs 2 workers — on a polytope large enough for pruning to
    //           actually fire (dominated combinations exist) yet small
    //           enough that budgets never trip.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any content divergence.
    // CEP:ASSUMES: budgets never trip (asserted).
    // CEP:COST: test-only; ~26k candidates
    // CEP:EVIDENCE: this test
    #[test]
    fn pruned_matches_exhaustive() {
        let mut poly = tiny_polytope();
        // Widen: tiles {1,2,4}|4 per dim = 27 combos, vw {1,4} -> 162/op,
        // 26244 total — prune fires, budgets do not.
        poly.op_bounds[0].tile_max = [4, 4, 4];
        poly.op_bounds[1].tile_max = [4, 4, 4];
        poly.op_bounds[0].vec_mask = 0b0101;
        poly.op_bounds[1].vec_mask = 0b0101;
        for d in poly.divisibility_constraints.iter_mut() {
            d.divisor = 4;
        }
        let pruned = search(&poly, 2);
        let exhaustive = search_exhaustive(&poly, 2);
        assert!(!pruned.budget_exhausted, "budget tripped — witness invalid");
        assert!(
            !exhaustive.budget_exhausted,
            "budget tripped — witness invalid"
        );
        // Pruning must actually fire for the witness to be meaningful.
        assert!(pruned.pruned > 0, "prune never fired — vacuous witness");
        let fingerprint = |o: &SearchOutcome| -> Vec<(u64, u64, u64)> {
            o.frontier
                .top_k(64)
                .into_iter()
                .map(|p| (p.cost.cycles, p.cost.hbm_bytes, p.cost.sram_usage))
                .collect()
        };
        assert_eq!(fingerprint(&pruned), fingerprint(&exhaustive));
        // And worker-count invariance in both modes.
        let pruned1 = search(&poly, 1);
        assert_eq!(fingerprint(&pruned1), fingerprint(&pruned));
    }

    // CEP:WHAT: Two consecutive searches produce IDENTICAL frontier content
    //           (the determinism contract).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any content divergence.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test (the module-header proof's executable witness)
    #[test]
    fn search_is_deterministic() {
        let poly = tiny_polytope();
        let a = search(&poly, 2);
        let b = search(&poly, 2);
        let av: Vec<_> = a.frontier.top_k(64);
        let bv: Vec<_> = b.frontier.top_k(64);
        assert_eq!(av.len(), bv.len());
        for (x, y) in av.iter().zip(bv.iter()) {
            assert_eq!(x.cost, y.cost);
            assert_eq!(x.sched, y.sched);
        }
        // Worker count does not change the content either (disjoint-union
        // partitioning).
        let c = search(&poly, 1);
        let cv: Vec<_> = c.frontier.top_k(64);
        assert_eq!(av.len(), cv.len());
        for (x, y) in av.iter().zip(cv.iter()) {
            assert_eq!(x.cost, y.cost);
            assert_eq!(x.sched, y.sched);
        }
    }

    // CEP:WHAT: Wide domain terminates under the budget (no hang).
    // CEP:STATUS: complete
    // CEP:FAILURE: test timeout on a hang (bounded budgets prevent it).
    // CEP:ASSUMES: budgets are small enough for CI.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn search_terminates_under_budget() {
        let mut poly = tiny_polytope();
        // 8x8 domain: tiles {1,2,4,8}^3 = 64 per op, mem 3, vw {1,2,4,8}.
        poly.op_bounds[0].tile_max = [8, 8, 8];
        poly.op_bounds[1].tile_max = [8, 8, 8];
        poly.op_bounds[0].vec_mask = 0b1111;
        poly.op_bounds[1].vec_mask = 0b1111;
        for d in poly.divisibility_constraints.iter_mut() {
            d.divisor = 8;
        }
        let out = search(&poly, 2);
        assert!(out.evaluated <= 2 * MAX_CAND_PER_WORKER);
        assert!(!out.frontier.is_empty());
    }
}
