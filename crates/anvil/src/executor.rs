// CEP:FILE: crates/anvil/src/executor.rs
// CEP:WHAT: Anvil executor — thread-per-core fork/join with work stealing
//           (Gear 2) and static partitioning (Gear 1).
// CEP:WHY: Compiler passes declare their topology (master architecture
//          section 7) and the executor picks the gear. Gear 1 slices work
//          statically: zero atomics, zero locks, zero stealing inside task
//          bodies. Gear 2 runs generic fork/join jobs on per-worker Chase-Lev
//          deques with round-robin stealing; the FuelMeter policy decides
//          inline-vs-dequeue for search branches. The persistent park-based
//          pool (CEP-3) lives in pool.rs; the free entry points here route
//          to it when the topology matches, with the scoped variants as the
//          nesting-safe fallback.
// CEP:CLASS: CEP-1 (thread orchestration) / CEP-0 (task bodies)
// CEP:STATUS: complete
// CEP:FAILURE: `ExecError::DequeFull` when a worker's bounded deque overflows
//              (caller must drain or fail loudly); `ExecError::TooManyWorkers`
//              above MAX_WORKERS; `ExecError::ZeroWorkers`; `ExecError::
//              LengthMismatch` for run_partitioned slice disagreement;
//              `ExecError::Pool(JobPanicked)` when a job panics — the panic
//              is CAUGHT inside worker_loop (credit-safe) and reported
//              loudly instead of unwinding across a thread boundary.
// CEP:ASSUMES: worker count = available parallelism capped by MAX_WORKERS;
//              jobs are deterministic and side-effect isolated (HPC-0
//              contract) so scheduling nondeterminism never becomes
//              observable IR nondeterminism (CEP&CC 38.10): results are
//              combined by deterministic reduction, never by completion order.
// CEP:COST: scoped region setup = N thread spawns + N deque allocations
//           (CEP-1 init boundary); pooled region setup = one handoff
//           (see pool.rs); task dispatch = 1 deque pop or steal (CEP-0).
// CEP:EVIDENCE: tests `gear1_partition_is_exact`, `gear2_fib_tree`,
//           `gear2_steals_when_unbalanced`, `gear2_pending_reaches_zero`,
//           `gear1_empty_inputs_are_noop` (audit F-1 regression).
// CEP:SECURITY: scoped threads borrow only the region's deques; no 'static
//           leakage; jobs are first-party code.
// CEP:HPC-DETERMINISM: Gear 1 fully deterministic; Gear 2 task *results* are
//           deterministic (pure jobs + deterministic reduction); execution
//           interleaving is nondeterministic by design and unobservable.
// CEP:TODO(main-agent): CEP-2: NUMA-aware allocation policies (config extension).
//! Thread-per-core executor (Gears 1 and 2).

use core::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

use crate::chase_lev::DequeError;
use crate::config::{DEQUE_CAPACITY, MAX_WORKERS};
use crate::pad::CachePadded;

/// Failure enumeration for executor regions.
///
/// CEP:WHAT: Explicit error type for gear entry points.
/// CEP:WHY: Law 6: bounded deques, worker bounds and slice contracts must
///          fail loudly and explicitly, never silently.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size enum
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecError {
    /// A worker's bounded deque overflowed.
    DequeFull,
    /// Worker count exceeded MAX_WORKERS.
    TooManyWorkers,
    /// Worker count must be at least 1.
    ZeroWorkers,
    /// inputs and outputs lengths disagree (run_partitioned contract).
    LengthMismatch,
    /// Pool failure (panic escape or spawn failure — see pool::PoolError).
    Pool(crate::pool::PoolError),
}

/// A unit of fork/join work (Gear 2).
///
/// CEP:WHAT: Job trait: monomorphized, statically dispatched task type.
/// CEP:WHY: CEP&CC 25.6 bans trait objects in hot paths; the executor is
///          generic over `J` so every dispatch is a direct call, jobs live
///          inline in deque slots (no per-task Box), and one pass's whole
///          search tree shares one concrete job type.
/// CEP:STATUS: complete
/// CEP:FAILURE: implementations must not panic (panic-free job contract,
///              enforced by clippy::panic = deny on this workspace).
/// CEP:ASSUMES: `run` may call `ctx.spawn()` for children; results are
///              communicated through deterministic reduction state.
/// CEP:COST: one static call per job
/// CEP:EVIDENCE: gear2 tests
/// CEP:SECURITY: first-party implementations only
pub trait Job: Send {
    /// CEP:WHAT: Executes one job, possibly spawning children via ctx.
    /// CEP:WHY: The fork/join entry point.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: implementations must be panic-free.
    /// CEP:ASSUMES: ctx borrows are region-scoped.
    /// CEP:COST: implementation-defined
    /// CEP:EVIDENCE: implementation-defined
    /// CEP:SECURITY: implementation-defined
    fn run(&mut self, ctx: &mut WorkerCtx<'_, '_, Self>)
    where
        Self: Sized;
}

/// Per-worker context handed to running jobs.
///
/// CEP:WHAT: Borrow bundle: worker index, own deque, steal table, pending
///           counter.
/// CEP:WHY: Jobs spawn children through `spawn`, which increments the pending
///           counter BEFORE the push (termination correctness: a worker never
///           observes pending == 0 while an unpushed child exists) and fails
///           loudly on bounded-deque overflow.
/// CEP:STATUS: complete
/// CEP:FAILURE: `spawn` returns `ExecError::DequeFull` (caller decides).
/// CEP:ASSUMES: ctx is created by the executor per job execution.
/// CEP:COST: spawn = 1 fetch_add + 1 push.
/// CEP:EVIDENCE: gear2 tests
/// CEP:SECURITY: internal state only
pub struct WorkerCtx<'a, 'b, J: Job> {
    /// Index of the executing worker (0..n).
    pub worker_index: usize,
    /// This worker's deque (owner side).
    own: &'b crate::chase_lev::Deque<J>,
    /// All deques in the region (diagnostics; stealing is executor-internal).
    steal_table: &'b [CachePadded<crate::chase_lev::Deque<J>>],
    /// Global pending-job counter (termination).
    pending: &'b AtomicIsize,
    /// Lifetime binder for the region.
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<J: Job> WorkerCtx<'_, '_, J> {
    /// CEP:WHAT: Spawns a child job onto this worker's deque.
    /// CEP:WHY: Fork step of fork/join; see struct comment for the
    ///           increment-before-push termination argument.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: `DequeFull` when the bounded deque is at capacity; the
    ///               pending credit is rolled back so it never leaks.
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 fetch_add + 1 push
    /// CEP:EVIDENCE: test `gear2_pending_reaches_zero`
    /// CEP:SECURITY: none
    pub fn spawn(&mut self, child: J) -> Result<(), ExecError> {
        self.pending.fetch_add(1, Ordering::AcqRel);
        match self.own.push(child) {
            Ok(()) => Ok(()),
            Err(DequeError::Full) => {
                self.pending.fetch_sub(1, Ordering::AcqRel);
                Err(ExecError::DequeFull)
            }
            Err(_) => {
                self.pending.fetch_sub(1, Ordering::AcqRel);
                Err(ExecError::DequeFull)
            }
        }
    }

    /// CEP:WHAT: Number of worker deques in this region.
    /// CEP:WHY: Search jobs size their per-worker budgets deterministically.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: gear2 tests
    pub fn worker_count(&self) -> usize {
        self.steal_table.len()
    }
}

/// CEP:WHAT: Resolves the default worker count (thread-per-core).
/// CEP:WHY: available_parallelism reflects logical cores; capping at
///          MAX_WORKERS bounds thread-stack memory (CEP&CC 39).
/// CEP:STATUS: complete
/// CEP:FAILURE: none; falls back to 1 worker if the OS query fails.
/// CEP:ASSUMES: none
/// CEP:COST: one syscall at init
/// CEP:EVIDENCE: tests use explicit counts
/// CEP:SECURITY: none
pub fn default_worker_count() -> usize {
    match std::thread::available_parallelism() {
        Ok(n) => (n.get()).min(MAX_WORKERS),
        Err(_) => 1,
    }
}

/// CEP:WHAT: Gear 1 — static partitioning with zero synchronization
///           (scoped variant: threads spawn per region).
/// CEP:WHY: Level 0/1 canonicalization, DCE, CSE across independent slices:
///          workers get disjoint index ranges, write disjoint output ranges,
///          and never touch an atomic inside `f` (arch: "Workers are assigned
///          disjoint subgraphs ... zero atomics, zero locks, zero stealing").
///          The free function `run_partitioned` routes to the persistent
///          pool when the topology matches (CEP-3); this scoped variant is
///          the nesting-safe fallback and the explicit-control entry point.
/// CEP:STATUS: complete
/// CEP:FAILURE: see ExecError; panics from `f` propagate through
///              thread::scope (f is first-party CEP-0 code, panic-free by
///              lint).
/// CEP:ASSUMES: `f` is deterministic and pure w.r.t. its input element so
///              scheduling cannot change outputs (HPC determinism).
/// CEP:COST: N spawns + N joins (CEP-1 boundary); body cost is caller's.
/// CEP:EVIDENCE: test `gear1_partition_is_exact`
/// CEP:SECURITY: outputs written only at the caller's disjoint indices.
/// CEP:HPC-DETERMINISM: outputs[i] = f(inputs[i]) — order-independent.
pub fn run_partitioned_scoped<T, R, F>(
    inputs: &[T],
    outputs: &mut [R],
    num_workers: usize,
    f: F,
) -> Result<(), ExecError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    check_worker_count(num_workers)?;
    if inputs.len() != outputs.len() {
        return Err(ExecError::LengthMismatch);
    }
    // Empty inputs: nothing to compute. (chunks(0) would panic — audit F-1;
    // the pooled twin already treats empty as a no-op, so both paths agree.)
    if inputs.is_empty() {
        return Ok(());
    }
    let chunk = inputs.len().div_ceil(num_workers);
    let chunks: Vec<&[T]> = inputs.chunks(chunk).collect();
    let mut out_chunks: Vec<&mut [R]> = outputs.chunks_mut(chunk).collect();
    std::thread::scope(|s| {
        for (cin, cout) in chunks.iter().zip(out_chunks.iter_mut()) {
            let f = &f;
            s.spawn(move || {
                // Scoped workers carry the in-region flag (audit F-2): a
                // nested pooled call from inside this body must fall back
                // to scoped spawning too, or it would wait on the enclosing
                // region's pool — a self-deadlock.
                crate::pool::set_in_region_scoped(true);
                // Disjoint slices: no atomics, no locks, no stealing.
                for (i, item) in cin.iter().enumerate() {
                    cout[i] = f(item);
                }
                crate::pool::set_in_region_scoped(false);
            });
        }
    });
    Ok(())
}

/// CEP:WHAT: Gear 1 — static partitioning (free entry point; routes to the
///           persistent pool when the topology matches).
/// CEP:WHY: CEP-3: production callers pass default_worker_count(); routing
///          through the global pool cuts per-region thread spawns without
///          API changes. Nested calls and mismatched topologies use the
///          scoped fallback — deadlock-free by construction.
/// CEP:STATUS: complete
/// CEP:FAILURE: see pool::run_partitioned_pooled / run_partitioned_scoped.
/// CEP:ASSUMES: `f` deterministic and pure w.r.t. its element.
/// CEP:COST: pooled handoff or scoped spawns.
/// CEP:EVIDENCE: pool tests (`pool_partition_matches_scoped`, routing).
/// CEP:HPC-DETERMINISM: identical outputs on both paths.
pub fn run_partitioned<T, R, F>(
    inputs: &[T],
    outputs: &mut [R],
    num_workers: usize,
    f: F,
) -> Result<(), ExecError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    crate::pool::run_partitioned_pooled(inputs, outputs, num_workers, f)
}

/// CEP:WHAT: Gear 2 — fork/join with work stealing over generic jobs
///           (scoped variant: threads spawn per region).
/// CEP:WHY: Fusion search, e-graph saturation, auto-tuning: irregular trees
///          where stealing beats static slicing (arch section 2). Workers pop
///          LIFO from their own deque (cache-warm continuation) and steal
///          FIFO from neighbors (oldest = biggest subtree). The free
///          function `run_fork_join` routes to the persistent pool when the
///          topology matches (CEP-3); this scoped variant is the
///          nesting-safe fallback.
/// CEP:STATUS: complete
/// CEP:FAILURE: `DequeFull` if the root push overflows; job panics propagate
///              via scope (jobs are panic-free by contract).
/// CEP:ASSUMES: jobs communicate through deterministic shared reduction state
///              guarded by their own protocol (e.g. atomic pruning in fusion).
/// CEP:COST: region setup N deque allocs + N spawns; dispatch = pop/steal.
/// CEP:EVIDENCE: tests `gear2_fib_tree`, `gear2_steals_when_unbalanced`
/// CEP:SECURITY: scoped borrows only
/// CEP:HPC-DETERMINISM: nondeterministic interleaving, deterministic results
///           (pure jobs + deterministic reduction — the pass's contract).
pub fn run_fork_join_scoped<J: Job>(root: J, num_workers: usize) -> Result<(), ExecError> {
    check_worker_count(num_workers)?;
    let mut deques: Vec<CachePadded<crate::chase_lev::Deque<J>>> = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        deques.push(CachePadded::new(*crate::chase_lev::Deque::new(
            DEQUE_CAPACITY,
        )));
    }
    // Seed the root job on worker 0 and open its pending credit.
    let pending = AtomicIsize::new(1);
    if deques[0].value.push(root).is_err() {
        return Err(ExecError::DequeFull);
    }
    let table: &[CachePadded<crate::chase_lev::Deque<J>>] = &deques;
    let pending_ref = &pending;
    // Panic reporting: worker_loop catches job panics (credit-safe) and
    // reports them through its return value; the flag converts that into
    // the loud ExecError below (no unwind crosses the scope boundary).
    let any_panic = AtomicBool::new(false);
    let flag = &any_panic;
    std::thread::scope(|s| {
        for w in 0..num_workers {
            let own = &table[w].value;
            s.spawn(move || {
                // Scoped workers carry the in-region flag (audit F-2): nested
                // pooled calls from job bodies fall back to scoped spawning.
                crate::pool::set_in_region_scoped(true);
                let saw = worker_loop(w, own, table, pending_ref);
                crate::pool::set_in_region_scoped(false);
                if saw {
                    flag.store(true, Ordering::Release);
                }
            });
        }
    });
    // Quiescent: workers exit only at pending == 0, and scope joins them all.
    debug_assert_eq!(pending.load(Ordering::Acquire), 0);
    if any_panic.load(Ordering::Acquire) {
        return Err(ExecError::Pool(crate::pool::PoolError::JobPanicked));
    }
    Ok(())
}

/// CEP:WHAT: Gear 2 — fork/join (free entry point; routes to the
///           persistent pool when the topology matches).
/// CEP:WHY: CEP-3 routing discipline shared with run_partitioned: matching
///          topologies amortize thread cost through the global pool;
///          nested calls and mismatches use the scoped fallback.
/// CEP:STATUS: complete
/// CEP:FAILURE: see pool::run_fork_join_pooled / run_fork_join_scoped.
/// CEP:ASSUMES: see the Job trait.
/// CEP:COST: pooled handoff or scoped spawns.
/// CEP:EVIDENCE: pool tests.
/// CEP:HPC-DETERMINISM: deterministic results on both paths.
pub fn run_fork_join<J: Job>(root: J, num_workers: usize) -> Result<(), ExecError> {
    crate::pool::run_fork_join_pooled(root, num_workers)
}

/// CEP:WHAT: One worker's dispatch loop (pop own -> run; else steal).
/// CEP:WHY: Termination soundness: every queued/running job holds one pending
///           credit; a worker exits only when pending == 0; a Busy steal
///           retries own pop first (a push may have landed meanwhile).
///           PANIC DISCIPLINE (pool-era, CEP-3): jobs run under
///           catch_unwind and the credit is released on BOTH paths — a
///           panicked job must never leak its credit, or quiescence never
///           arrives and the pool barrier deadlocks. A caught panic is
///           reported through the return value (saw-panic) instead of an
///           unwind crossing a thread boundary.
/// CEP:STATUS: complete
/// CEP:FAILURE: deques drain fully on Drop (executor contract in
///              chase_lev.rs).
/// CEP:ASSUMES: deques outlive the scope; pending is shared.
/// CEP:COST: 1 pop or steal attempt per iteration (+ unwind setup only on
///           the panic path).
/// CEP:EVIDENCE: gear2 tests; chase_lev stress test;
///           `pool_panicking_job_reported_loudly`.
/// CEP:SECURITY: none
pub(crate) fn worker_loop<J: Job>(
    worker_index: usize,
    own: &crate::chase_lev::Deque<J>,
    table: &[CachePadded<crate::chase_lev::Deque<J>>],
    pending: &AtomicIsize,
) -> bool {
    let mut saw_panic = false;
    loop {
        if pending.load(Ordering::Acquire) == 0 {
            return saw_panic;
        }
        match own.pop() {
            Ok(mut job) => {
                let mut ctx = WorkerCtx {
                    worker_index,
                    own,
                    steal_table: table,
                    pending,
                    _marker: core::marker::PhantomData,
                };
                if run_job_catching(&mut job, &mut ctx) {
                    saw_panic = true;
                }
                pending.fetch_sub(1, Ordering::AcqRel);
            }
            Err(DequeError::Empty) => {
                let n = table.len();
                let mut activity = false;
                if n > 1 {
                    for k in 1..n {
                        let victim = (worker_index + k) % n;
                        match table[victim].value.steal() {
                            Ok(mut job) => {
                                let mut ctx = WorkerCtx {
                                    worker_index,
                                    own,
                                    steal_table: table,
                                    pending,
                                    _marker: core::marker::PhantomData,
                                };
                                if run_job_catching(&mut job, &mut ctx) {
                                    saw_panic = true;
                                }
                                pending.fetch_sub(1, Ordering::AcqRel);
                                activity = true;
                                break;
                            }
                            Err(DequeError::Busy) => {
                                // Own deque state may have changed: retry pop.
                                activity = true;
                                break;
                            }
                            Err(DequeError::Empty) => continue,
                            Err(DequeError::Full) => continue,
                        }
                    }
                }
                if !activity {
                    if pending.load(Ordering::Acquire) == 0 {
                        return saw_panic;
                    }
                    std::thread::yield_now();
                }
            }
            Err(DequeError::Busy) => continue,
            Err(DequeError::Full) => continue,
        }
    }
}

/// CEP:WHAT: Runs one job under catch_unwind; reports a caught panic.
/// CEP:WHY: The panic-free-jobs contract is enforced by lint on THIS
///          workspace, but third-party Job impls could break it; an
///          escaped unwind would leak the job's pending credit (pool
///          deadlock) or cross a thread boundary (scoped executor).
///          Converting it to a boolean keeps termination sound on both
///          executors.
/// CEP:STATUS: complete
/// CEP:FAILURE: returns true when the job panicked (loud report path).
/// CEP:ASSUMES: job is first-party or contract-bound.
/// CEP:COST: unwind setup only on the panic path.
/// CEP:EVIDENCE: `pool_panicking_job_reported_loudly`.
fn run_job_catching<J: Job>(job: &mut J, ctx: &mut WorkerCtx<'_, '_, J>) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run(ctx))).is_err()
}

/// CEP:WHAT: Validates worker-count bounds.
/// CEP:WHY: Law 6 + bounded resources (CEP&CC 39): 1..=MAX_WORKERS.
/// CEP:STATUS: complete
/// CEP:FAILURE: ZeroWorkers / TooManyWorkers.
/// CEP:ASSUMES: none
/// CEP:COST: 2 compares
/// CEP:EVIDENCE: tests
fn check_worker_count(n: usize) -> Result<(), ExecError> {
    if n == 0 {
        Err(ExecError::ZeroWorkers)
    } else if n > MAX_WORKERS {
        Err(ExecError::TooManyWorkers)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    // CEP:WHAT: Gear 1 partitions every input exactly once.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on lost/duplicated indices.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    // CEP:WHAT: Empty inputs are a no-op on BOTH routing paths (audit F-1:
    //           chunks(0) panicked on the scoped path; the pooled twin
    //           already no-oped — behavior must not depend on routing).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if either path panics or errors.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn gear1_empty_inputs_are_noop() {
        let inputs: [i64; 0] = [];
        let mut outputs: [i64; 0] = [];
        let r = run_partitioned_scoped(&inputs, &mut outputs, 4, |x| x.wrapping_mul(2));
        assert_eq!(r, Ok(()));
        let pool = crate::pool::Pool::new(2);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            let r2 = p.run_partitioned(&inputs, &mut outputs, |x| x.wrapping_mul(2));
            assert_eq!(r2, Ok(()));
        }
    }

    #[test]
    fn gear1_partition_is_exact() {
        let inputs: Vec<u64> = (0..1000u64).collect();
        let mut outputs: Vec<u64> = vec![0; 1000];
        let r = run_partitioned(&inputs, &mut outputs, 4, |x| x.wrapping_mul(2));
        assert!(r.is_ok());
        for (i, o) in outputs.iter().enumerate() {
            assert_eq!(*o, (i as u64) * 2);
        }
        // Length mismatch is loud.
        let mut bad: Vec<u64> = vec![0; 999];
        assert_eq!(
            run_partitioned(&inputs, &mut bad, 4, |x| x.wrapping_mul(2)),
            Err(ExecError::LengthMismatch)
        );
    }

    /// Recursive Fibonacci job for Gear 2.
    struct FibJob<'a> {
        n: u32,
        result_slot: &'a AtomicU64,
    }

    impl Job for FibJob<'_> {
        fn run(&mut self, ctx: &mut WorkerCtx<'_, '_, Self>) {
            if self.n <= 12 {
                self.result_slot
                    .fetch_add(fib_plain(self.n), Ordering::AcqRel);
                return;
            }
            let a = FibJob {
                n: self.n - 1,
                result_slot: self.result_slot,
            };
            let b = FibJob {
                n: self.n - 2,
                result_slot: self.result_slot,
            };
            if ctx.spawn(a).is_ok() && ctx.spawn(b).is_ok() {
                // Children took over this subtree.
            } else {
                // Deque full: compute inline (degradation is allowed and loud).
                self.result_slot
                    .fetch_add(fib_plain(self.n), Ordering::AcqRel);
            }
        }
    }

    fn fib_plain(n: u32) -> u64 {
        let (mut a, mut b) = (0u64, 1u64);
        for _ in 0..n {
            let t = a.wrapping_add(b);
            a = b;
            b = t;
        }
        a
    }

    // CEP:WHAT: Gear 2 computes a fib tree correctly under stealing.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on wrong total.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn gear2_fib_tree() {
        let acc = AtomicU64::new(0);
        let root = FibJob {
            n: 20,
            result_slot: &acc,
        };
        assert!(run_fork_join(root, 4).is_ok());
        assert_eq!(acc.load(Ordering::Acquire), fib_plain(20));
    }

    // CEP:WHAT: One deep chain forces stealing or patient polling and still
    //           completes (termination correctness).
    // CEP:STATUS: complete
    // CEP:FAILURE: test hangs on a termination bug (CI timeout catches it).
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn gear2_steals_when_unbalanced() {
        struct Chain<'a> {
            depth: u32,
            hits: &'a AtomicU64,
        }
        impl Job for Chain<'_> {
            fn run(&mut self, ctx: &mut WorkerCtx<'_, '_, Self>) {
                self.hits.fetch_add(1, Ordering::AcqRel);
                if self.depth > 0 {
                    let child = Chain {
                        depth: self.depth - 1,
                        hits: self.hits,
                    };
                    if ctx.spawn(child).is_err() {
                        // Inline fallback keeps correctness.
                        self.depth -= 1;
                        self.run(ctx);
                    }
                }
            }
        }
        let hits = AtomicU64::new(0);
        let root = Chain {
            depth: 500,
            hits: &hits,
        };
        assert!(run_fork_join(root, 4).is_ok());
        assert_eq!(hits.load(Ordering::Acquire), 501);
    }

    // CEP:WHAT: Pending counter returns exactly to zero after a region.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires (debug) if credits leak.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn gear2_pending_reaches_zero() {
        struct Noop {
            children: u32,
        }
        impl Job for Noop {
            fn run(&mut self, ctx: &mut WorkerCtx<'_, '_, Self>) {
                if self.children > 0 {
                    let c = Noop {
                        children: self.children - 1,
                    };
                    if ctx.spawn(c).is_err() {
                        self.children -= 1;
                    }
                }
            }
        }
        let root = Noop { children: 100 };
        assert!(run_fork_join(root, 4).is_ok());
    }

    // CEP:WHAT: Worker-count bounds are enforced loudly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if bounds drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn worker_bounds_are_enforced() {
        struct Empty {}
        impl Job for Empty {
            fn run(&mut self, _ctx: &mut WorkerCtx<'_, '_, Self>) {}
        }
        assert_eq!(run_fork_join(Empty {}, 0), Err(ExecError::ZeroWorkers));
        assert_eq!(
            run_fork_join(Empty {}, MAX_WORKERS + 1),
            Err(ExecError::TooManyWorkers)
        );
    }
}
