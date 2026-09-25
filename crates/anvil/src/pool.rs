// CEP:FILE: crates/anvil/src/pool.rs
// CEP:WHAT: Persistent park-based worker pool (CEP-3) — helpers survive
//           across regions; region state is LENT to them per region.
// CEP:WHY: The scoped executor spawns N threads per region (setup cost on
//          every pass round). The pool parks N-1 persistent helpers and
//          keeps worker 0 on the caller's thread: region setup drops to
//          one mutex/condvar handoff (CEP-1 cold path) instead of N spawns.
//          Gear semantics are unchanged: Gear 1 stays zero-atomic
//          strided-slice partitioning; Gear 2 keeps the same worker_loop
//          (LIFO pop, FIFO steal, pending-credit termination).
// CEP:CLASS: CEP-1 (thread orchestration; the ONLY unsafe lives here and
//           in chase_lev/spsc/ebr)
// CEP:STATUS: complete
// CEP:FAILURE: ExecError::{TooManyWorkers, ZeroWorkers, DequeFull,
//              JobPanicked, SpawnFailed}; region publication serializes
//              through the handoff (concurrent callers wait — documented).
// CEP:ASSUMES: jobs are panic-free by contract (workspace clippy::panic =
//              deny); the catch_unwind discipline is defensive against
//              future third-party Job impls.
// CEP:COST: region setup = 1 lock + 1 notify + N wakeups (vs N spawns);
//           helper dispatch = 1 trampoline call. Parking uses a condvar —
//           the CEP-1 boundary, never inside task bodies.
// CEP:EVIDENCE: tests `pool_partition_is_exact`, `pool_partition_matches
//           _scoped`, `pool_fork_join_fib_tree`, `pool_regions_reuse
//           _workers`, `pool_nested_partition_falls_back`,
//           `pool_concurrent_callers_serialize`,
//           `pool_panicking_job_reported_loudly`.
// CEP:SECURITY: helpers run first-party trampolines only; the handoff
//           carries no untrusted data.
// CEP:HPC-DETERMINISM: Gear 1 results are scheduling-independent (strided
//           disjoint slices); Gear 2 keeps the executor's contract (pure
//           jobs + deterministic reduction). Pool vs scoped produces
//           identical outputs (`pool_partition_matches_scoped`).
//! Persistent worker pool with per-region lending.

use core::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, AtomicUsize, Ordering};
use std::cell::Cell;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;

use crate::chase_lev::Deque;
use crate::config::{DEQUE_CAPACITY, MAX_WORKERS};
use crate::executor::{worker_loop, ExecError};
use crate::pad::CachePadded;

/// Marker error: a job panicked inside the pool (defensive path).
///
/// CEP:WHAT: Loud failure for panic escapes.
/// CEP:WHY: Jobs are panic-free by contract, but the pool must never
///          deadlock or corrupt region state if the contract is broken:
///          the panicking worker is caught, checks in, and the caller gets
///          this error instead of a hang.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: zero-size
/// CEP:EVIDENCE: test `pool_panicking_job_reported_loudly`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolError {
    /// A job panicked on a pooled worker (region state is now suspect).
    JobPanicked,
    /// A helper thread could not be spawned (OOM-class).
    SpawnFailed,
}

/// Region handoff state (under the pool mutex).
///
/// CEP:WHAT: Single-occupancy region slot + done barrier.
/// CEP:WHY: One region runs at a time per pool (callers serialize); the
///          `done` counter is the lending barrier that keeps raw pointers
///          sound — see the safety argument on `RegionDesc`.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (guarded state).
/// CEP:ASSUMES: mutated only under the pool mutex.
/// CEP:COST: 32 bytes.
/// CEP:EVIDENCE: pool tests.
#[derive(Default)]
struct Handoff {
    /// The published region (raw lending descriptor).
    desc: Option<RegionDesc>,
    /// Monotonic region generation.
    generation: u64,
    /// Helpers checked in for the current generation.
    done: usize,
    /// A worker panicked during the current generation.
    panicked: bool,
    /// A region is published and not yet reaped.
    busy: bool,
}

/// The lending descriptor: one region's entry points.
///
/// CEP:WHAT: Trampoline fn + raw context pointer + generation.
/// CEP:WHY: Helpers are spawned once and cannot borrow region-scoped state
///          (deques, pending, chunks live on the lending caller's stack);
///          the descriptor carries the addresses instead.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: see the safety invariant below.
/// CEP:COST: 24 bytes.
/// CEP:EVIDENCE: pool tests + `pool_concurrent_callers_serialize`.
/// CEP:SECURITY: internal state only.
// CEP:UNSAFE | Safety: `ctx` is a raw pointer into the lending caller's
//             stack frame. Soundness invariant (the done-barrier argument):
//             (1) the caller publishes the descriptor and then runs worker
//             0 on the same frame; (2) every helper dereferences `ctx` only
//             inside the trampoline, which completes BEFORE the helper
//             re-locks and increments `done`; (3) the caller reaps the
//             region only after `done == helpers` (all trampolines for this
//             generation returned) and only then returns / drops the frame.
//             Therefore no dereference can race the frame's teardown, even
//             if the caller's own worker_loop panics (it is caught and the
//             barrier still runs). Concurrent regions serialize through the
//             handoff's `busy` flag, so a descriptor is never overwritten
//             while live. `Send` is thus sound for the mutex-guarded move
//             from publisher to helpers.
// CEP:UNSAFE | Safety: the Send impl moves raw pointers guarded by the
//             handoff mutex; soundness follows from the done-barrier
//             argument above (deref only while the lending frame lives).
unsafe impl Send for RegionDesc {}

struct RegionDesc {
    /// Monomorphized region entry: `(worker_index, ctx) -> saw_panic`.
    trampoline: unsafe fn(usize, *const ()) -> bool,
    /// Raw context (lending caller's frame).
    ctx: *const (),
}

impl Copy for RegionDesc {}

impl Clone for RegionDesc {
    fn clone(&self) -> Self {
        *self
    }
}

/// Spin budget before parking (adaptive back-to-back regions).
///
/// CEP:WHAT: Bounded PAUSE-spin iterations before a condvar park.
/// CEP:WHY: CEP-3's measured evidence (benches/anvil_bench: region_pooled
///          vs region_scoped) shows a futex sleep/wake cycle costs as much
///          as a thread spawn for micro-regions; bursty region phases
///          (saturation rounds, universe scoring) re-publish within
///          microseconds, so a bounded spin catches the next region warm.
///          ~8k PAUSE instructions ≈ 40-80 µs on current x86 cores — bounded
///          duty-cycle waste during long idle gaps (Law 4 calibrated).
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: bounded spin then park.
/// CEP:EVIDENCE: `region_pooled` vs `region_scoped` in the bench archive.
const SPIN_BUDGET: u32 = 8_192;

/// Shared pool state (the park).
struct PoolInner {
    /// Region handoff + barrier (authoritative state).
    lock: Mutex<Handoff>,
    /// Helper parking.
    parked: Condvar,
    /// Shutdown signal (Drop).
    shutdown: AtomicBool,
    /// Wakeup HINT: last published generation (set before notify; the
    /// mutex state remains authoritative — the hint only accelerates the
    /// adaptive spin).
    pub_generation: AtomicU64,
    /// Wakeup HINT: helpers checked in so far for the current region
    /// (authoritative count stays under the mutex).
    done_hint: AtomicUsize,
    /// Helper count (for the check-in completion notify).
    helpers: usize,
}

impl PoolInner {
    /// CEP:WHAT: Locks the handoff without unwrapping.
    /// CEP:WHY: unwrap is denied; poisoning cannot occur (no panic path
    ///          holds this mutex — trampolines run unlocked), and
    ///          into_inner() recovery keeps the pool sound regardless.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 lock.
    /// CEP:EVIDENCE: pool tests.
    fn guard(&self) -> std::sync::MutexGuard<'_, Handoff> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The persistent pool.
///
/// CEP:WHAT: N-1 parked helper threads + the caller as worker 0.
/// CEP:WHY: CEP-3: amortize region setup across the compilation session.
/// CEP:STATUS: complete
/// CEP:FAILURE: see PoolError / ExecError.
/// CEP:ASSUMES: one region at a time (callers serialize); nesting falls
///              back to scoped spawning via the thread-local in-region
///              flag (deadlock-free by construction).
/// CEP:COST: setup = helper spawns (once); region = 1 handoff.
/// CEP:EVIDENCE: pool tests.
/// CEP:SECURITY: first-party trampolines only.
/// CEP:HPC-DETERMINISM: same contracts as the scoped executor.
pub struct Pool {
    inner: Arc<PoolInner>,
    helpers: Vec<JoinHandle<()>>,
    helper_count: usize,
}

thread_local! {
    /// CEP:WHAT: Per-thread "inside a pooled region" flag.
    /// CEP:WHY: A region body that calls back into the pool would wait on
    ///           its own region's completion (self-deadlock). The flag
    ///           routes nested calls to the scoped fallback.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none.
    /// CEP:ASSUMES: set on every thread executing region bodies (caller +
    ///           helpers).
    /// CEP:COST: 1 TLS load per entry check.
    /// CEP:EVIDENCE: `pool_nested_partition_falls_back`.
    static IN_REGION: Cell<bool> = const { Cell::new(false) };
}

/// CEP:WHAT: True while this thread executes a pooled region body.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 TLS load
/// CEP:EVIDENCE: nested-fallback test
pub fn in_pooled_region() -> bool {
    IN_REGION.with(|c| c.get())
}

fn set_in_region(v: bool) {
    IN_REGION.with(|c| c.set(v));
}

/// CEP:WHAT: Sets the in-region TLS flag from the SCOPED executor's
///           spawned workers.
/// CEP:WHY: Audit F-2: a scoped fallback worker executing a region body
///           must route any nested pooled call to the scoped path too —
///           without the flag, a nested call from a scoped worker inside a
///          (pooled) enclosing region would wait on the enclosing region's
///          pool completion: a self-deadlock. Scoped workers are fresh
///          threads, so the flag must be set there explicitly.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: called in pairs (set true at body start, false at end).
/// CEP:COST: 1 TLS store.
/// CEP:EVIDENCE: reasoning over the routing condition; unreachable with
///           today's pure bodies (documented defensive closure of the
///           routing invariant).
pub(crate) fn set_in_region_scoped(v: bool) {
    IN_REGION.with(|c| c.set(v));
}

impl Pool {
    /// CEP:WHAT: Builds a pool with `helpers` persistent workers (plus the
    ///           caller = `helpers + 1` total).
    /// CEP:WHY: Worker topology is fixed per pool (passes declare it); the
    ///           bounded count honors MAX_WORKERS.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: TooManyWorkers / ZeroWorkers / SpawnFailed.
    /// CEP:ASSUMES: none.
    /// CEP:COST: helper spawns (once).
    /// CEP:EVIDENCE: pool tests.
    pub fn new(helpers: usize) -> Result<Pool, ExecError> {
        let total = helpers.checked_add(1).ok_or(ExecError::TooManyWorkers)?;
        if total > MAX_WORKERS {
            return Err(ExecError::TooManyWorkers);
        }
        let inner = Arc::new(PoolInner {
            lock: Mutex::new(Handoff::default()),
            parked: Condvar::new(),
            shutdown: AtomicBool::new(false),
            pub_generation: AtomicU64::new(0),
            done_hint: AtomicUsize::new(0),
            helpers,
        });
        let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(helpers);
        for h in 0..helpers {
            let inner2 = Arc::clone(&inner);
            let spawn = std::thread::Builder::new()
                .name(format!("anvil-pool-{h}"))
                .spawn(move || helper_loop(inner2, h + 1));
            match spawn {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    // Spawn failure: shut down what we started and report.
                    inner.shutdown.store(true, Ordering::Release);
                    inner.parked.notify_all();
                    for handle in handles.drain(..) {
                        let _ = handle.join();
                    }
                    return Err(ExecError::Pool(PoolError::SpawnFailed));
                }
            }
        }
        Ok(Pool {
            inner,
            helpers: handles,
            helper_count: helpers,
        })
    }

    /// CEP:WHAT: Builds a solo pool (caller-only) without spawning.
    /// CEP:WHY: The global-pool initializer cannot fail: on spawn failure
    ///          the degradation is caller-thread execution — slow but
    ///          correct and loud in the conformance catalog.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (no spawns).
    /// CEP:ASSUMES: none.
    /// CEP:COST: 1 allocation.
    /// CEP:EVIDENCE: global routing test.
    fn solo() -> Pool {
        Pool {
            inner: Arc::new(PoolInner {
                lock: Mutex::new(Handoff::default()),
                parked: Condvar::new(),
                shutdown: AtomicBool::new(false),
                pub_generation: AtomicU64::new(0),
                done_hint: AtomicUsize::new(0),
                helpers: 0,
            }),
            helpers: Vec::new(),
            helper_count: 0,
        }
    }

    /// CEP:WHAT: Builds a pool, degrading to solo on spawn failure.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: never (documented degradation).
    /// CEP:ASSUMES: none.
    /// CEP:COST: see new().
    /// CEP:EVIDENCE: global pool initializer.
    pub fn new_or_solo(helpers: usize) -> Pool {
        match Pool::new(helpers) {
            Ok(p) => p,
            Err(_) => Pool::solo(),
        }
    }

    /// CEP:WHAT: Total workers (helpers + the caller).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: routing tests
    pub fn workers(&self) -> usize {
        self.helper_count + 1
    }

    /// CEP:WHAT: Publishes one region and returns its generation.
    /// CEP:WHY: The lending half of the pool: descriptor under the mutex,
    ///          helpers woken, caller proceeds to worker 0.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: blocks while another region is busy (serialization).
    /// CEP:ASSUMES: caller keeps the ctx frame alive until wait_region.
    /// CEP:COST: 1 lock + 1 notify.
    /// CEP:EVIDENCE: pool tests.
    fn publish(&self, trampoline: unsafe fn(usize, *const ()) -> bool, ctx: *const ()) -> u64 {
        let mut guard = self.inner.guard();
        while guard.busy {
            guard = self
                .inner
                .parked
                .wait(guard)
                .unwrap_or_else(|e| e.into_inner());
        }
        guard.generation += 1;
        guard.busy = true;
        guard.done = 0;
        guard.panicked = false;
        guard.desc = Some(RegionDesc { trampoline, ctx });
        let gen = guard.generation;
        drop(guard);
        // Wakeup hint FIRST (spinners re-check the authoritative mutex
        // state before proceeding), then the condvar broadcast.
        self.inner.pub_generation.store(gen, Ordering::Release);
        self.inner.done_hint.store(0, Ordering::Release);
        self.inner.parked.notify_all();
        gen
    }

    /// CEP:WHAT: Waits until every helper checked in, then reaps the region.
    /// CEP:WHY: The done-barrier — the soundness anchor of the lending
    ///          discipline (see RegionDesc) and the panic report point.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: JobPanicked if any worker panicked (loud, no hang).
    /// CEP:ASSUMES: publish() returned this generation on this thread.
    /// CEP:COST: 1 lock + waits.
    /// CEP:EVIDENCE: pool tests + panic test.
    fn wait_region(&self, gen: u64) -> Result<(), ExecError> {
        // Adaptive spin on the done hint (barrier accelerator; the mutex
        // state below remains authoritative).
        let mut spin = 0u32;
        while spin < SPIN_BUDGET && self.inner.done_hint.load(Ordering::Acquire) < self.helper_count
        {
            spin += 1;
            core::hint::spin_loop();
        }
        let mut guard = self.inner.guard();
        while guard.busy && guard.generation == gen && guard.done < self.helper_count {
            guard = self
                .inner
                .parked
                .wait(guard)
                .unwrap_or_else(|e| e.into_inner());
        }
        let panicked = guard.panicked;
        if guard.busy && guard.generation == gen && guard.done >= self.helper_count {
            guard.busy = false;
            guard.desc = None;
            guard.done = 0;
            guard.panicked = false;
            // Wake publishers waiting for the region slot (the condvar is
            // shared by helpers and publishers; without this notify a
            // serialized caller sleeps until some unrelated future notify).
            self.inner.parked.notify_all();
        }
        if panicked {
            return Err(ExecError::Pool(PoolError::JobPanicked));
        }
        Ok(())
    }

    /// CEP:WHAT: Gear 1 through the pool — strided static partitioning.
    /// CEP:WHY: Identical semantics to the scoped variant: worker w owns
    ///           chunks w, w+n, ... — disjoint slices, ZERO atomics inside
    ///           `f` (the discipline of arch section 2, Gear 1). The
    ///           chunk stride is deterministic, so outputs are
    ///           scheduling-independent.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: LengthMismatch; PoolError::JobPanicked (defensive).
    /// CEP:ASSUMES: `f` deterministic and pure w.r.t. its element.
    /// CEP:COST: region handoff + body cost.
    /// CEP:EVIDENCE: `pool_partition_is_exact`, `pool_partition_matches_scoped`.
    /// CEP:HPC-DETERMINISM: outputs[i] = f(inputs[i]) — order-independent.
    pub fn run_partitioned<T, R, F>(
        &self,
        inputs: &[T],
        outputs: &mut [R],
        f: F,
    ) -> Result<(), ExecError>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync,
    {
        if inputs.len() != outputs.len() {
            return Err(ExecError::LengthMismatch);
        }
        let n = self.workers();
        let chunk = inputs.len().div_ceil(n);
        let ctx = PartitionCtx {
            inputs_ptr: inputs.as_ptr(),
            len: inputs.len(),
            outputs_ptr: outputs.as_mut_ptr(),
            f_ptr: &f as *const F,
            n_workers: n,
            chunk,
        };
        let gen = self.publish(
            partition_trampoline::<T, R, F>,
            &ctx as *const _ as *const (),
        );
        set_in_region(true);
        // The caller is worker 0; the trampoline catches task panics
        // internally (partition) or via worker_loop's credit-safe catch
        // (fork/join), so no unwind can escape past the barrier below.
        // CEP:UNSAFE | Safety: caller-side half of the lending argument — all
        //             pointers target THIS frame (`inputs`, `outputs`, `f`,
        //             `ctx`), which lives until wait_region below.
        let caller_panicked =
            unsafe { partition_trampoline::<T, R, F>(0, &ctx as *const _ as *const ()) };
        set_in_region(false);
        self.wait_region(gen)?;
        if caller_panicked {
            return Err(ExecError::Pool(PoolError::JobPanicked));
        }
        Ok(())
    }

    /// CEP:WHAT: Gear 2 through the pool — fork/join with stealing.
    /// CEP:WHY: Same worker_loop as the scoped executor (LIFO pop, FIFO
    ///           steal, pending-credit termination); only the thread
    ///           sourcing differs (parked helpers vs fresh spawns).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: DequeFull on the root push; JobPanicked (defensive).
    /// CEP:ASSUMES: jobs communicate via deterministic reduction state.
    /// CEP:COST: region handoff + dispatch.
    /// CEP:EVIDENCE: `pool_fork_join_fib_tree`, `pool_regions_reuse_workers`.
    /// CEP:HPC-DETERMINISM: nondeterministic interleaving, deterministic
    ///           results (pure jobs + deterministic reduction).
    pub fn run_fork_join<J: crate::executor::Job>(&self, root: J) -> Result<(), ExecError> {
        let n = self.workers();
        let mut deques: Vec<CachePadded<Deque<J>>> = Vec::with_capacity(n);
        for _ in 0..n {
            deques.push(CachePadded::new(*Deque::new(DEQUE_CAPACITY)));
        }
        let pending = AtomicIsize::new(1);
        if deques[0].value.push(root).is_err() {
            return Err(ExecError::DequeFull);
        }
        let ctx = ForkJoinCtx {
            table_ptr: deques.as_ptr(),
            table_len: n,
            pending_ptr: &pending as *const AtomicIsize,
        };
        let gen = self.publish(fork_join_trampoline::<J>, &ctx as *const _ as *const ());
        set_in_region(true);
        // Caller is worker 0: worker_loop's internal catch releases the
        // pending credit on the panic path, so quiescence always arrives.
        // CEP:UNSAFE | Safety: caller-side lending — `deques`, `pending` and
        //             `ctx` live in THIS frame until wait_region below; the
        //             slice reconstruction uses the exact published length.
        let caller_panicked = unsafe {
            let table: &[CachePadded<Deque<J>>] =
                core::slice::from_raw_parts(ctx.table_ptr, ctx.table_len);
            worker_loop(0, &table[0].value, table, &pending)
        };
        set_in_region(false);
        self.wait_region(gen)?;
        if caller_panicked {
            return Err(ExecError::Pool(PoolError::JobPanicked));
        }
        Ok(())
    }
}

impl Drop for Pool {
    /// CEP:WHAT: Wakes the helpers, signals shutdown, joins them.
    /// CEP:WHY: No region can be in flight (regions hold &self; Drop needs
    ///           exclusive access — the borrow system enforces the
    ///           ordering), so a notify + join is sufficient and cannot
    ///           hang on region work.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: join errors ignored (thread already gone).
    /// CEP:ASSUMES: no concurrent region.
    /// CEP:COST: N joins (once).
    /// CEP:EVIDENCE: pool tests drop pools routinely.
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.parked.notify_all();
        for handle in self.helpers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// CEP:WHAT: One helper's park-run-checkin loop.
/// CEP:WHY: The persistent half: wait for a fresh generation, run the
///          region trampoline OUTSIDE the lock (region bodies may lock),
///          check in under the lock, repeat. The catch_unwind discipline
///          keeps a broken panic contract from deadlocking the barrier.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (panics are caught and reported via the barrier).
/// CEP:ASSUMES: helpers are first-party code.
/// CEP:COST: park wakeup per region + body.
/// CEP:EVIDENCE: all pool tests.
fn helper_loop(inner: Arc<PoolInner>, helper_index: usize) {
    let mut seen: u64 = 0;
    loop {
        // Adaptive spin: back-to-back regions re-publish within
        // microseconds; parking first would pay a full futex cycle
        // (measured in benches/anvil_bench — see SPIN_BUDGET).
        let mut spin = 0u32;
        while spin < SPIN_BUDGET {
            if inner.shutdown.load(Ordering::Acquire) {
                return;
            }
            if inner.pub_generation.load(Ordering::Acquire) > seen {
                break;
            }
            spin += 1;
            core::hint::spin_loop();
        }
        let mut guard = inner.guard();
        // Wait for a fresh region or shutdown.
        let my_gen: u64;
        loop {
            if inner.shutdown.load(Ordering::Acquire) {
                return;
            }
            if guard.busy && guard.generation > seen {
                my_gen = guard.generation;
                break;
            }
            guard = inner.parked.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        let (trampoline, ctx) = match guard.desc {
            Some(d) => (d.trampoline, d.ctx),
            None => {
                // Busy without a descriptor cannot happen (busy is set with
                // the descriptor); defensive re-park.
                continue;
            }
        };
        // Run the region body unlocked.
        drop(guard);
        set_in_region(true);
        // Combine BOTH panic signals (audit-remediation regression): the
        // trampoline's return value carries panics caught INSIDE
        // worker_loop/partition_trampoline (no unwind), while the outer
        // catch guards the trampoline scaffolding itself.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // CEP:UNSAFE | Safety: helper-side lending — the done-barrier
            //             argument on RegionDesc proves `ctx` is valid here:
            //             the lending caller cannot reap the region (and its
            //             frame) until THIS helper checks in below.
            unsafe { trampoline(helper_index, ctx) }
        }));
        let panicked = outcome.unwrap_or(true);
        set_in_region(false);
        // Check in.
        inner.done_hint.fetch_add(1, Ordering::Release);
        let mut guard = inner.guard();
        seen = my_gen;
        guard.done += 1;
        if panicked {
            guard.panicked = true;
        }
        if guard.done >= inner.helpers {
            // The LAST helper's check-in completes the barrier: wake the
            // caller's wait_region (audit F-9: notify exactly once per
            // region, not on every check-in).
            inner.parked.notify_all();
        }
    }
}

/// Gear-1 lending context (caller frame).
struct PartitionCtx<T, R, F> {
    inputs_ptr: *const T,
    len: usize,
    outputs_ptr: *mut R,
    f_ptr: *const F,
    n_workers: usize,
    chunk: usize,
}

/// CEP:WHAT: Gear-1 trampoline — worker w processes chunks w, w+n, ...
/// CEP:WHY: Strided static assignment: disjoint slices with zero atomics
///          inside the task body (the Gear-1 letter); results are
///          scheduling-independent.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (bounds-guarded).
/// CEP:ASSUMES: the done-barrier lending argument holds.
/// CEP:COST: O(owned chunk elements).
/// CEP:EVIDENCE: `pool_partition_is_exact`.
/// CEP:HPC-DETERMINISM: deterministic per-element outputs.
unsafe fn partition_trampoline<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(
    worker: usize,
    ctx_ptr: *const (),
) -> bool {
    // CEP:UNSAFE | Safety: `ctx` points into the lending caller's frame; the
    //             done-barrier (publish -> trampoline -> check-in -> reap)
    //             proves the frame outlives every dereference. Slices are
    //             reconstructed with exact bounds from `len`/`chunk`.
    let (inputs, f, n, chunk, len, outputs_ptr) = unsafe {
        let ctx = &*(ctx_ptr as *const PartitionCtx<T, R, F>);
        let inputs: &[T] = core::slice::from_raw_parts(ctx.inputs_ptr, ctx.len);
        let f = &*ctx.f_ptr;
        (
            inputs,
            f,
            ctx.n_workers,
            ctx.chunk,
            ctx.len,
            ctx.outputs_ptr,
        )
    };
    let mut saw_panic = false;
    let mut c = worker;
    loop {
        let start = c * chunk;
        if start >= len {
            break;
        }
        let end = (start + chunk).min(len);
        let in_chunk = &inputs[start..end];
        // CEP:UNSAFE | Safety: output slice for chunk c — disjoint from every
        //             other worker's slice (stride ownership), and valid for
        //             the caller's frame lifetime (barrier argument).
        let out_chunk: &mut [R] =
            unsafe { core::slice::from_raw_parts_mut(outputs_ptr.add(start), end - start) };
        // Per-chunk panic catch: a panicking element skips the rest of THIS
        // chunk only; other chunks (this worker's and others') complete; the
        // panic is reported through the return value (loud, no unwind).
        let body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for (i, item) in in_chunk.iter().enumerate() {
                out_chunk[i] = f(item);
            }
        }));
        if body.is_err() {
            saw_panic = true;
        }
        c += n;
    }
    saw_panic
}

/// Gear-2 lending context (caller frame).
struct ForkJoinCtx<J: crate::executor::Job> {
    table_ptr: *const CachePadded<Deque<J>>,
    table_len: usize,
    pending_ptr: *const AtomicIsize,
}

/// CEP:WHAT: Gear-2 trampoline — reconstructs the deque table + pending and
///           enters the executor's worker_loop for this worker index.
/// CEP:WHY: The identical dispatch loop as the scoped executor: LIFO own
///           pops, FIFO steals, pending-credit termination. Only the
///           thread sourcing differs.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (worker_loop is total).
/// CEP:ASSUMES: the done-barrier lending argument holds.
/// CEP:COST: dispatch cost of worker_loop.
/// CEP:EVIDENCE: `pool_fork_join_fib_tree`, `pool_regions_reuse_workers`.
unsafe fn fork_join_trampoline<J: crate::executor::Job>(worker: usize, ctx_ptr: *const ()) -> bool {
    // CEP:UNSAFE | Safety: `ctx` points into the lending caller's frame
    //             (deques + pending + ctx itself); the done-barrier proves
    //             the frame outlives this call. The slice reconstruction
    //             uses the exact length published in the context.
    let (table, pending) = unsafe {
        let ctx = &*(ctx_ptr as *const ForkJoinCtx<J>);
        let table: &[CachePadded<Deque<J>>] =
            core::slice::from_raw_parts(ctx.table_ptr, ctx.table_len);
        let pending = &*ctx.pending_ptr;
        (table, pending)
    };
    worker_loop(worker, &table[worker].value, table, pending)
}

// ---------------------------------------------------------------------------
// Global pool + free-function routing
// ---------------------------------------------------------------------------

static GLOBAL_POOL: OnceLock<Pool> = OnceLock::new();

/// CEP:WHAT: The process-global pool (default topology, never dropped).
/// CEP:WHY: Free-function callers (run_partitioned / run_fork_join) route
///          here when their worker count matches, so production passes get
///          the pool without API changes. The pool intentionally outlives
///          every region (static lifetime): its threads park forever and
///          the process reaps them at exit — no drop-order hazard, no
///          join-on-exit race.
/// CEP:STATUS: complete
/// CEP:FAILURE: degrades to a solo pool on spawn failure (documented).
/// CEP:ASSUMES: none.
/// CEP:COST: one init.
/// CEP:EVIDENCE: routing tests.
pub fn global_pool() -> &'static Pool {
    // helpers = default - 1 so that workers() == default_worker_count():
    // production callers (egraph saturation, fusion scoring) pass exactly
    // that count and therefore route through the pool.
    GLOBAL_POOL.get_or_init(|| {
        Pool::new_or_solo(crate::executor::default_worker_count().saturating_sub(1))
    })
}

/// CEP:WHAT: True when `num_workers` matches the global pool topology (the
///           routing condition of the free functions).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 compare
/// CEP:EVIDENCE: routing tests
fn global_topology_matches(num_workers: usize) -> bool {
    global_pool().workers() == num_workers
}

/// CEP:WHAT: Gear 1 via the global pool when the topology matches.
/// CEP:WHY: Production callers pass default_worker_count(); routing them
///          through the persistent pool cuts per-region spawn cost (CEP-3)
///          without touching their code. Nested calls (from inside a
///          region body) and mismatched topologies fall back to the scoped
///          executor — deadlock-free by construction.
/// CEP:STATUS: complete
/// CEP:FAILURE: see Pool::run_partitioned / run_partitioned_scoped.
/// CEP:ASSUMES: f deterministic and pure w.r.t. its element.
/// CEP:COST: pooled handoff or scoped spawns.
/// CEP:EVIDENCE: `pool_partition_matches_scoped`, `pool_nested_partition
///           _falls_back`.
/// CEP:HPC-DETERMINISM: identical outputs on both paths.
pub fn run_partitioned_pooled<T, R, F>(
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
    if !in_pooled_region() && global_topology_matches(num_workers) {
        global_pool().run_partitioned(inputs, outputs, f)
    } else {
        crate::executor::run_partitioned_scoped(inputs, outputs, num_workers, f)
    }
}

/// CEP:WHAT: Gear 2 via the global pool when the topology matches.
/// CEP:WHY: Same routing discipline as run_partitioned_pooled.
/// CEP:STATUS: complete
/// CEP:FAILURE: see Pool::run_fork_join / run_fork_join_scoped.
/// CEP:ASSUMES: see the Job trait.
/// CEP:COST: pooled handoff or scoped spawns.
/// CEP:EVIDENCE: pool tests.
pub fn run_fork_join_pooled<J: crate::executor::Job>(
    root: J,
    num_workers: usize,
) -> Result<(), ExecError> {
    if !in_pooled_region() && global_topology_matches(num_workers) {
        global_pool().run_fork_join(root)
    } else {
        crate::executor::run_fork_join_scoped(root, num_workers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{Job, WorkerCtx};

    fn fib_pool(n: u64) -> u64 {
        struct Fib(u64);
        impl Job for Fib {
            fn run(&mut self, ctx: &mut WorkerCtx<'_, '_, Self>) {
                if self.0 <= 1 {
                    return;
                }
                // Reduction state is the shared sum (deterministic: adds are
                // commutative over u64 wrapping arithmetic).
                static SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if self.0 < 10 {
                    SUM.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                let a = Fib(self.0 - 1);
                let b = Fib(self.0 - 2);
                let _ = ctx.spawn(a);
                let _ = ctx.spawn(b);
            }
        }
        let pool = Pool::new(3);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            assert!(p.run_fork_join(Fib(n)).is_ok());
        }
        0
    }

    // CEP:WHAT: Pooled Gear-1 partitioning is exact and complete.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on slice drift.
    // CEP:ASSUMES: 3 helpers.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_partition_is_exact() {
        let inputs: Vec<i64> = (0..100).map(|i| i as i64 * 3).collect();
        let mut outputs: Vec<i64> = vec![0; inputs.len()];
        let pool = Pool::new(3);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            let r = p.run_partitioned(&inputs, &mut outputs, |x| x.wrapping_mul(2));
            assert!(r.is_ok());
            for (i, o) in outputs.iter().enumerate() {
                assert_eq!(*o, inputs[i].wrapping_mul(2));
            }
        }
    }

    // CEP:WHAT: Pooled and scoped partitioning produce IDENTICAL outputs
    //           (the routing switch can never change results).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on divergence.
    // CEP:ASSUMES: 3 helpers.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_partition_matches_scoped() {
        let inputs: Vec<u64> = (0..997).map(|i| (i * 2_654_435_761) % 1_000_003).collect();
        let mut a: Vec<u64> = vec![0; inputs.len()];
        let mut b: Vec<u64> = vec![0; inputs.len()];
        let pool = Pool::new(3);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            assert!(p
                .run_partitioned(&inputs, &mut a, |x| x.wrapping_mul(*x))
                .is_ok());
        }
        assert!(
            crate::executor::run_partitioned_scoped(&inputs, &mut b, 4, |x| { x.wrapping_mul(*x) })
                .is_ok()
        );
        assert_eq!(a, b);
    }

    // CEP:WHAT: Pooled Gear-2 runs a fib tree to quiescence.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the pending credit never zeroes.
    // CEP:ASSUMES: 3 helpers.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_fork_join_fib_tree() {
        let _ = fib_pool(14);
        // Reaching here means quiescence (no hang, no error).
    }

    // CEP:WHAT: Sequential regions through ONE pool stay isolated and
    //           deterministic (worker reuse across regions).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on cross-region state drift.
    // CEP:ASSUMES: 2 helpers.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_regions_reuse_workers() {
        let pool = Pool::new(2);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            for round in 0..50u64 {
                let inputs: Vec<u64> = (0..64).map(|i| i + round).collect();
                let mut outputs: Vec<u64> = vec![0; 64];
                assert!(p
                    .run_partitioned(&inputs, &mut outputs, |x| x + round)
                    .is_ok());
                for (i, o) in outputs.iter().enumerate() {
                    assert_eq!(*o, (i as u64) + round * 2);
                }
            }
        }
    }

    // CEP:WHAT: A partition body that calls run_partitioned_pooled again
    //           (nested) falls back to scoped spawning — no self-deadlock.
    // CEP:STATUS: complete
    // CEP:FAILURE: the test hangs if nesting deadlocks.
    // CEP:ASSUMES: body runs on a region thread (flag set).
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_nested_partition_falls_back() {
        // The outer region body IS `f`, which itself routes a nested call.
        let inputs: Vec<u64> = (0..32).collect();
        let mut outputs: Vec<u64> = vec![0; 32];
        let f = |x: &u64| -> u64 {
            // Inner call happens while this thread is flagged in-region only
            // when executed BY the pool; the outer call below routes through
            // the pool, so workers land here flagged -> inner call scopes.
            let mut inner_out = vec![0u64; 3];
            let inner_in = [*x, x + 1, x + 2];
            let r =
                run_partitioned_pooled(&inner_in, &mut inner_out, global_pool().workers(), |v| {
                    v.wrapping_mul(7)
                });
            assert!(r.is_ok());
            inner_out[0] + inner_out[1] + inner_out[2]
        };
        // Route the outer call through the pool explicitly.
        let pool = Pool::new(2);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            assert!(p.run_partitioned(&inputs, &mut outputs, f).is_ok());
            for (i, o) in outputs.iter().enumerate() {
                let x = i as u64;
                assert_eq!(*o, x * 7 + (x + 1) * 7 + (x + 2) * 7);
            }
        }
    }

    // CEP:WHAT: Concurrent callers serialize through the handoff (no lost
    //           regions, no deadlock; all results correct).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on lost or wrong regions.
    // CEP:ASSUMES: 2 helpers.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_concurrent_callers_serialize() {
        let pool = Arc::new(Pool::new(2).ok().unwrap_or_else(Pool::solo));
        let pool2 = Arc::clone(&pool);
        let handle = std::thread::spawn(move || {
            let inputs: Vec<u64> = (0..100).collect();
            let mut outputs = vec![0u64; 100];
            let r = pool2.run_partitioned(&inputs, &mut outputs, |x| x.wrapping_mul(11));
            (r.is_ok(), outputs)
        });
        let inputs: Vec<u64> = (0..100).collect();
        let mut outputs = vec![0u64; 100];
        let r0 = pool.run_partitioned(&inputs, &mut outputs, |x| x.wrapping_mul(13));
        let joined = handle.join();
        assert!(r0.is_ok());
        assert!(joined.is_ok());
        if let Ok((ok, outs)) = joined {
            assert!(ok);
            for (i, o) in outs.iter().enumerate() {
                assert_eq!(*o, (i as u64).wrapping_mul(11));
            }
        }
        for (i, o) in outputs.iter().enumerate() {
            assert_eq!(*o, (i as u64).wrapping_mul(13));
        }
    }

    // CEP:WHAT: A panicking job is reported loudly (JobPanicked) — never a
    //           barrier deadlock. The panic is induced by arithmetic
    //           overflow (overflow-checks are on in ALL profiles; this is
    //           the only lint-clean panic escape in this workspace).
    // CEP:STATUS: complete
    // CEP:FAILURE: the test hangs if the barrier deadlocks; assert fires
    //               if the panic is swallowed.
    // CEP:ASSUMES: helpers catch the unwind.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pool_panicking_job_reported_loudly() {
        struct Boom;
        impl Job for Boom {
            fn run(&mut self, _ctx: &mut WorkerCtx<'_, '_, Self>) {
                // Overflow panic (checked in all profiles): a genuine,
                // lint-clean runtime panic. black_box blocks const-eval so
                // the overflow fires at RUNTIME, not at compile time.
                let big = std::hint::black_box(i64::MAX);
                let _detonate = big + 1;
            }
        }
        let pool = Pool::new(2);
        assert!(pool.is_ok());
        if let Ok(p) = pool {
            let r = p.run_fork_join(Boom);
            assert_eq!(
                r.err(),
                Some(ExecError::Pool(PoolError::JobPanicked)),
                "panic must surface as JobPanicked, got {r:?}"
            );
            // The pool stays usable after the report.
            let inputs: Vec<u64> = (0..10).collect();
            let mut outputs = vec![0u64; 10];
            assert!(p.run_partitioned(&inputs, &mut outputs, |x| x + 1).is_ok());
            assert_eq!(outputs[9], 10);
        }
    }

    // CEP:WHAT: The global-pool routing functions work on both paths
    //           (matching topology -> pool; mismatched -> scoped).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on routing breakage.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn global_routing_works_on_both_paths() {
        let inputs: Vec<u64> = (0..50).collect();
        let mut a = vec![0u64; 50];
        let mut b = vec![0u64; 50];
        // Matching topology (routes through the global pool).
        let n = global_pool().workers();
        assert!(run_partitioned_pooled(&inputs, &mut a, n, |x| x + 5).is_ok());
        // Mismatched topology (scoped fallback).
        let mismatched = if n == 2 { 3 } else { 2 };
        assert!(run_partitioned_pooled(&inputs, &mut b, mismatched, |x| x + 5).is_ok());
        assert_eq!(a, b);
    }
}
