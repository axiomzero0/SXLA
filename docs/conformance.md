# SXLA Conformance Status Catalog (CEP&CC 0.1)

Honest status per CEP&CC 10.5. `complete` = implemented, tested, documented.
`partial` = implemented for a documented subset with a ticketed gap.
`stub`/`placeholder` = loud, minimal behavior.

## Completed subsystems

| Subsystem | Status | Evidence |
|-----------|--------|----------|
| Anvil Gear 1 (static partition) | complete | `anvil::executor` tests + gear-1 consumers |
| Anvil Gear 2 (fork/join + stealing) | complete | `gear2_*` tests, chase_lev stress (100k tasks, 16 threads) |
| Anvil Gear 3 (SPSC + batching) | complete | spsc stress (1M elements), telemetry bus |
| Anvil Gear 4 (EBR + sharded map) | complete | ebr epoch tests, 8-reader stress |
| Bounded bump arenas | complete | bump tests |
| Packed ids + generations | complete | id tests |
| Deterministic FNV hashing | complete | reference vectors |
| IR verifier (38.18 checks) | complete | verifier tests |
| GVN/CSE with dominance legality | complete | gvn tests |
| Constant folding (int-exact, FP-exact) | complete | fold tests |
| DCE with root anchoring | complete | dce tests |
| Scheduler (deterministic Kahn) | complete | schedule tests |
| Text IR v1 round-trip | complete | text tests |
| Transactional snapshot commits | complete | snapshot tests |
| Pass manager + manifests (38.20) | complete | passman tests |
| Layout inference | complete | level1 tests |
| Cluster hypergraph | complete | level2 tests |
| Level-3 projection + bufferization | complete | level3/codegen tests |
| CPU target lowering + interpreter | complete | level4/runtime tests, cli tests |
| E-graph union-find (order-independent) | complete | union_find/egraph tests |
| Rewrite legality gates (38.24 float ban) | complete | rules tests |
| Fusion-aware extraction penalties | complete | extract tests |
| Saturation (Gear-1 local rules, pooled) | complete | saturate tests |
| E-graph extraction application (CEP-17) | complete | apply tests + jit tier-2 differential |
| Persistent worker pool (CEP-3) | complete | pool tests; bench 1.7us vs 53.7us region setup |
| If-region text round-trip (CEP-12) | complete | text/verifier if-region tests |
| Conv reference kernel (CEP-26) | complete | conv_valid/same/stride/multichannel tests |
| Fusion legality (affine subset) | partial | legality tests; full Presburger = CEP-19 |
| Resource model + repair | complete | resource/repair tests |
| Tiered cost model | partial | Tier 1 complete; Tier 2 placeholder; Tier 3 loud stub (CEP-20/21) |
| Universe search + atomic pruning | partial | search tests; recursive tree granularity = CEP-22 |
| JIT cache (EBR-sharded) | complete | cache tests |
| SPSC JIT boundary | complete | boundary tests (incl. cross-thread) |
| Tier manifests 0/1/2 | complete | driver tests + differential tier test |
| Benchmarks (Law 4 evidence) | complete | benches/anvil_bench |

## Partial / stub / placeholder items (with tickets)

| Item | Status | Ticket |
|------|--------|--------|
| NUMA-aware allocation | placeholder | CEP-2 |
| E-graph cross-worker SPSC merges | partial (Gear-1 pooled local rules now) | CEP-17 |
| Presburger/ISL legality | partial (affine subset) | CEP-19 |
| Tier-2 ML ranker | placeholder (Tier-1 fallback, reported) | CEP-20 |
| Tier-3 autotuning | stub (loud failure) | CEP-21 |
| Tier-3 PGO | placeholder (loud failure) | CEP-25 |
| Speculative compilation | placeholder | CEP-25 |
| Software pipelining | partial (stage ops exist) | CEP-23 |
| Shared-memory promotion | partial (budgets + spaces) | CEP-24 |
| GPU targets | placeholder (CPU only, loud) | CEP-16 |
| Strided/block layouts | placeholder | CEP-7 |
| Conv interpreter kernel | complete (naive NCHW/FCHW reference; valid/same/stride) | closed |
| CI sanitizer matrix | partial (workflow present; nightly TSan gated) | CEP-27 |

## Second audit round (session 2) and remediation

A second standards audit over the four new features (egraph apply/lift,
pool, text if-regions, conv kernel, driver) found 0 S0, 3 S1, 10 S2. All
were remediated with regression tests:

- **S1-1 (panic)**: `run_partitioned_scoped` panicked on empty inputs
  (`chunks(0)`); both routing paths now no-op (`gear1_empty_inputs_are_noop`).
- **S1-2 (deadlock hole)**: scoped-fallback workers did not carry the
  in-region TLS flag, so nested pooled calls from a scoped worker inside a
  pooled enclosing region could self-deadlock; the flag is now set on every
  scoped worker (`set_in_region_scoped`).
- **S1-3 (swallowed failure + dead stage)**: Tier-2's fusion search
  discarded its Result and fed nothing downstream; the error now propagates
  (`JitError::Pipeline("fusion-search")`) and the docs state the ClusterSet
  does not yet feed structurize (CEP-22).
- S2s: float bits escape form for non-finite/oversized constants
  (`float_bits_roundtrip`); EGraph::merge canonicalizes both arguments;
  provable folds fire when the folded const dedups onto an existing arena
  constant (`apply_folds_when_const_dedups`); conv Valid rejects oversized
  kernels (`conv_valid_oversized_kernel_rejected`); degraded if-form
  printing keeps body nodes; check-in notifies exactly once per region;
  stale executor header corrected; roots = last ROOT-region node
  (`if_program_fails_loudly_at_lowering`); text-level dominance rejection
  moved to the integration crate (`if_dominance_violation_rejected`);
  parser/print nesting bounded at MAX_NEST (`rejects_runaway_nesting`).
- Remediation itself found and fixed a flag-loss bug: the helper-side
  job-panic signal was discarded when `run_caught` swallowed the
  trampoline's saw-panic return value (flaky `pool_panicking_job_reported
  _loudly`); both panic channels now combine.

The pool's lending discipline (the audit's focus area A) was certified
airtight across publish/trampoline/check-in/reap interleavings, panic
paths, Drop ordering, and hint/authority races.

## Waivers

See [.cep/waivers.md](../.cep/waivers.md) — all temporary, owner-assigned,
review-dated (CEP&CC 34.7).

## Independent audit and remediation

A full standards audit (task 16) against CEP&CC 0.1 found 19 issues (3×S0,
8×S1, 8×S2). All were remediated with regression tests:

- **S0-1 (memory safety)**: EBR pin/collect ordering strengthened to the
  crossbeam-epoch SeqCst discipline; `concurrent_pin_retire_stress` covers
  the in-flight-pin/double-advance race.
- **S0-2 (miscompilation)**: constant folding now computes i64 arithmetic in
  checked i64 (never through f64); `large_integer_fold_is_exact` pins
  exactness above 2^53.
- **S0-3 (false optimization claim)**: the JIT driver now consumes the
  optimized snapshot end-to-end; the tier differential test proves Tier 1/2
  actually shrink foldable programs while preserving executed values.
  Wiring this exposed and fixed a slot-index sizing bug (live count vs slot
  space) across seven crates.
