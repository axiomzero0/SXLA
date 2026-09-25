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
| Saturation (Gear-1 local rules) | complete | saturate tests |
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
| graph.if regions in text v1 | partial (flat functions only) | CEP-12 |
| Persistent worker pool (vs scoped regions) | partial | CEP-3 |
| NUMA-aware allocation | placeholder | CEP-2 |
| E-graph cross-worker SPSC merges | partial (Gear-1 local rules now) | CEP-17 |
| Extraction application to snapshots | partial (analysis + telemetry) | CEP-17 |
| Presburger/ISL legality | partial (affine subset) | CEP-19 |
| Tier-2 ML ranker | placeholder (Tier-1 fallback, reported) | CEP-20 |
| Tier-3 autotuning | stub (loud failure) | CEP-21 |
| Tier-3 PGO | placeholder (loud failure) | CEP-25 |
| Speculative compilation | placeholder | CEP-25 |
| Software pipelining | partial (stage ops exist) | CEP-23 |
| Shared-memory promotion | partial (budgets + spaces) | CEP-24 |
| GPU targets | placeholder (CPU only, loud) | CEP-16 |
| Strided/block layouts | placeholder | CEP-7 |
| Conv interpreter kernel | stub (UnsupportedInstr, loud) | CEP-26 |
| CI sanitizer matrix | partial (workflow present; nightly TSan gated) | CEP-27 |

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
