# SXLA Architecture Notes

This file maps the implemented code to the master architecture document.
Module-level CEP:WHY fields carry the full reasoning at each site.

## Anvil gears

| Gear | Where | Contract |
|------|-------|----------|
| 1 — Static partitioning | `anvil::executor::run_partitioned` (used by e-graph local rules and fusion universe scoring) | Disjoint slices, zero atomics inside task bodies, zero stealing |
| 2 — Fork/join + work stealing | `anvil::executor::run_fork_join` (+ `anvil::pool::Pool`, CEP-3) + `chase_lev::Deque` | LIFO own-deque pops, FIFO steals, pending-credit termination; persistent park-based pool with per-region lending (bench: 1.7us vs 53.7us region setup) |
| Task fuel | `anvil::fuel::FuelMeter` | Deterministic abstract work units (64/branch) replacing the wall-clock 500ns rule — time inputs would violate HPC determinism (CEP&CC 38.10) |
| 3 — SPSC pipelines | `anvil::spsc::SpscRing` + `telemetry::TelemetryBus` + `jit::boundary` | Acquire/Release pairs, batched drains |
| 4 — EBR | `anvil::ebr::Collector` + `sharded::ShardedMap` + `jit::cache` | 2-atomic pin/unpin; lock-free reads; deferred frees |

## XIR levels

| Level | Module | Notes |
|-------|--------|-------|
| 0 graph | `xir-graph` (sea-of-nodes: GVN, fold, DCE, dominance, verifier, scheduler) | Token edges for effects |
| 1 tensor | `xir-levels::level1` (layout inference) + tensor ops in `xir-core::op` | Layouts first-class |
| 2 fusion | `xir-levels::level2` (ClusterSet) + `fusion` (search) | Hypergraph of clusters |
| 3 loop | `xir-levels::level3` (LoopProgram, bufferization records) | Structured projection |
| 4 target | `xir-levels::level4` (CPU TargetProgram) + `codegen` | target.mma → Matmul |

## Pass manager

`xir-levels::passman` implements the architecture's Pass trait
(`required_form`, `concurrency`, `run`) extended with HPC class + version +
telemetry ids. The manager verifies entry IR (CEP&CC 38.18), verifies after
every HPC-0 pass, and emits PassStart/PassEnd into the lock-free telemetry bus.

## Fusion search

`fusion::search` runs universes A (aggressive epilogue), B (split reductions),
C (rematerialize) as pure scoring functions under Gear-1 partitioning; a
global `AtomicU64` best-cost implements the atomic pruning protocol; the
winner is chosen by `(cost, universe priority)` — completion order can never
leak into results. The winning clustering is re-materialized sequentially
(determinism over a microsecond of recompute, per the CEP&CC 38.4 priority
order).

## JIT

`jit::driver` holds the versioned manifests:
- `sxla-tier1-2026-09`: verify → canonicalize-l0 (fold+GVN+DCE)
- `sxla-tier2-2026-09`: + egraph-saturate + fusion universe search
- `jit::cache` — EBR-sharded cache keyed on (fingerprint, shape, layout)
- `jit::boundary` — the SPSC request/response rings
