# Cost Model Calibration (Tier-1 Analytic)

Abstract cost units; the JIT never converts these to wall-clock (HPC
determinism, CEP&CC 38.10).

## Fusion cost

`cost(cluster plan) = external_edge_bytes + Σ op_arithmetic + spill_penalty`

- External edge bytes: 1 unit per byte of value edges crossing cluster
  boundaries (fusion internalizes them into registers).
- `op_arithmetic` (units/op): const/param 1, elementwise 2, reduce 4, rng 4,
  dot 8, if 8, transpose/broadcast 16, matmul 16, mma 12, conv 24, custom 32.
- Spill penalty: 8 units per register unit over `REGISTER_BUDGET_UNITS`.

## E-graph extraction

`cost(node) = base_cost(op) + Σ child class costs`, tie-break by insertion
seq. Fusion-aware penalties (master architecture section 4):

- transpose/broadcast base cost = `FUSION_LOCALITY_PENALTY` = 64 units
  (dwarfs per-op costs so layout-breaking rewrites survive only when they
  save a whole materialization).

## Benchmarks

`cargo run --release --bin anvil_bench` measures the CEP-0 primitives;
CI archives output under `.cep/evidence/`. Numbers from the reference runner
(dev container, x86_64):

- bump alloc: ~16 ns/op (64 allocations per measured op group)
- chase-lev push+pop: ~63 ns/pair (incl. deque construction)
- spsc push+pop: ~64 ns/pair (incl. ring construction)
- ebr pin+unpin: ~123 ns/pair (incl. collector construction)

Per-op amortized costs within a pre-built structure are lower; the harness
measures the conservative construct+operate cycle. CEP-21 wires these as CI
regression gates.
