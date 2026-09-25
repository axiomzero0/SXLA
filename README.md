# SXLA

An MLIR-inspired, Rust-native compiler stack for accelerators and modern CPUs,
built to the CEP&CC 0.1 ("Cycle-Exact Performance & Clean Code, Psychopathic
Tier") engineering standard in [standards.md](standards.md).

SXLA implements the master architecture:

- **Anvil** — a custom thread-per-core, NUMA-aware, lock-free execution
  engine with 4 gears: static partitioning (zero-sync), fork/join work
  stealing with deterministic task fuel, SPSC lock-free pipelines, and
  epoch-based reclamation.
- **XIR** — a 5-level internal IR (graph → tensor → fusion → loop → target)
  with packed deterministic ids, bounded arenas, transactional snapshots, and
  a canonical round-trippable text format.
- **E-graph** — optional equality saturation with fusion-aware extraction
  (layout/locality penalties) and float-reassociation gating per CEP&CC 38.24.
- **Fusion** — a parallel multi-universe search (aggressive epilogue /
  split reductions / rematerialization) with atomic pruning via a global
  `AtomicU64` best-cost, a legality engine, resource budgets, repair passes
  and a tiered cost model.
- **JIT** — tiered compilation (Tier 0 interpreter → Tier 1 fast → Tier 2
  full search), an SPSC compilation boundary, and an EBR-sharded kernel
  cache with lock-free reads.
- **Runtime** — CPU device, streams/events, and the reference interpreter.

## Quick start

```sh
cargo build --workspace        # zero external dependencies
cargo test --workspace         # 140+ unit/integration/differential tests
cargo clippy --workspace --all-targets   # warnings-as-errors
cargo fmt --check

# Optimize and print IR:
./target/debug/xla-opt --explain-pipeline demo.xir
./target/debug/xla-opt demo.xir

# Compile at a tier and execute:
./target/debug/xla-run --tier 2 --input 21 demo.xir
```

Example `demo.xir`:

```
xir v1 func @main {
  %0 = param 0
  %1 = const.f64 2.0
  %2 = binary.mul %0, %1
  %3 = binary.add %2, %2
}
```

## Layout

```
crates/anvil        Concurrency engine (bump arenas, Chase-Lev, SPSC, EBR, executor)
crates/xir-core     Node/edge storage, types, ops, bounded arenas, snapshots, text IR
crates/xir-graph    Sea-of-nodes: verifier, dominance, GVN/CSE, const-fold, DCE, scheduler
crates/xir-levels   The 5 IR levels + the Anvil-aware pass manager
crates/egraph       Union-find, rewrite rules, fusion-aware extraction, saturation
crates/fusion       Legality, resource model, tiered cost, universe search, repair
crates/codegen      Bufferization, tiling, vectorization, target lowering
crates/runtime      CPU device, streams, Tier-0 reference interpreter
crates/jit          Tiered dispatcher, SPSC boundary, EBR kernel cache
tools/xla-opt       CLI: IR manipulation and debugging
tools/xla-run       Standalone JIT executor
benches/anvil_bench Measurement harness (CEP&CC Law 4 evidence)
tests/              Integration + differential tests
.cep/               Lint config, waivers, evidence
```

## Standards conformance

Every first-party file carries a machine-parseable `CEP:FILE` header and every
nontrivial item carries CEP comment blocks (`CEP:WHAT/WHY/STATUS/FAILURE/
ASSUMES/COST/EVIDENCE/SECURITY`); unsafe blocks carry `CEP:UNSAFE | Safety`
comments enforced by clippy. Run the mechanical checker:

```sh
python3 scripts/cep_lint.py crates tools benches tests
```

See [docs/conformance.md](docs/conformance.md) for the honest status catalog
(complete / partial / stub / placeholder per CEP&CC 10.5) and
[docs/pipeline.md](docs/pipeline.md) for the versioned pass manifests.

Zero external dependencies: the supply chain is pinned by construction
(CEP&CC 22.11) and builds are deterministic (38.10).
