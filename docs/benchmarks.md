# SXLA Cross-System Kernel Benchmarks

End-to-end measurement of the SXLA tier stack on numerical kernels, against
NumPy and XLA reference points (CEP&CC Law 4: no unmeasured hot code; 13.2:
measurement artifacts). The interpreter's kernels carried only O(...) COST
claims before this harness existed; the tables below are the measured truth,
archived in `.cep/evidence/kernel-bench-2026-10-19.txt`.

## Harness

| Piece | Where | Role |
|-------|-------|------|
| Rust harness | `benches/kernel_bench` | Compiles 13 XIR kernels at Tiers 0/1/2, executes with deterministic LCG tensor inputs, times compile + execute, checks cross-tier BIT agreement, dumps inputs/results + `manifest.json` |
| Python reference | `benches/kernel_bench/ref_xla.py` | Re-generates the same LCG inputs (bit-checked against the dumps), times NumPy / JAX eager / JAX jit (= real XLA compilation), checks value agreement vs the SXLA dumps, writes `xla_results.json` |

Kernel classes: elementwise chain `relu(x*0.5+0.25)` (64/256/1024 square),
CSE demonstrator `(x*0.5)+(x*0.5)` (256), axis-0 sum reduction (64/256/1024),
vector dot (1024/4096/16384), square matmul (64/128/256). Fixed per-kernel
iteration counts live in `benches/kernel_bench/src/suite.rs` (the protocol);
`--scale 0.1` is the CI smoke variant.

## Methodology

- Data: per-param LCG stream (`v = ((s >> 33) / 2^31) * 2 - 1`, wrapping u64
  arithmetic, per-param seed = 12345 + param index). Only exact power-of-two
  divisions — Rust and Python generate identical bits, and the Python side
  bit-checks its stream against the harness dumps before any comparison: a
  disagreement can only come from compute, never from data.
- Timing: warmup 10% of iterations, then the mean over the timed loop
  (`std::time::Instant` / `time.perf_counter`). One SXLA iteration = one full
  `execute()` call (value-table setup including input cloning + every
  instruction) — the Tier-0 dispatch contract, measured as shipped. JAX calls
  `block_until_ready()`.
- Environment (this run): Intel Xeon, 2 cores, x86_64 linux; rustc 1.98.1
  (release: lto, codegen-units=1, overflow-checks=true, debug_assertions off —
  the harness refuses debug builds); Python 3.12.14, NumPy 2.5.4, JAX 0.12.0
  (jaxlib 0.12.0, CPU backend, x64 enabled, default XLA flags).
- Tolerances: elementwise must agree within one contraction step
  (<= 1e-12 rel; bit-exactness reported); reduction/dot/matmul within
  summation-order noise (<= 1e-9 rel, observed values reported alongside
  exact `math.fsum` ground-truth errors where computable).
- Wall-clock numbers are advisory evidence, never translation inputs
  (CEP&CC 38.10 / anvil_bench precedent).

## Results (evidence run 2026-10-19)

SXLA per tier (mean ns per `execute()`; compile is single-shot):

| Kernel | Tier 0 | Tier 1 | Tier 2 | compile t0/t1/t2 (us) |
|--------|-------:|-------:|-------:|----------------------|
| ew_chain_64 | 9,783 | 9,753 | 9,963 | 51 / 71 / 540 |
| ew_chain_256 | 217,499 | 215,733 | 217,031 | 24 / 113 / 374 |
| ew_chain_1024 | 8,548,103 | 5,760,173 | 5,896,932 | 34 / 43 / 337 |
| ew_cse_256 | 227,732 | 155,848 | 155,169 | 27 / 53 / 321 |
| reduce_sum_64 | 5,633 | 5,611 | 5,651 | 20 / 15 / 270 |
| reduce_sum_256 | 81,755 | 81,817 | 81,322 | 10 / 20 / 277 |
| reduce_sum_1024 | 10,551,138 | 10,506,499 | 10,440,534 | 24 / 40 / 340 |
| dot_1024 | 1,181 | 1,762 | 1,171 | 18 / 17 / 287 |
| dot_4096 | 5,274 | 5,270 | 5,278 | 10 / 16 / 248 |
| dot_16384 | 20,585 | 20,620 | 20,590 | 10 / 14 / 239 |
| matmul_64 | 268,152 | 268,718 | 267,726 | 12 / 19 / 297 |
| matmul_128 | 1,971,668 | 1,965,645 | 1,977,232 | 13 / 23 / 266 |
| matmul_256 | 16,763,725 | 16,597,578 | 16,539,841 | 13 / 22 / 253 |

Cross-system (dumped kernels; ns per call):

| Kernel | SXLA t2 | NumPy | JAX eager | XLA jit | t2 / XLA jit |
|--------|--------:|------:|----------:|--------:|-------------:|
| ew_chain_64 | 9,963 | 5,770 | 57,803 | 16,392 | 0.6x |
| ew_cse_256 | 155,169 | 50,903 | 179,882 | 56,698 | 2.7x |
| reduce_sum_64 | 5,651 | 8,682 | 25,553 | 23,762 | 0.2x |
| reduce_sum_1024 | 10,440,534 | 382,560 | 361,852 | 359,703 | 29x |
| dot_1024 | 1,171 | 969 | 23,852 | 23,918 | 0.05x |
| dot_16384 | 20,590 | 3,423 | 35,500 | 52,156 | 0.4x |
| matmul_64 | 267,726 | 8,999 | 47,587 | 47,785 | 5.6x |
| matmul_128 | 1,977,232 | 67,853 | 103,484 | 102,784 | 19x |
| matmul_256 | 16,539,841 | 334,260 | 604,884 | 587,928 | 28x |

Value agreement (the conformance result; all input bit-checks passed):

| Kernel | SXLA vs NumPy | SXLA vs XLA jit | vs exact (fsum) sxla/np/xla |
|--------|---------------|-----------------|----------------------------|
| ew_chain_64 | bit-exact | bit-exact | - |
| ew_cse_256 | bit-exact | bit-exact | - |
| reduce_sum_64 | bit-exact | bit-exact | 0/0/0 |
| reduce_sum_1024 | bit-exact | bit-exact | 0/0/0 |
| dot_1024 | 1.5e-16 | 7.3e-16 | 1.5e-16 / 2.9e-16 / 5.9e-16 |
| dot_16384 | 6.5e-16 | 3.2e-16 | 3.2e-16 / 3.2e-16 / 0 |
| matmul_64 | 1.2e-12 | 1.2e-12 | - |
| matmul_128 | 3.2e-12 | 3.2e-12 | - |
| matmul_256 | 5.7e-12 | 5.7e-12 | - |

## Findings

1. **Value agreement is pinned**: all three SXLA tiers agree bitwise on
   every kernel (in-harness regression `tiers_agree_bitwise_per_class`), and
   against NumPy and XLA the elementwise/reduction kernels agree BIT-EXACT;
   dot and matmul agree to summation-order noise (1e-16 / 1e-12), with every
   system within the same distance of the exact fsum ground truth. This is
   the cross-system differential evidence for the interpreter's semantics.
2. **The measured cost of Tier 2 compilation**: 240-540us single-shot on
   these graphs, vs 10-70us for Tiers 0/1 — the e-graph saturation + fusion
   search price on 4-6 node programs. Tier 1 sits far inside its < 5ms
   budget claim; Tier 2's own budget accounting is CEP-20/21 territory.
3. **Optimization shows up as execution time only where it removes work**:
   the CSE demonstrator (duplicate `x*0.5`) runs 32% faster at Tiers 1/2
   (228us -> 156us; 4 -> 3 instructions). Where tiers do not change the
   instruction count (elementwise chains, reductions, matmul at these
   shapes), execution time is flat across tiers — no phantom speedups.
4. **The honest gap**: on compute-dense kernels SXLA Tier 0 is 5.6-29x
   slower than XLA jit and 6-50x slower than NumPy (BLAS) — matmul_256 is
   the worst case. The disassembly explains it: the release-build matmul
   inner loop is scalar `mulsd`+`addsd` with a loop-carried accumulator and
   overflow-checked indices (baseline x86-64, no fast-math contraction, no
   SIMD), while XLA/BLAS ship blocked, vectorized, FMA kernels. The dot
   kernel auto-vectorizes 2-wide (`mulpd`) and correspondingly reaches
   1.3 ns/element vs matmul's ~1 ns/FMA.
5. **The honest win**: on dispatch-bound kernels (reduce_sum_64, dot_1024,
   ew_chain_64) SXLA BEATS XLA jit by 1.7-20x and JAX eager by ~5-25x — a
   native interpreter pays no per-call dispatch/platform fee. The crossover
   on this target sits between 64^2 and 128^2 matmul work.
6. **Tier 0 vs 1 layout effects**: ew_chain_1024 measured 8.5ms (t0) vs
   5.8ms (t1) at IDENTICAL instruction counts — canonicalization changes
   the value-table layout; reported as observed, min/max timing statistics
   are follow-up work (CEP-28).
7. **Variance**: this is a shared 2-core box; run-to-run spread was
   20-35% on the memory-bound 1M-element kernels and a few percent on
   matmul. The ratio conclusions (0.05x to 29x) are robust to that spread;
   single-digit-percent comparisons are not — do not make them from this
   data.

## Interpretation

Tier 0 is the correctness-first reference: per-instruction dispatch, safe
bounds-checked scalar kernels, value cloning per execute() — the measured
gap is the price of that contract, and the value agreement above is what it
buys. Tiers 1/2 currently optimize instruction count and fusion decisions,
not kernel microarchitecture; closing the compute-dense gap (blocked/vector
kernels, target features) is exactly the Tier-1 fast path and Tier-2 cost
model work tracked by CEP-20/21, now with a measured baseline to beat.

## Reproduction

```sh
cargo run --release --bin kernel_bench -- --out /tmp/kb
# Python side (numpy + jax; see harness header for the venv):
python3 benches/kernel_bench/ref_xla.py --dir /tmp/kb
```

Law 4 checklist: benchmark results + target description + compiler version +
compile flags + input description + date — in the evidence file, the
manifest, and this document; disassembly excerpts for the two hot loops are
appended to the evidence file.

## Follow-ups

- CEP-28: conv kernel in the cross-system suite (needs semantics-matched
  padding/stride agreement with XLA's `conv_general_dilated`), per-iteration
  min/max timing statistics, and a CI wiring decision for the Python side
  (CI has no jax; the Rust smoke run is what CI can archive).
