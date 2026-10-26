# SXLA Cross-System Kernel Benchmarks

End-to-end measurement of the SXLA tier stack on numerical kernels, against
NumPy and XLA reference points (CEP&CC Law 4: no unmeasured hot code; 13.2:
measurement artifacts). The interpreter's kernels carried only O(...) COST
claims before this harness existed; the tables below are the measured truth,
archived in `.cep/evidence/kernel-bench-2026-10-20.txt` (the CEP-28 run: conv
class + per-iteration min/max statistics; the prior 2026-10-19 run remains
archived alongside it).

## Always-on benchmarking (session 6)

Benchmark evidence accrues continuously, not on demand:
`.github/workflows/bench.yml` (mirrored at
`docs/bench-workflow.reference.yml`) runs `anvil_bench` and the kernel bench
smoke (`--scale 0.1`) on EVERY push to main, every pull request, daily at
06:00 UTC, and on manual dispatch, uploading per-run artifacts with 90-day
retention so timing trends are comparable across runner generations. The
daily schedule exists to catch environment drift (runner hardware,
toolchain) independent of code changes. Locally, `scripts/bench.sh [scale]`
produces the same evidence files under `.cep/evidence/`. Numbers are
advisory evidence (shared-runner timing is noise-prone; nothing is gated on
them), while the benches' built-in correctness assertions (cross-tier bit
agreement, reference value checks) remain hard. Full-scale cross-system
reference runs against NumPy/XLA stay manual, per the reproduction section
below — CI runners have no jax.

## Harness

| Piece | Where | Role |
|-------|-------|------|
| Rust harness | `benches/kernel_bench` | Compiles 16 XIR kernels at Tiers 0/1/2, executes with deterministic LCG tensor inputs, times compile + execute (mean, plus min/max from a separate per-iteration sampled loop), checks cross-tier BIT agreement, dumps inputs/results + `manifest.json` |
| Python reference | `benches/kernel_bench/ref_xla.py` | Re-generates the same LCG inputs (bit-checked against the dumps), times NumPy / JAX eager / JAX jit (real XLA compilation) with the same mean + min/max protocol, checks value agreement vs the SXLA dumps (conv included: explicit TF-formula padding, NCHW/OIHW dimension numbers), writes `xla_results.json` |

Kernel classes: elementwise chain `relu(x*0.5+0.25)` (64/256/1024 square),
CSE demonstrator `(x*0.5)+(x*0.5)` (256), axis-0 sum reduction (64/256/1024),
vector dot (1024/4096/16384), square matmul (64/128/256), and since CEP-28
three NCHW x OIHW convs with 3x3 kernels — valid/stride-1
(`[1,8,32,32]x[16,8,3,3] -> [1,16,30,30]`), valid/stride-2 (same tensors ->
`[1,16,15,15]`), and same/stride-2 on `[1,4,16,16]x[8,4,3,3]` (pad_total 1
per axis, split (0,1) — the odd pixel bottom/right). Fixed per-kernel
iteration counts live in `benches/kernel_bench/src/suite.rs` (the protocol);
`--scale 0.1` is the CI smoke variant.

## Methodology

- Data: per-param LCG stream (`v = ((s >> 33) / 2^31) * 2 - 1`, wrapping u64
  arithmetic, per-param seed = 12345 + param index). Only exact power-of-two
  divisions — Rust and Python generate identical bits, and the Python side
  bit-checks its stream against the harness dumps before any comparison: a
  disagreement can only come from compute, never from data.
- Timing: warmup 10% of iterations, then the mean over the un-instrumented
  timed loop (`std::time::Instant` / `time.perf_counter`) — the protocol is
  unchanged from the first evidence run, so means stay comparable across
  runs. Since CEP-28 a SECOND per-iteration sampled loop records min/max;
  each sample carries ~2 clock reads of instrumentation (~50ns Rust,
  ~150ns Python — negligible above ~1us kernels, documented here). One SXLA
  iteration = one full `execute()` call (value-table setup including input
  cloning + every instruction) — the Tier-0 dispatch contract, measured as
  shipped. JAX calls `block_until_ready()`.
- Conv semantics matching (CEP-28): SXLA's conv follows the TF/XLA
  conventions — valid output size `floor((h-kh)/s)+1` equals TF's
  `ceil((h-kh+1)/s)`; same pads `(oh-1)*s+kh-h` total, floor split, odd
  pixel bottom/right. The Python side passes EXPLICIT padding lists computed
  from that formula to XLA's `conv_general_dilated` with dimension numbers
  `("NCHW", "OIHW", "NCHW")` (the interpreter's exact layouts), so any value
  disagreement can only be summation order, never padding/stride semantics.
  The two suite known-value tests pin the conventions against hand-computed
  values that were cross-verified against XLA before pinning.
- Environment (this run): Intel Xeon, 2 cores, x86_64 linux; rustc 1.98.1
  (release: lto, codegen-units=1, overflow-checks=true, debug_assertions off —
  the harness refuses debug builds); Python 3.12.14, NumPy 2.5.4, JAX 0.12.0
  (jaxlib 0.12.0, CPU backend, x64 enabled, default XLA flags).
- Tolerances: elementwise must agree within one contraction step
  (<= 1e-12 rel; bit-exactness reported); reduction/dot/matmul/conv within
  summation-order noise (<= 1e-9 rel, observed values reported alongside
  exact `math.fsum` ground-truth errors where computable — conv included:
  one fsum per output position over its taps).
- Wall-clock numbers are advisory evidence, never translation inputs
  (CEP&CC 38.10 / anvil_bench precedent).

## Results (evidence run 2026-10-20)

SXLA per tier (mean ns per `execute()`; compile is single-shot):

| Kernel | Tier 0 | Tier 1 | Tier 2 | compile t0/t1/t2 (us) |
|--------|-------:|-------:|-------:|----------------------|
| ew_chain_64 | 9,752 | 12,009 | 10,022 | 54 / 103 / 633 |
| ew_chain_256 | 231,166 | 231,428 | 226,329 | 30 / 171 / 400 |
| ew_chain_1024 | 8,688,954 | 5,800,001 | 5,761,551 | 31 / 52 / 375 |
| ew_cse_256 | 240,286 | 158,710 | 158,645 | 30 / 66 / 358 |
| reduce_sum_64 | 5,499 | 5,461 | 5,459 | 20 / 27 / 298 |
| reduce_sum_256 | 85,804 | 86,137 | 87,179 | 10 / 32 / 418 |
| reduce_sum_1024 | 11,002,593 | 11,125,882 | 11,208,759 | 28 / 40 / 334 |
| dot_1024 | 1,191 | 1,204 | 1,198 | 20 / 28 / 336 |
| dot_4096 | 5,351 | 5,353 | 7,024 | 15 / 16 / 293 |
| dot_16384 | 21,052 | 21,148 | 21,076 | 15 / 14 / 315 |
| matmul_64 | 269,268 | 268,962 | 269,864 | 14 / 33 / 373 |
| matmul_128 | 1,986,210 | 1,999,869 | 1,997,253 | 21 / 29 / 334 |
| matmul_256 | 16,697,878 | 17,589,710 | 16,673,797 | 18 / 33 / 323 |
| conv_valid_32 | 3,037,383 | 3,238,673 | 3,034,757 | 28 / 33 / 354 |
| conv_valid_s2_32 | 772,006 | 766,889 | 771,333 | 22 / 42 / 357 |
| conv_same_s2_16 | 53,291 | 53,071 | 52,961 | 32 / 32 / 368 |

Cross-system (dumped kernels; mean ns per call):

| Kernel | SXLA t2 | NumPy | JAX eager | XLA jit | t2 / XLA jit |
|--------|--------:|------:|----------:|--------:|-------------:|
| ew_chain_64 | 10,022 | 4,977 | 71,070 | 20,999 | 0.5x |
| ew_cse_256 | 158,645 | 53,514 | 615,666 | 95,358 | 1.7x |
| reduce_sum_64 | 5,459 | 3,640 | 29,283 | 30,685 | 0.2x |
| reduce_sum_1024 | 11,208,759 | 431,355 | 412,124 | 372,102 | 30.1x |
| dot_1024 | 1,198 | 880 | 29,348 | 28,500 | 0.04x |
| dot_16384 | 21,076 | 3,503 | 46,908 | 26,775 | 0.8x |
| matmul_64 | 269,864 | 9,054 | 48,481 | 36,434 | 7.4x |
| matmul_128 | 1,997,253 | 79,947 | 148,061 | 108,746 | 18.4x |
| matmul_256 | 16,673,797 | 371,226 | 603,327 | 615,346 | 27.1x |
| conv_valid_32 | 3,034,757 | 286,210 | 379,064 | 369,965 | 8.2x |
| conv_valid_s2_32 | 771,333 | 188,054 | 202,356 | 177,357 | 4.3x |
| conv_same_s2_16 | 52,961 | 92,003 | 77,193 | 41,463 | 1.3x |

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
| conv_valid_32 | 4.0e-12 | 3.4e-12 | 8.5e-12 / 1.3e-11 / 5.1e-12 |
| conv_valid_s2_32 | 1.6e-13 | 2.0e-13 | 1.5e-13 / 2.1e-13 / 1.6e-13 |
| conv_same_s2_16 | 8.1e-13 | 7.0e-12 | 6.2e-12 / 5.4e-12 / 1.3e-11 |

Per-iteration spread, min..max ns (the CEP-28 diagnostic; full table in the
evidence file; sampled loops carry the documented timer-read overhead):

| Kernel | SXLA t2 min..max | NumPy min..max | XLA jit min..max |
|--------|-----------------:|---------------:|-----------------:|
| ew_chain_64 | 9,508..21,580 | 4,833..26,946 | 8,268..24,279,973 |
| ew_chain_1024 t0 / t1 | 7,813,204..8,626,006 / 5,505,605..6,222,669 | - | - |
| dot_1024 | 1,181..9,949 | 893..16,194 | 10,498..26,335,477 |
| matmul_256 | 16,617,159..16,887,038 | 343,956..382,841 | 560,452..586,716 |
| conv_valid_32 | 2,944,268..3,124,615 | 235,977..377,372 | 291,009..567,724 |
| conv_same_s2_16 | 49,120..73,169 | 79,258..160,448 | 23,969..152,078 |

## Findings

1. **Value agreement is pinned, conv included**: all three SXLA tiers agree
   bitwise on every kernel (in-harness regression `tiers_agree_bitwise_per_class`,
   conv case included), and against NumPy and XLA the elementwise/reduction
   kernels agree BIT-EXACT; dot, matmul, and conv agree to summation-order
   noise (1e-16 / 1e-12 / 1e-12), with every system within the same distance
   of the exact fsum ground truth. The conv cross-check is the sharp one: the
   same/stride-2 kernel's odd padding pixel lands bottom/right in both SXLA
   and XLA — the semantics-matched explicit padding proves the conventions
   agree, so the residual 1e-12 differences are summation order only.
2. **The measured cost of Tier 2 compilation**: 290-630us single-shot on
   these graphs, vs 10-103us for Tiers 0/1 — the e-graph saturation + fusion
   search price on 4-6 node programs. Tier 1 sits far inside its < 5ms
   budget claim; Tier 2's own budget accounting is CEP-20/21 territory.
3. **Optimization shows up as execution time only where it removes work**:
   the CSE demonstrator (duplicate `x*0.5`) runs 34% faster at Tiers 1/2
   (240us -> 159us; 4 -> 3 instructions). Where tiers do not change the
   instruction count (elementwise chains, reductions, matmul and conv at
   these shapes), execution time is flat across tiers — no phantom speedups.
4. **The honest gap**: on compute-dense kernels SXLA Tier 0 is 7-30x slower
   than XLA jit (matmul_128/256 the worst compute cases; reduce_sum_1024's
   30x is the sequential-axis-0 fold vs XLA's vectorized tree reduction).
   The disassembly explains it: the release-build matmul/conv inner loops
   are scalar `mulsd`+`addsd` with overflow-checked index math (`imul`+`jo`)
   and, for conv, per-tap bounds checks — baseline x86-64, no fast-math
   contraction, no SIMD — while XLA/BLAS ship blocked, vectorized, FMA
   kernels. The dot kernel auto-vectorizes 2-wide (`mulpd`); conv runs
   ~3 ns/FMA vs matmul's ~1 ns/FMA (deeper index arithmetic per tap).
5. **The honest win**: on dispatch-bound kernels (reduce_sum_64, dot_1024,
   dot_16384, ew_chain_64) SXLA BEATS XLA jit by 1.3-24x and JAX eager by
   2.2-25x — a native interpreter pays no per-call dispatch/platform fee.
   conv_same_s2_16 is the new boundary datum: SXLA (53us) beats NumPy's
   einsum path (92us) and sits within 1.3x of XLA jit. The crossover on
   this target sits between 64^2 and 128^2 matmul work.
6. **Min/max separates layout effects from scheduler noise (CEP-28's
   question answered)**: the ew_chain_1024 Tier-0-vs-Tier-1 gap at IDENTICAL
   instruction counts is a STABLE LAYOUT EFFECT — t0 min 7.81ms vs t1 min
   5.51ms, spreads tight within each tier (max/min ~1.1) — not run-to-run
   luck. Meanwhile XLA jit's per-iteration maxima on tiny kernels hit 24-26ms
   (dispatch/GC hiccups, ~3000x its min), and NumPy shows a 13ms outlier on
   reduce_sum_64: SXLA's execution-time distribution is the tightest of the
   three systems on this box. XLA's minima, though, are genuinely fast
   (ew_chain_64 min 8.3us vs SXLA min 9.5us) — the platform's amortized
   cost, not its kernel, is what the mean reports. The min also exposes
   mean-level noise inside SXLA itself: this run's ew_chain_64 tier-1 MEAN
   (12.0us) reads above tier-0 (9.8us) while their minima are identical
   (9.44us both) — on a 2-core box the mean of a 10us kernel carries
   outlier weight that the min does not.
7. **Variance**: this is a shared 2-core box; run-to-run spread was
   20-35% on the memory-bound 1M-element kernels and a few percent on
   matmul/conv. The ratio conclusions (0.04x to 30x) are robust to that
   spread; single-digit-percent comparisons are not — do not make them from
   this data.

## Interpretation

Tier 0 is the correctness-first reference: per-instruction dispatch, safe
bounds-checked scalar kernels, value cloning per execute() — the measured
gap is the price of that contract, and the value agreement above is what it
buys. Tiers 1/2 currently optimize instruction count and fusion decisions,
not kernel microarchitecture; closing the compute-dense gap (blocked/vector
kernels, target features) is exactly the Tier-1 fast path and Tier-2 cost
model work tracked by CEP-20/21, now with a measured baseline to beat —
conv included, where the reference gap is 1.3-8.2x (narrower than matmul's
7-27x, since XLA's own CPU conv lowering is less aggressively tuned than
BLAS GEMM at these sizes, and SXLA's conv is index-heavier per FMA).

## Reproduction

```sh
cargo run --release --bin kernel_bench -- --out /tmp/kb
# Python side (numpy + jax; see harness header for the venv):
python3 benches/kernel_bench/ref_xla.py --dir /tmp/kb
```

Law 4 checklist: benchmark results + target description + compiler version +
compile flags + input description + date — in the evidence file, the
manifest, and this document; disassembly excerpts for the three hot loops
(matmul, dot, conv tap loop) are appended to the evidence file.

## Follow-ups

- CEP-28 is CLOSED (this run): conv kernels cross-checked vs NumPy/XLA with
  semantics-matched padding; per-iteration min/max statistics in both the
  Rust harness and the Python reference (mean protocol unchanged, so
  cross-run comparisons stay valid). The CI smoke archive step runs the
  suite as-is (conv kernels included; the cross-system Python side stays a
  reference-target activity — CI runners have no jax).
- Remaining majors: CEP-20/21 (Tier-2 ranker + autotuning against this
  measured baseline), CEP-22/CEP-19 (the multi-session scheduling/Presburger
  work).
