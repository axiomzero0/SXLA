#!/usr/bin/env python3
# CEP:FILE: benches/kernel_bench/ref_xla.py
# CEP:WHAT: Python reference side of the cross-system kernel benchmark:
#           re-generates the LCG inputs, bit-checks them against the Rust
#           harness dumps, times NumPy / JAX eager / JAX jit (XLA), and
#           checks value agreement against the SXLA result dumps.
# CEP:WHY: CEP&CC Law 4 (measurement evidence) + cross-system value
#          agreement: the Rust workspace stays zero-dependency (22.11),
#          so the reference systems run here instead. This file is not
#          cep_lint-scanned (the linter covers *.rs); CEP blocks are kept
#          for house consistency.
# CEP:CLASS: CEP-2
# CEP:STATUS: complete
# CEP:FAILURE: exit 1 on input bit mismatch, tolerance violation, or
#              missing dumps/manifest; exit 2 on CLI misuse.
# CEP:ASSUMES: numpy present; jax present unless --skip-jax; jax x64
#              enabled at import; run against a kernel_bench --out dump
#              directory produced on the same target.
# CEP:COST: seconds (LCG regeneration dominates for 1M-element dumps).
# CEP:EVIDENCE: docs/benchmarks.md; xla_results.json written to --dir.
# CEP:SECURITY: reads only the harness's own dumps/manifest.
# CEP:HPC-DETERMINISM: input regeneration and agreement checks are
#          bit-deterministic; timings are wall-clock advisory (13.2).

"""Cross-system reference benchmark: NumPy / JAX eager / JAX jit (XLA).

Tolerance policy (documented, not ad hoc):
- elementwise kernels: IEEE per-element ops agree bitwise OR within one
  contraction (FMA) step -> assert max_rel_err <= 1e-12, report bitexact.
- reduction/dot/matmul: summation ORDER differs between the sequential
  left-to-right SXLA fold and pairwise/SIMD reference folds -> assert
  max_rel_err <= 1e-9, report the observed value and the error of every
  system against an exact (math.fsum) ground truth where computable.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# Relative-error tolerances (see module docstring).
TOL_ELEMENTWISE = 1e-12
TOL_SUMMATION = 1e-9

MASK64 = 0xFFFFFFFFFFFFFFFF


def lcg_fill(seed: int, n: int) -> np.ndarray:
    """Rust-suite-identical LCG (see benches/kernel_bench/src/suite.rs)."""
    mul = 6364136223846793005
    add = 1442695040888963407
    s = seed & MASK64
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = (s * mul + add) & MASK64
        out[i] = ((s >> 33) / 2147483648.0) * 2.0 - 1.0
    return out


def read_f64(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 8 != 0:
        raise ValueError(f"{path}: length {len(data)} not a multiple of 8")
    return np.frombuffer(data, dtype="<f8").copy()


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip() == "model name":
                        return v.strip()
    except OSError:
        pass
    return "unknown"


def max_rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    denom = np.maximum(np.abs(b), 1e-30)
    return float(np.max(np.abs(a - b) / denom))


def bitexact(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.atleast_1d(np.asarray(a, dtype=np.float64))
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def numpy_impl(name: str, cls: str, xs):
    if cls == "elementwise":
        x = xs[0]
        if name.startswith("ew_chain"):
            return np.maximum(x * 0.5 + 0.25, 0.0)
        return x * 0.5 + x * 0.5
    if cls == "reduction":
        return np.sum(xs[0], axis=0)
    if cls == "dot":
        return np.dot(xs[0], xs[1])
    return xs[0] @ xs[1]


def fsum_exact(cls: str, xs):
    """Exact ground truth where computable (reduction, dot)."""
    if cls == "reduction":
        x = xs[0]
        return np.array([math.fsum(x[:, j]) for j in range(x.shape[1])])
    if cls == "dot":
        return np.array([math.fsum(xs[0] * xs[1])])
    return None


def time_fn(fn, iters: int) -> float:
    """House timing protocol: warmup 10%, mean over the timed loop (ns)."""
    warm = iters // 10 + 1
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e9 / iters


def main() -> int:
    ap = argparse.ArgumentParser(description="SXLA cross-system reference benchmark")
    ap.add_argument("--dir", required=True, help="kernel_bench --out dump directory")
    ap.add_argument("--skip-jax", action="store_true", help="numpy-only mode")
    args = ap.parse_args()

    manifest_path = os.path.join(args.dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ref_xla: missing {manifest_path}", file=sys.stderr)
        return 1
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    have_jax = not args.skip_jax
    if have_jax:
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
        except ImportError:
            print("ref_xla: jax unavailable — numpy-only mode", file=sys.stderr)
            have_jax = False

    def jax_impl(name: str, cls: str, xs):
        if cls == "elementwise":
            x = xs[0]
            if name.startswith("ew_chain"):
                return jnp.maximum(x * 0.5 + 0.25, 0.0)
            return x * 0.5 + x * 0.5
        if cls == "reduction":
            return jnp.sum(xs[0], axis=0)
        if cls == "dot":
            return jnp.dot(xs[0], xs[1])
        return xs[0] @ xs[1]

    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": None,
        "jaxlib": None,
        "backend": None,
        "x64": False,
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "default"),
    }
    if have_jax:
        import jax
        import jaxlib

        env["jax"] = jax.__version__
        env["jaxlib"] = jaxlib.__version__
        env["backend"] = jax.default_backend()
        env["x64"] = bool(jax.config.x64_enabled)

    print(
        f"ref_xla — python {env['python']} | numpy {env['numpy']} | "
        f"jax {env['jax']} ({env['backend']}, x64={env['x64']}) | "
        f"{env['cpu']} | cores={env['cores']}"
    )
    print(
        f"{'kernel':<16} {'class':<12} {'inputs=bits':<12} {'sxla~np':<18} "
        f"{'sxla~xla':<18} {'exact-err(sxla/np/xla)':<24}"
    )

    failures = []
    out_kernels = []
    for k in manifest["kernels"]:
        if not k.get("dump") or not k.get("result", {}).get("file"):
            continue
        name, cls, iters = k["name"], k["class"], int(k["iters"])
        # Regenerate inputs and bit-check against the dumps.
        xs, inputs_ok = [], True
        for pi, p in enumerate(k["params"]):
            if not p.get("file"):
                inputs_ok = False
                break
            gen = lcg_fill(12345 + pi, int(p["elems"]))
            dump = read_f64(os.path.join(args.dir, p["file"]))
            if not bitexact(gen, dump):
                print(
                    f"ref_xla: INPUT MISMATCH {name} p{pi}: LCG protocol broken",
                    file=sys.stderr,
                )
                failures.append(f"inputs:{name}:p{pi}")
                inputs_ok = False
                break
            xs.append(gen.reshape([int(d) for d in p["dims"]]))
        if not inputs_ok:
            continue

        rdims = [int(d) for d in k["result"]["dims"]]
        sx = read_f64(os.path.join(args.dir, k["result"]["file"]))
        # Flat dump -> result shape ([] = scalar, kept as a 1-element array).
        sx = sx.reshape(rdims) if rdims else sx

        # NumPy reference.
        np_res = numpy_impl(name, cls, xs)
        agree_np = {
            "bitexact": bitexact(np_res, sx),
            "max_rel_err": max_rel_err(np_res, sx),
        }

        # JAX references (eager + jit/XLA). Inputs convert to jax arrays so
        # eager dispatch stays on the jax path (numpy operands would win
        # operator dispatch and never reach XLA).
        agree_jit = None
        jax_rows = {}
        if have_jax:
            xs_j = [jnp.asarray(x) for x in xs]
            eager_res = jax_impl(name, cls, xs_j)
            eager_res.block_until_ready()
            jax_rows["eager_ns"] = time_fn(
                lambda: jax_impl(name, cls, xs_j).block_until_ready(), iters
            )
            jitted = jax.jit(lambda *a: jax_impl(name, cls, list(a)))
            t0 = time.perf_counter()
            jit_res = jitted(*xs_j)
            jit_res.block_until_ready()
            compile_ms = (time.perf_counter() - t0) * 1e3
            jit_ns = time_fn(
                lambda: jitted(*xs_j).block_until_ready(), iters
            )
            jax_rows["jit_compile_ms"] = compile_ms
            jax_rows["jit_ns"] = jit_ns
            agree_jit = {
                "bitexact": bitexact(np.asarray(jit_res), sx),
                "max_rel_err": max_rel_err(np.asarray(jit_res), sx),
            }

        # Exact ground truth where computable.
        exact = fsum_exact(cls, xs)
        exact_errs = None
        if exact is not None:
            exact_errs = {
                "sxla": max_rel_err(sx, exact),
                "numpy": max_rel_err(np_res, exact),
            }
            if have_jax:
                exact_errs["xla_jit"] = max_rel_err(np.asarray(jit_res), exact)

        # Tolerance gates.
        tol = TOL_ELEMENTWISE if cls == "elementwise" else TOL_SUMMATION
        for label, agree in (("numpy", agree_np), ("xla_jit", agree_jit)):
            if agree is not None and agree["max_rel_err"] > tol:
                print(
                    f"ref_xla: TOLERANCE VIOLATION {name} vs {label}: "
                    f"{agree['max_rel_err']:.3e} > {tol:.1e}",
                    file=sys.stderr,
                )
                failures.append(f"tolerance:{name}:{label}")

        np_ns = time_fn(lambda: numpy_impl(name, cls, xs), iters)
        row = {
            "name": name,
            "class": cls,
            "iters": iters,
            "numpy_ns": np_ns,
            "jax_eager_ns": jax_rows.get("eager_ns"),
            "jax_jit_ns": jax_rows.get("jit_ns"),
            "jax_jit_compile_ms": jax_rows.get("jit_compile_ms"),
            "sxla_ns": {
                "t0": k["tiers"]["0"]["exec_ns_mean"],
                "t1": k["tiers"]["1"]["exec_ns_mean"],
                "t2": k["tiers"]["2"]["exec_ns_mean"],
            },
            "sxla_compile_us": {
                "t0": k["tiers"]["0"]["compile_us"],
                "t1": k["tiers"]["1"]["compile_us"],
                "t2": k["tiers"]["2"]["compile_us"],
            },
            "agreement": {
                "inputs_bitexact": True,
                "sxla_vs_numpy": agree_np,
                "sxla_vs_xla_jit": agree_jit,
                "exact_errs": exact_errs,
            },
        }
        out_kernels.append(row)

        exact_txt = "-"
        if exact_errs:
            exact_txt = "/".join(
                f"{v:.2e}" for v in (
                    exact_errs["sxla"],
                    exact_errs["numpy"],
                    exact_errs.get("xla_jit", float("nan")),
                )
            )
        np_txt = ("bits" if agree_np["bitexact"] else "rel") + f":{agree_np['max_rel_err']:.2e}"
        if agree_jit:
            jit_txt = ("bits" if agree_jit["bitexact"] else "rel") + (
                f":{agree_jit['max_rel_err']:.2e}"
            )
        else:
            jit_txt = "-"
        print(
            f"{name:<16} {cls:<12} {'OK':<12} "
            f"{np_txt:<18} {jit_txt:<18} {exact_txt}"
        )

    result = {"env": env, "manifest_date": manifest.get("date"), "kernels": out_kernels}
    out_path = os.path.join(args.dir, "xla_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(f"results written to {out_path}")

    # Timing summary.
    print(
        f"\n{'kernel':<16} {'sxla t0':>10} {'sxla t2':>10} {'numpy':>10} "
        f"{'jax eager':>10} {'xla jit':>10} {'t2/jit':>8}"
    )
    for r in out_kernels:
        jit = r["jax_jit_ns"]
        ratio = (r["sxla_ns"]["t2"] / jit) if jit else float("nan")
        print(
            f"{r['name']:<16} {r['sxla_ns']['t0']:>10.0f} {r['sxla_ns']['t2']:>10.0f} "
            f"{r['numpy_ns']:>10.0f} "
            f"{(r['jax_eager_ns'] or float('nan')):>10.0f} "
            f"{(jit or float('nan')):>10.0f} {ratio:>8.1f}"
        )

    if failures:
        print(f"ref_xla: FAIL — {len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print("ref_xla: all input bit-checks and tolerance gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
