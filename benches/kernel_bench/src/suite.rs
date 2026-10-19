// CEP:FILE: benches/kernel_bench/src/suite.rs
// CEP:WHAT: The measured kernel suite + the deterministic LCG data protocol
//           shared bit-exactly with benches/kernel_bench/ref_xla.py.
// CEP:WHY: CEP&CC Law 4 requires an "input description" for every
//          measurement; cross-system value agreement additionally needs
//          BOTH sides to derive identical input bits from a documented
//          protocol (not from shipped files): the Python side re-generates
//          the stream and bit-checks it against the harness dumps, so a
//          disagreement can only come from compute, never from data.
// CEP:CLASS: CEP-2
// CEP:STATUS: complete
// CEP:FAILURE: constructors return Err on shape/value disagreement; tests
//              assert golden vectors.
// CEP:ASSUMES: little-endian target (docs/targets.md); IEEE 754 binary64.
// CEP:COST: suite construction is O(1); input generation is O(elements).
// CEP:EVIDENCE: unit tests below (LCG golden vector, per-class known
//           values incl. conv padding/stride conventions, all-tier
//           compilation, cross-tier bit agreement, CSE demonstrator);
//           docs/benchmarks.md.
// CEP:SECURITY: no untrusted input.
// CEP:HPC-DETERMINISM: the LCG protocol is bit-reproducible across
//           languages (wrapping u64 arithmetic + exact power-of-two
//           divisions only).

//! Kernel suite + deterministic data protocol.

use runtime::value::Value;
use xir_core::ty::Shape;

/// LCG multiplier (Knuth 64-bit constants).
pub const LCG_MUL: u64 = 6364136223846793005;
/// LCG increment.
pub const LCG_ADD: u64 = 1442695040888963407;
/// Per-param seed base (param i seeds with LCG_SEED0 + i).
pub const LCG_SEED0: u64 = 12345;

/// CEP:WHAT: Fills n deterministic f64 values in [-1, 1) from the LCG.
/// CEP:WHY: The documented data protocol (module header): s advances by
///          wrapping u64 arithmetic; v = ((s >> 33) / 2^31) * 2 - 1. Every
///          operation is exact in binary64 (power-of-two divide/multiply,
///          subtract one), so Rust and Python produce identical bits.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (total function).
/// CEP:ASSUMES: none.
/// CEP:COST: O(n).
/// CEP:EVIDENCE: test `lcg_golden_vector`.
/// CEP:SECURITY: no untrusted input.
/// CEP:HPC-DETERMINISM: bit-reproducible.
pub fn lcg_fill(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        let v = ((s >> 33) as f64 / 2147483648.0) * 2.0 - 1.0;
        out.push(v);
    }
    out
}

/// One measured kernel.
///
/// CEP:WHAT: Kernel definition: XIR source, param shapes, result shape,
///           fixed iteration count, abstract work units, dump flag.
/// CEP:WHY: Law 4 evidence needs the exact program + input shape + a fixed
///          measurement protocol per kernel (documented, not ad hoc).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: src parses (pinned by `suite_compiles_at_all_tiers`).
/// CEP:COST: O(1) construction.
/// CEP:EVIDENCE: suite tests.
/// CEP:SECURITY: no untrusted input.
pub struct KernelDef {
    /// Short name (also the dump file stem).
    pub name: String,
    /// Kernel class: elementwise | reduction | dot | matmul | conv.
    pub class: &'static str,
    /// XIR program text (params tensor-typed; ops result-typed).
    pub src: String,
    /// Per-param dims.
    pub param_dims: Vec<Vec<i64>>,
    /// Result dims ([] = scalar).
    pub result_dims: Vec<i64>,
    /// Timed iterations (fixed protocol; scaled by --scale).
    pub iters: u64,
    /// Abstract work units (elements for ew/reduce; FMA count for dot/
    /// matmul/conv) — the ns-per-work denominator.
    pub work: u64,
    /// Conv parameters (padding name, stride) when class == "conv"; the
    /// manifest carries them so ref_xla.py can mirror the semantics.
    pub conv: Option<(&'static str, u8)>,
    /// Write inputs + tier-0 result dumps for the Python reference side.
    pub dump: bool,
}

/// CEP:WHAT: relu(x * 0.5 + 0.25) over a [r,c] tensor — 3 elementwise passes.
/// CEP:STATUS: complete
fn ew_src(r: i64, c: i64) -> String {
    let mut s = String::new();
    s.push_str("xir v1 func @main {\n");
    s.push_str(&format!("  %0 = param 0 : tensor<f64>[{r},{c}]\n"));
    s.push_str("  %1 = const.f64 0.5\n");
    s.push_str(&format!(
        "  %2 = binary.mul %0, %1 : tensor<f64>[{r},{c}]\n"
    ));
    s.push_str("  %3 = const.f64 0.25\n");
    s.push_str(&format!(
        "  %4 = binary.add %2, %3 : tensor<f64>[{r},{c}]\n"
    ));
    s.push_str(&format!("  %5 = unary.relu %4 : tensor<f64>[{r},{c}]\n"));
    s.push_str("}\n");
    s
}

/// CEP:WHAT: (x*0.5) + (x*0.5) — the CSE demonstrator: Tier 0 runs two mul
///           passes, Tier 1+ merges them (GVN) into one, so the tier gap
///           is visible in execution time, not only in instr count.
/// CEP:STATUS: complete
fn cse_src(r: i64, c: i64) -> String {
    let mut s = String::new();
    s.push_str("xir v1 func @main {\n");
    s.push_str(&format!("  %0 = param 0 : tensor<f64>[{r},{c}]\n"));
    s.push_str("  %1 = const.f64 0.5\n");
    s.push_str(&format!(
        "  %2 = binary.mul %0, %1 : tensor<f64>[{r},{c}]\n"
    ));
    s.push_str(&format!(
        "  %3 = binary.mul %0, %1 : tensor<f64>[{r},{c}]\n"
    ));
    s.push_str(&format!(
        "  %4 = binary.add %2, %3 : tensor<f64>[{r},{c}]\n"
    ));
    s.push_str("}\n");
    s
}

/// CEP:WHAT: Sum over axis 0 of [r,c] -> [c] — sequential left-to-right
///           float fold (38.24: no reassociation).
/// CEP:STATUS: complete
fn reduce_src(r: i64, c: i64) -> String {
    let mut s = String::new();
    s.push_str("xir v1 func @main {\n");
    s.push_str(&format!("  %0 = param 0 : tensor<f64>[{r},{c}]\n"));
    s.push_str(&format!(
        "  %1 = reduce axis=0 monoid=add %0 : tensor<f64>[{c}]\n"
    ));
    s.push_str("}\n");
    s
}

/// CEP:WHAT: Vector dot product of two [n] tensors -> scalar.
/// CEP:STATUS: complete
fn dot_src(n: i64) -> String {
    let mut s = String::new();
    s.push_str("xir v1 func @main {\n");
    s.push_str(&format!("  %0 = param 0 : tensor<f64>[{n}]\n"));
    s.push_str(&format!("  %1 = param 1 : tensor<f64>[{n}]\n"));
    s.push_str("  %2 = dot %0, %1 : scalar<f64>\n");
    s.push_str("}\n");
    s
}

/// CEP:WHAT: Plain [n,n] x [n,n] matmul (ta=false tb=false).
/// CEP:STATUS: complete
fn mm_src(n: i64) -> String {
    let mut s = String::new();
    s.push_str("xir v1 func @main {\n");
    s.push_str(&format!("  %0 = param 0 : tensor<f64>[{n},{n}]\n"));
    s.push_str(&format!("  %1 = param 1 : tensor<f64>[{n},{n}]\n"));
    s.push_str(&format!(
        "  %2 = matmul ta=false tb=false %0, %1 : tensor<f64>[{n},{n}]\n"
    ));
    s.push_str("}\n");
    s
}

/// CEP:WHAT: Convolution kernel geometry: NCHW input [1,c,h,w], OIHW
///           filter [f,c,k,k], padding valid/same, stride s, expected
///           output [1,f,oh,ow] — one struct so the source builder stays
///           inside clippy's argument budget.
/// CEP:WHY: Law 4: every measured row states its exact shapes; grouping
///           them keeps the suite entries self-documenting.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: oh/ow match the interpreter's formula (pinned by the
///              conv known-value tests + the harness tier checks).
/// CEP:COST: O(1).
/// CEP:EVIDENCE: suite conv kernels + conv known-value tests.
/// CEP:SECURITY: no untrusted input.
struct ConvSpec {
    c: i64,
    h: i64,
    w: i64,
    f: i64,
    k: i64,
    padding: &'static str,
    stride: u8,
    oh: i64,
    ow: i64,
}

/// CEP:WHAT: NCHW x OIHW convolution (batch 1) from a ConvSpec — CEP-28's
///           kernel class.
/// CEP:WHY: The conv semantics are TF/XLA-conv compatible so the Python
///          reference can match them exactly (docs/benchmarks.md): valid
///          output size floor((h-kh)/s)+1 equals TF's ceil((h-kh+1)/s);
///          same pads (oh-1)*s+kh-h total, floor-split, odd pixel
///          bottom/right — identical to XLA conv_general_dilated with
///          explicit padding and ("NCHW","OIHW","NCHW") dimension numbers
///          (verified against XLA before the suite landed).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (string builder; parse/verify failures surface in
///              suite_compiles_at_all_tiers).
/// CEP:ASSUMES: spec.oh/ow match the interpreter's formula.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: conv known-value tests; docs/benchmarks.md conv rows.
/// CEP:SECURITY: no untrusted input.
fn conv_src(s: &ConvSpec) -> String {
    let mut out = String::new();
    out.push_str("xir v1 func @main {\n");
    out.push_str(&format!(
        "  %0 = param 0 : tensor<f64>[1,{},{},{}]\n",
        s.c, s.h, s.w
    ));
    out.push_str(&format!(
        "  %1 = param 1 : tensor<f64>[{},{},{},{}]\n",
        s.f, s.c, s.k, s.k
    ));
    out.push_str(&format!(
        "  %2 = conv padding={} stride={} %0, %1 : tensor<f64>[1,{},{},{}]\n",
        s.padding, s.stride, s.f, s.oh, s.ow
    ));
    out.push_str("}\n");
    out
}

/// CEP:WHAT: The full measured suite (fixed sizes + iteration protocol).
/// CEP:WHY: Law 4: every measured row must state its exact program, shapes
///           and iteration count. Sizes are chosen so the release-profile
///           full run stays in the tens of seconds; --scale 0.1 is the CI
///           smoke protocol (same values, fewer iterations).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (construction only).
/// CEP:ASSUMES: none.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: suite tests + docs/benchmarks.md.
/// CEP:SECURITY: no untrusted input.
pub fn suite() -> Vec<KernelDef> {
    let mut v: Vec<KernelDef> = Vec::new();
    // Elementwise chain: 3 passes over the tensor.
    for &(r, c, iters, dump) in &[
        (64i64, 64i64, 5000u64, true),
        (256, 256, 800, false),
        (1024, 1024, 60, false),
    ] {
        v.push(KernelDef {
            name: format!("ew_chain_{r}"),
            class: "elementwise",
            src: ew_src(r, c),
            param_dims: vec![vec![r, c]],
            result_dims: vec![r, c],
            iters,
            work: (r * c) as u64,
            conv: None,
            dump,
        });
    }
    // CSE demonstrator: tier gap visible in execution time.
    v.push(KernelDef {
        name: "ew_cse_256".to_string(),
        class: "elementwise",
        src: cse_src(256, 256),
        param_dims: vec![vec![256, 256]],
        result_dims: vec![256, 256],
        iters: 400,
        work: 256 * 256,
        conv: None,
        dump: true,
    });
    // Reduction: sum over axis 0.
    for &(r, c, iters, dump) in &[
        (64i64, 64i64, 5000u64, true),
        (256, 256, 800, false),
        (1024, 1024, 60, true),
    ] {
        v.push(KernelDef {
            name: format!("reduce_sum_{r}"),
            class: "reduction",
            src: reduce_src(r, c),
            param_dims: vec![vec![r, c]],
            result_dims: vec![c],
            iters,
            work: (r * c) as u64,
            conv: None,
            dump,
        });
    }
    // Dot product.
    for &(n, iters, dump) in &[
        (1024i64, 20000u64, true),
        (4096, 5000, false),
        (16384, 1000, true),
    ] {
        v.push(KernelDef {
            name: format!("dot_{n}"),
            class: "dot",
            src: dot_src(n),
            param_dims: vec![vec![n], vec![n]],
            result_dims: vec![],
            iters,
            work: n as u64,
            conv: None,
            dump,
        });
    }
    // Matmul (FMA work units).
    for &(n, iters, dump) in &[(64i64, 300u64, true), (128, 40, true), (256, 5, true)] {
        v.push(KernelDef {
            name: format!("matmul_{n}"),
            class: "matmul",
            src: mm_src(n),
            param_dims: vec![vec![n, n], vec![n, n]],
            result_dims: vec![n, n],
            iters,
            work: (n * n * n) as u64,
            conv: None,
            dump,
        });
    }
    // Convolution (CEP-28): NCHW x OIHW, batch 1, 3x3 kernels.
    // valid/s1 (c=8->f=16): work = f*oh*ow*c*9 = 1,036,800 FMAs.
    v.push(KernelDef {
        name: "conv_valid_32".to_string(),
        class: "conv",
        src: conv_src(&ConvSpec {
            c: 8,
            h: 32,
            w: 32,
            f: 16,
            k: 3,
            padding: "valid",
            stride: 1,
            oh: 30,
            ow: 30,
        }),
        param_dims: vec![vec![1, 8, 32, 32], vec![16, 8, 3, 3]],
        result_dims: vec![1, 16, 30, 30],
        iters: 50,
        work: 16 * 30 * 30 * 8 * 9,
        conv: Some(("valid", 1)),
        dump: true,
    });
    // valid/s2: same tensors, stride 2 -> 15x15 out (259,200 FMAs);
    // pins the stride output-size formula cross-system.
    v.push(KernelDef {
        name: "conv_valid_s2_32".to_string(),
        class: "conv",
        src: conv_src(&ConvSpec {
            c: 8,
            h: 32,
            w: 32,
            f: 16,
            k: 3,
            padding: "valid",
            stride: 2,
            oh: 15,
            ow: 15,
        }),
        param_dims: vec![vec![1, 8, 32, 32], vec![16, 8, 3, 3]],
        result_dims: vec![1, 16, 15, 15],
        iters: 400,
        work: 16 * 15 * 15 * 8 * 9,
        conv: Some(("valid", 2)),
        dump: true,
    });
    // same/s2 (c=4->f=8, 16x16 input): pad_total = 1 per axis, split
    // (0,1) — the odd pixel lands bottom/right; the sharp padding
    // convention test against XLA (18,432 FMAs).
    v.push(KernelDef {
        name: "conv_same_s2_16".to_string(),
        class: "conv",
        src: conv_src(&ConvSpec {
            c: 4,
            h: 16,
            w: 16,
            f: 8,
            k: 3,
            padding: "same",
            stride: 2,
            oh: 8,
            ow: 8,
        }),
        param_dims: vec![vec![1, 4, 16, 16], vec![8, 4, 3, 3]],
        result_dims: vec![1, 8, 8, 8],
        iters: 1500,
        work: 8 * 8 * 8 * 4 * 9,
        conv: Some(("same", 2)),
        dump: true,
    });
    v
}

/// CEP:WHAT: Bit-extraction of a runtime value (f64 bits; tensor as the
///           bit sequence of its row-major data).
/// CEP:WHY: Cross-tier agreement and cross-system agreement are checked at
///          the BIT level (0.0 vs -0.0 and NaN payloads must diverge).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (total function).
/// CEP:ASSUMES: none.
/// CEP:COST: O(elements).
/// CEP:EVIDENCE: `tiers_agree_bitwise_per_class`.
/// CEP:SECURITY: no untrusted input.
pub fn value_bits(v: &Value) -> Vec<u64> {
    match v {
        Value::F64(x) => vec![x.to_bits()],
        Value::I64(x) => vec![*x as u64],
        Value::Tensor { data, .. } => data.iter().map(|x| x.to_bits()).collect(),
    }
}

/// CEP:WHAT: Builds the LCG inputs for a kernel (param i seeds LCG_SEED0+i).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err on tensor construction disagreement (length vs shape).
/// CEP:ASSUMES: param_dims match the src annotations (pinned by tests).
/// CEP:COST: O(elements).
/// CEP:EVIDENCE: `lcg_golden_vector` + tier-agreement tests.
/// CEP:SECURITY: no untrusted input.
pub fn build_inputs(k: &KernelDef) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(k.param_dims.len());
    for (i, dims) in k.param_dims.iter().enumerate() {
        let shape = Shape::from_dims(dims).map_err(|e| format!("bad dims {dims:?}: {e:?}"))?;
        let n = shape.num_elements().max(0) as usize;
        let data = lcg_fill(LCG_SEED0 + i as u64, n);
        let v = Value::tensor(data, shape)
            .map_err(|e| format!("tensor construction failed for {}/p{}: {e:?}", k.name, i))?;
        out.push(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jit::driver::compile_text;
    use jit::tier::Tier;
    use runtime::interp::execute;

    fn tiers() -> [Tier; 3] {
        [Tier::Tier0, Tier::Tier1, Tier::Tier2]
    }

    /// CEP:WHAT: The LCG protocol golden vector (Python-derived, bit-pinned).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any protocol drift between languages.
    // CEP:ASSUMES: none.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test; ref_xla.py bit-checks the same protocol.
    #[test]
    fn lcg_golden_vector() {
        let v = lcg_fill(LCG_SEED0, 4);
        let bits: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            bits,
            vec![
                0xbfe8_fcaa_0400_0000,
                0xbfde_07da_c900_0000,
                0x3fe8_ae10_4100_0000,
                0x3fe5_7cb8_c280_0000,
            ]
        );
    }

    /// CEP:WHAT: LCG output stays in [-1, 1) and is reproducible.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on range or determinism drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn lcg_range_and_determinism() {
        let a = lcg_fill(999, 4096);
        let b = lcg_fill(999, 4096);
        assert_eq!(a, b);
        for x in &a {
            assert!(*x >= -1.0 && *x < 1.0, "value out of range: {x}");
        }
    }

    /// CEP:WHAT: Every suite kernel compiles at every tier (fingerprint
    ///           nonzero, tier echoed) — pins the tensor-param path.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if any program/tier pair stops compiling.
    // CEP:ASSUMES: none
    // CEP:COST: 13 kernels x 3 tiers (fast)
    // CEP:EVIDENCE: this test
    #[test]
    fn suite_compiles_at_all_tiers() {
        for k in suite() {
            for t in tiers() {
                let out = compile_text(&k.src, t);
                assert!(out.is_ok(), "{} tier {:?}: {:?}", k.name, t, out.err());
                if let Ok(p) = out {
                    assert_ne!(p.fingerprint, 0, "{}: zero fingerprint", k.name);
                    assert_eq!(p.tier, t);
                }
            }
        }
    }

    /// CEP:WHAT: Tensor helper for known-value tests (Result-returning: no
    ///           panic/unwrap paths anywhere in the crate).
    // CEP:STATUS: complete
    // CEP:FAILURE: Err on shape/length disagreement (test fails loudly).
    // CEP:ASSUMES: dims valid.
    // CEP:COST: test-only
    // CEP:EVIDENCE: callers below
    fn tensor(data: Vec<f64>, dims: &[i64]) -> Result<Value, String> {
        let shape = Shape::from_dims(dims).map_err(|e| format!("bad dims {dims:?}: {e:?}"))?;
        Value::tensor(data, shape).map_err(|e| format!("tensor construction: {e:?}"))
    }

    /// CEP:WHAT: Matmul known values (2x2 hand-checked).
    // CEP:STATUS: complete
    // CEP:FAILURE: Err/assert fires on wrong values or shapes.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn matmul_known_values_2x2() -> Result<(), String> {
        let a = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let c = compile_text(&mm_src(2), Tier::Tier0).map_err(|e| format!("{e:?}"))?;
        let out = execute(&c.target, &[a, b]).map_err(|e| format!("{e:?}"))?;
        assert_eq!(out.len(), 1);
        let Value::Tensor { data, .. } = &out[0] else {
            return Err(format!("expected tensor, got {:?}", out[0]));
        };
        assert_eq!(data, &vec![19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    /// CEP:WHAT: Reduce known values (2x3 axis-0 sums).
    // CEP:STATUS: complete
    // CEP:FAILURE: Err/assert fires on wrong values or shapes.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn reduce_known_values_2x3() -> Result<(), String> {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        let c = compile_text(&reduce_src(2, 3), Tier::Tier0).map_err(|e| format!("{e:?}"))?;
        let out = execute(&c.target, &[x]).map_err(|e| format!("{e:?}"))?;
        let Value::Tensor { data, .. } = &out[0] else {
            return Err(format!("expected tensor, got {:?}", out[0]));
        };
        assert_eq!(data, &vec![5.0, 7.0, 9.0]);
        Ok(())
    }

    /// CEP:WHAT: Dot known values ([1,2,3] . [4,5,6] = 32).
    // CEP:STATUS: complete
    // CEP:FAILURE: Err/assert fires on wrong values.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn dot_known_values() -> Result<(), String> {
        let a = tensor(vec![1.0, 2.0, 3.0], &[3])?;
        let b = tensor(vec![4.0, 5.0, 6.0], &[3])?;
        let c = compile_text(&dot_src(3), Tier::Tier0).map_err(|e| format!("{e:?}"))?;
        let out = execute(&c.target, &[a, b]).map_err(|e| format!("{e:?}"))?;
        assert_eq!(out[0], Value::F64(32.0));
        Ok(())
    }

    /// CEP:WHAT: Elementwise chain known values (incl. relu clamp to 0).
    // CEP:STATUS: complete
    // CEP:FAILURE: Err/assert fires on wrong values or shapes.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn ew_chain_known_values() -> Result<(), String> {
        let x = tensor(vec![1.0, -2.0, 0.5, -0.75], &[2, 2])?;
        let c = compile_text(&ew_src(2, 2), Tier::Tier0).map_err(|e| format!("{e:?}"))?;
        let out = execute(&c.target, &[x]).map_err(|e| format!("{e:?}"))?;
        let Value::Tensor { data, .. } = &out[0] else {
            return Err(format!("expected tensor, got {:?}", out[0]));
        };
        assert_eq!(data, &vec![0.75, 0.0, 0.5, 0.0]);
        Ok(())
    }

    /// CEP:WHAT: Conv valid/stride-2 known values: 5x5 input (1..25),
    ///           3x3 ones filter, floor((h-kh)/s)+1 = 2 windows per axis.
    ///           Values cross-verified against XLA (conv_general_dilated,
    ///           NCHW/OIHW, explicit zero padding) before pinning.
    // CEP:STATUS: complete
    // CEP:FAILURE: Err/assert fires on wrong values or shapes.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test + the XLA cross-check in docs/benchmarks.md
    #[test]
    fn conv_valid_s2_known_values() -> Result<(), String> {
        let x = tensor((1..=25i64).map(|i| i as f64).collect(), &[1, 1, 5, 5])?;
        let w = tensor(vec![1.0; 9], &[1, 1, 3, 3])?;
        let c = compile_text(
            &conv_src(&ConvSpec {
                c: 1,
                h: 5,
                w: 5,
                f: 1,
                k: 3,
                padding: "valid",
                stride: 2,
                oh: 2,
                ow: 2,
            }),
            Tier::Tier0,
        )
        .map_err(|e| format!("{e:?}"))?;
        let out = execute(&c.target, &[x, w]).map_err(|e| format!("{e:?}"))?;
        let Value::Tensor { data, .. } = &out[0] else {
            return Err(format!("expected tensor, got {:?}", out[0]));
        };
        assert_eq!(data, &vec![63.0, 81.0, 153.0, 171.0]);
        Ok(())
    }

    /// CEP:WHAT: Conv same/stride-2 known values: 4x4 input (1..16), 3x3
    ///           ones filter; pad_total = 1 per axis, split (0,1) — the
    ///           odd pixel lands bottom/right (the sharp convention
    ///           difference a top/left split would produce 68, not 54,
    ///           at position (1,0)). Values cross-verified against XLA
    ///           (explicit [(0,1),(0,1)] padding, which also matches
    ///           XLA's own 'SAME' string on this input).
    // CEP:STATUS: complete
    // CEP:FAILURE: Err/assert fires on wrong values, shapes, or a
    //              padding-split convention drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test + the XLA cross-check in docs/benchmarks.md
    #[test]
    fn conv_same_s2_known_values() -> Result<(), String> {
        let x = tensor((1..=16i64).map(|i| i as f64).collect(), &[1, 1, 4, 4])?;
        let w = tensor(vec![1.0; 9], &[1, 1, 3, 3])?;
        let c = compile_text(
            &conv_src(&ConvSpec {
                c: 1,
                h: 4,
                w: 4,
                f: 1,
                k: 3,
                padding: "same",
                stride: 2,
                oh: 2,
                ow: 2,
            }),
            Tier::Tier0,
        )
        .map_err(|e| format!("{e:?}"))?;
        let out = execute(&c.target, &[x, w]).map_err(|e| format!("{e:?}"))?;
        let Value::Tensor { data, .. } = &out[0] else {
            return Err(format!("expected tensor, got {:?}", out[0]));
        };
        assert_eq!(data, &vec![54.0, 45.0, 72.0, 54.0]);
        Ok(())
    }

    /// CEP:WHAT: All three tiers agree BITWISE on every kernel class
    ///           (elementwise, CSE shape, reduction, dot, matmul, conv) —
    ///           the in-harness differential regression.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any tier value divergence.
    // CEP:ASSUMES: none
    // CEP:COST: 6 programs x 3 tiers (small sizes)
    // CEP:EVIDENCE: this test
    #[test]
    fn tiers_agree_bitwise_per_class() {
        let cases: Vec<KernelDef> = vec![
            KernelDef {
                name: "ew8".into(),
                class: "elementwise",
                src: ew_src(8, 8),
                param_dims: vec![vec![8, 8]],
                result_dims: vec![8, 8],
                iters: 1,
                work: 64,
                conv: None,
                dump: false,
            },
            KernelDef {
                name: "cse8".into(),
                class: "elementwise",
                src: cse_src(8, 8),
                param_dims: vec![vec![8, 8]],
                result_dims: vec![8, 8],
                iters: 1,
                work: 64,
                conv: None,
                dump: false,
            },
            KernelDef {
                name: "red8".into(),
                class: "reduction",
                src: reduce_src(8, 8),
                param_dims: vec![vec![8, 8]],
                result_dims: vec![8],
                iters: 1,
                work: 64,
                conv: None,
                dump: false,
            },
            KernelDef {
                name: "dot8".into(),
                class: "dot",
                src: dot_src(8),
                param_dims: vec![vec![8], vec![8]],
                result_dims: vec![],
                iters: 1,
                work: 8,
                conv: None,
                dump: false,
            },
            KernelDef {
                name: "mm8".into(),
                class: "matmul",
                src: mm_src(8),
                param_dims: vec![vec![8, 8], vec![8, 8]],
                result_dims: vec![8, 8],
                iters: 1,
                work: 512,
                conv: None,
                dump: false,
            },
            KernelDef {
                name: "cvs2".into(),
                class: "conv",
                src: conv_src(&ConvSpec {
                    c: 1,
                    h: 4,
                    w: 4,
                    f: 1,
                    k: 3,
                    padding: "same",
                    stride: 2,
                    oh: 2,
                    ow: 2,
                }),
                param_dims: vec![vec![1, 1, 4, 4], vec![1, 1, 3, 3]],
                result_dims: vec![1, 1, 2, 2],
                iters: 1,
                work: 4 * 9,
                conv: Some(("same", 2)),
                dump: false,
            },
        ];
        for k in &cases {
            let inputs = build_inputs(k);
            assert!(inputs.is_ok(), "{}: {:?}", k.name, inputs.err());
            if let Ok(ins) = inputs {
                let mut baseline: Option<Vec<Vec<u64>>> = None;
                for t in tiers() {
                    let p = compile_text(&k.src, t);
                    assert!(p.is_ok(), "{} {:?}: {:?}", k.name, t, p.err());
                    if let Ok(c) = p {
                        let r = execute(&c.target, &ins);
                        assert!(r.is_ok(), "{} {:?}: {:?}", k.name, t, r.err());
                        if let Ok(out) = r {
                            let bits: Vec<Vec<u64>> = out.iter().map(value_bits).collect();
                            if baseline.is_none() {
                                baseline = Some(bits);
                            } else if let Some(base) = &baseline {
                                assert_eq!(
                                    &bits, base,
                                    "{}: tier {:?} diverged from tier 0",
                                    k.name, t
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// CEP:WHAT: The CSE demonstrator actually fires: Tier 1 emits fewer
    ///           instructions than Tier 0 on the duplicated-mul program.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if GVN stops merging the duplicate muls.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test (plus the harness timing row ew_cse_256)
    #[test]
    fn cse_demonstrator_shrinks_tier1() {
        let p0 = compile_text(&cse_src(8, 8), Tier::Tier0);
        let p1 = compile_text(&cse_src(8, 8), Tier::Tier1);
        assert!(p0.is_ok() && p1.is_ok());
        if let (Ok(c0), Ok(c1)) = (p0, p1) {
            assert!(
                c1.target.instrs.len() < c0.target.instrs.len(),
                "tier1 {} !< tier0 {}",
                c1.target.instrs.len(),
                c0.target.instrs.len()
            );
        }
    }
}
