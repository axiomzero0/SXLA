// CEP:FILE: benches/kernel_bench/src/main.rs
// CEP:WHAT: kernel_bench — the cross-system kernel benchmark harness:
//           compiles the suite's XIR kernels at Tiers 0/1/2, executes them
//           with deterministic LCG tensor inputs, times compilation and
//           execution per tier, checks cross-tier BIT agreement, and (with
//           --out) dumps inputs/results plus a manifest.json for the
//           Python reference side (ref_xla.py: NumPy + JAX/XLA timings and
//           value agreement).
// CEP:WHY: CEP&CC Law 4 (no unmeasured hot code): the Tier-0 interpreter
//          kernels carried O(...) COST claims with no end-to-end
//          measurement; this harness is that measurement. Also pins the
//          tensor-parameter path (param : tensor<f64>[..]) end-to-end —
//          the xla-run CLI binds scalar inputs only, so this path was
//          compile-only-tested before this harness existed.
// CEP:CLASS: CEP-2
// CEP:STATUS: complete
// CEP:FAILURE: exit 2 on CLI misuse or debug-profile refusal (timings
//              meaningless); exit 1 on execution failure, dump/manifest
//              write failure, or any tier bit divergence; exit 0 on
//              success.
// CEP:ASSUMES: run via `cargo run --release --bin kernel_bench`; reference
//              target per docs/targets.md; wall-clock timings are advisory
//              evidence (never translation inputs); /proc/cpuinfo and
//              `rustc -V` are environment DESCRIPTION inputs only (13.2).
// CEP:COST: full suite ≈ tens of seconds in release (fixed iteration
//           protocol in suite.rs); --scale 0.1 is the CI smoke protocol.
//           One timed iteration = one full execute() call (value-table
//           setup incl. input cloning + every instruction) — the Tier-0
//           dispatch contract, measured as shipped.
// CEP:EVIDENCE: docs/benchmarks.md; .cep/evidence/kernel-bench-*.txt;
//           suite.rs unit tests (LCG golden vector, per-class known
//           values, tier bit agreement, CSE demonstrator).
// CEP:SECURITY: no network; files written only under --out.
// CEP:HPC-DETERMINISM: VALUE fields (dumps, checksums, fingerprints,
//           instr counts) are bit-reproducible; TIMING fields are
//           wall-clock advisory (13.2) and never feed translation
//           decisions.
// CEP:WAIVER: print_stdout/print_stderr allowed in this CEP-2 harness
//           (terminal output is the artifact; W-1 in .cep/waivers.md).

// Waiver (see header): terminal output is the harness artifact.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod suite;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use jit::driver::compile_text;
use jit::tier::Tier;
use runtime::interp::execute;
use runtime::value::Value;
use suite::{build_inputs, suite, value_bits, LCG_ADD, LCG_MUL, LCG_SEED0};
use xir_core::hash::Fnv64;

/// CEP:WHAT: JSON string escaping (minimal, control-char safe).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (total function).
/// CEP:ASSUMES: none
/// CEP:COST: O(len)
/// CEP:EVIDENCE: manifest round-trips through Python's json module.
fn jesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// CEP:WHAT: Flat f64 view of a runtime value for dumps.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(elements)
/// CEP:EVIDENCE: ref_xla.py reads the dumps bit-exactly.
fn flat_f64s(v: &Value) -> Vec<f64> {
    match v {
        Value::F64(x) => vec![*x],
        Value::I64(x) => vec![*x as f64],
        Value::Tensor { data, .. } => data.clone(),
    }
}

/// CEP:WHAT: Writes raw little-endian f64s (docs/targets.md endianness).
/// CEP:STATUS: complete
/// CEP:FAILURE: Err(String) on I/O failure.
/// CEP:ASSUMES: none
/// CEP:COST: O(elements)
/// CEP:EVIDENCE: ref_xla.py np.fromfile(dtype='<f8').
fn write_f64s(path: &Path, vals: &[f64]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(vals.len() * 8);
    for v in vals {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

/// CEP:WHAT: FNV-64 digest of a result's bit stream (quick integrity eyeball).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: O(elements)
/// CEP:EVIDENCE: manifest checksums; the dumps themselves are the evidence.
fn fnv_bits(vals: &[Vec<u64>]) -> u64 {
    let mut h = Fnv64::new();
    for v in vals {
        for b in v {
            h.write_u64(*b);
        }
    }
    h.finish()
}

/// CEP:WHAT: CPU model string (/proc/cpuinfo, best effort).
/// CEP:STATUS: complete
/// CEP:FAILURE: "unknown" off-Linux or on parse failure.
/// CEP:ASSUMES: description input only (never a translation input).
/// CEP:COST: one file read
/// CEP:EVIDENCE: manifest.cpu
fn cpu_model() -> String {
    if let Ok(txt) = fs::read_to_string("/proc/cpuinfo") {
        for line in txt.lines() {
            if let Some((k, v)) = line.split_once(':') {
                if k.trim() == "model name" {
                    return v.trim().to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

/// CEP:WHAT: rustc version (`rustc -V`, best effort).
/// CEP:STATUS: complete
/// CEP:FAILURE: "unknown" when rustc is absent from PATH.
/// CEP:ASSUMES: description input only.
/// CEP:COST: one subprocess
/// CEP:EVIDENCE: manifest.rustc (Law 4: compiler version).
fn rustc_version() -> String {
    if let Ok(out) = std::process::Command::new("rustc").arg("-V").output() {
        if out.status.success() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                return s.trim().to_string();
            }
        }
    }
    "unknown".to_string()
}

/// CEP:WHAT: Artifact date stamp: --date arg > SOURCE_DATE_EPOCH > wall now.
/// CEP:STATUS: complete
/// CEP:FAILURE: "unknown" (pre-epoch clock).
/// CEP:ASSUMES: none
/// CEP:COST: O(1)
/// CEP:EVIDENCE: manifest.date (Law 4: date or artifact ID).
fn date_stamp(arg: Option<&str>) -> String {
    if let Some(s) = arg {
        return s.to_string();
    }
    if let Ok(d) = std::env::var("SOURCE_DATE_EPOCH") {
        if !d.is_empty() {
            return format!("epoch:{d}");
        }
    }
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("epoch:{}", d.as_secs()),
        Err(_) => "unknown".to_string(),
    }
}

/// CEP:WHAT: Per-tier measurement record (one kernel, one tier).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data)
/// CEP:ASSUMES: single-shot compile; mean over the timed loop.
/// CEP:COST: n/a
/// CEP:EVIDENCE: manifest tier objects.
struct TierRow {
    /// Tier code (0/1/2).
    code: u64,
    /// Single-shot compile wall time, microseconds.
    compile_us: f64,
    /// Lowered instruction count.
    instrs: usize,
    /// Program fingerprint (hex).
    fingerprint: String,
    /// Mean execution wall time per execute() call, nanoseconds.
    exec_ns_mean: f64,
}

fn main() -> ExitCode {
    if cfg!(debug_assertions) {
        eprintln!(
            "kernel_bench: refusing to time a debug build (timings are meaningless); \
             rebuild with --release"
        );
        return ExitCode::from(2);
    }
    // Arg contract: kernel_bench [--out DIR] [--scale F] [--date TEXT]
    let mut out_dir: Option<PathBuf> = None;
    let mut scale: f64 = 1.0;
    let mut date: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--out" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("kernel_bench: --out requires a directory path");
                    return ExitCode::from(2);
                };
                out_dir = Some(PathBuf::from(v));
            }
            "--scale" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("kernel_bench: --scale requires a positive factor");
                    return ExitCode::from(2);
                };
                match v.parse::<f64>() {
                    Ok(f) if f.is_finite() && f > 0.0 => scale = f,
                    _ => {
                        eprintln!("kernel_bench: bad --scale value {:?}", v);
                        return ExitCode::from(2);
                    }
                }
            }
            "--date" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("kernel_bench: --date requires a value");
                    return ExitCode::from(2);
                };
                date = Some(v.clone());
            }
            "--help" | "-h" => {
                println!("kernel_bench [--out DIR] [--scale F] [--date TEXT]");
                println!("  Times the kernel suite at Tiers 0/1/2 (release profile).");
                println!("  --out DIR   also write manifest.json + input/result dumps there");
                println!("  --scale F   scale all iteration counts by F (CI smoke: 0.1)");
                println!("  --date TEXT artifact date stamp (else SOURCE_DATE_EPOCH/now)");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("kernel_bench: unknown flag {}", other);
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    if let Some(dir) = &out_dir {
        if let Err(e) = fs::create_dir_all(dir) {
            eprintln!("kernel_bench: cannot create {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    }
    let date = date_stamp(date.as_deref());
    let cpu = cpu_model();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let rustc = rustc_version();
    println!(
        "sxla kernel bench — {} | {} | {} {} | cores={}",
        rustc,
        cpu,
        std::env::consts::ARCH,
        std::env::consts::OS,
        cores
    );
    println!("date: {} | profile: release per Cargo.toml (lto, cgu=1, overflow-checks on; debug_assertions off — guard passed)", date);
    println!(
        "{:<16} {:<12} {:>10} {:>4} {:>6} {:>10} {:>12} {:>9}",
        "kernel", "class", "work", "tier", "instrs", "compile_us", "exec_ns_mean", "ns/work"
    );

    let kernels = suite();
    let mut kernel_json: Vec<String> = Vec::new();
    let mut divergence = false;
    for k in &kernels {
        let inputs = match build_inputs(k) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("kernel_bench: {}: {e}", k.name);
                return ExitCode::from(1);
            }
        };
        let iters = ((k.iters as f64) * scale).round().max(1.0) as u64;
        let warm = iters / 10 + 1;
        let mut rows: Vec<TierRow> = Vec::new();
        let mut baseline_bits: Option<Vec<Vec<u64>>> = None;
        let mut first_result: Option<Vec<Value>> = None;
        for t in [Tier::Tier0, Tier::Tier1, Tier::Tier2] {
            let t0 = Instant::now();
            let compiled = match compile_text(&k.src, t) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("kernel_bench: {}: tier {:?} compile: {e:?}", k.name, t);
                    return ExitCode::from(1);
                }
            };
            let compile_us = t0.elapsed().as_nanos() as f64 / 1000.0;
            let exec = match execute(&compiled.target, &inputs) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("kernel_bench: {}: tier {:?} exec: {e:?}", k.name, t);
                    return ExitCode::from(1);
                }
            };
            let bits: Vec<Vec<u64>> = exec.iter().map(value_bits).collect();
            match &baseline_bits {
                None => {
                    baseline_bits = Some(bits);
                    first_result = Some(exec);
                }
                Some(base) => {
                    if &bits != base {
                        eprintln!(
                            "kernel_bench: TIER DIVERGENCE: {} tier {:?} bits differ from tier 0",
                            k.name, t
                        );
                        divergence = true;
                    }
                }
            }
            // Timed loop: warmup then mean (house protocol, anvil_bench).
            for _ in 0..warm {
                let r = execute(&compiled.target, &inputs);
                std::hint::black_box(&r);
            }
            let start = Instant::now();
            for _ in 0..iters {
                let r = execute(&compiled.target, &inputs);
                std::hint::black_box(&r);
            }
            let mean_ns = start.elapsed().as_nanos() as f64 / iters as f64;
            rows.push(TierRow {
                code: t.code(),
                compile_us,
                instrs: compiled.target.instrs.len(),
                fingerprint: format!("{:016x}", compiled.fingerprint),
                exec_ns_mean: mean_ns,
            });
        }
        let ns_per_work = |row: &TierRow| row.exec_ns_mean / k.work.max(1) as f64;
        for row in &rows {
            println!(
                "{:<16} {:<12} {:>9}{} {:>4} {:>6} {:>10.1} {:>12.0} {:>9.3}",
                k.name,
                k.class,
                k.work,
                if k.class == "matmul" { 'f' } else { 'e' },
                row.code,
                row.instrs,
                row.compile_us,
                row.exec_ns_mean,
                ns_per_work(row),
            );
        }
        // Dumps + JSON fragment.
        let mut param_json: Vec<String> = Vec::new();
        if k.dump {
            if let (Some(dir), Some(res)) = (&out_dir, &first_result) {
                for (pi, v) in inputs.iter().enumerate() {
                    let fname = format!("in_{}_p{}.f64", k.name, pi);
                    let flat = flat_f64s(v);
                    if let Err(e) = write_f64s(&dir.join(&fname), &flat) {
                        eprintln!("kernel_bench: {e}");
                        return ExitCode::from(1);
                    }
                    param_json.push(format!(
                        "{{\"dims\":{:?},\"elems\":{},\"file\":{}}}",
                        k.param_dims[pi],
                        flat.len(),
                        jesc(&fname)
                    ));
                }
                let rname = format!("res_{}.f64", k.name);
                if let Err(e) = write_f64s(&dir.join(&rname), &flat_f64s(&res[0])) {
                    eprintln!("kernel_bench: {e}");
                    return ExitCode::from(1);
                }
            }
        } else if out_dir.is_some() {
            for dims in k.param_dims.iter() {
                param_json.push(format!("{{\"dims\":{:?},\"file\":null}}", dims));
            }
        }
        let result_file = if k.dump && out_dir.is_some() {
            jesc(&format!("res_{}.f64", k.name))
        } else {
            "null".to_string()
        };
        let result_elems = flat_f64s(
            first_result
                .as_ref()
                .and_then(|r| r.first())
                .unwrap_or(&Value::F64(0.0)),
        )
        .len();
        let checksum = fnv_bits(&baseline_bits.unwrap_or_default());
        let tiers_json: Vec<String> = rows
            .iter()
            .map(|r| {
                format!(
                    "{{\"compile_us\":{:.1},\"instrs\":{},\"fingerprint\":{},\"exec_ns_mean\":{:.1},\"ns_per_work\":{:.4}}}",
                    r.compile_us,
                    r.instrs,
                    jesc(&r.fingerprint),
                    r.exec_ns_mean,
                    ns_per_work(r)
                )
            })
            .collect();
        kernel_json.push(format!(
            "{{\"name\":{},\"class\":{},\"iters\":{},\"work\":{},\"work_kind\":{},\"dump\":{},\"params\":[{}],\"result\":{{\"dims\":{:?},\"elems\":{},\"file\":{}}},\"tiers\":{{\"0\":{},\"1\":{},\"2\":{}}},\"checksum\":\"{:016x}\"}}",
            jesc(&k.name),
            jesc(k.class),
            iters,
            k.work,
            jesc(if k.class == "matmul" { "fma" } else { "elem" }),
            k.dump,
            param_json.join(","),
            k.result_dims,
            result_elems,
            result_file,
            tiers_json.first().cloned().unwrap_or_default(),
            tiers_json.get(1).cloned().unwrap_or_default(),
            tiers_json.get(2).cloned().unwrap_or_default(),
            checksum
        ));
    }
    if divergence {
        eprintln!("kernel_bench: FAIL — tier bit divergence detected (see above)");
        return ExitCode::from(1);
    }
    if let Some(dir) = &out_dir {
        let manifest = format!(
            "{{\"harness\":\"sxla-kernel-bench\",\"version\":\"0.1.0\",\"date\":{},\"profile\":{},\"rustc\":{},\"cpu\":{},\"cores\":{},\"lcg\":{{\"mul\":{},\"add\":{},\"seed0\":{},\"formula\":{}}},\"kernels\":[{}]}}\n",
            jesc(&date),
            jesc("release (lto, codegen-units=1, overflow-checks=true; debug_assertions off)"),
            jesc(&rustc),
            jesc(&cpu),
            cores,
            jesc(&LCG_MUL.to_string()),
            jesc(&LCG_ADD.to_string()),
            jesc(&LCG_SEED0.to_string()),
            jesc("v = ((s >> 33) / 2^31) * 2 - 1; per-param seed = seed0 + i; wrapping u64"),
            kernel_json.join(",")
        );
        if let Err(e) = fs::write(dir.join("manifest.json"), manifest) {
            eprintln!("kernel_bench: manifest write: {e}");
            return ExitCode::from(1);
        }
        println!("manifest + dumps written to {}", dir.display());
    }
    println!("done. all tiers agree bitwise across the suite.");
    ExitCode::SUCCESS
}
