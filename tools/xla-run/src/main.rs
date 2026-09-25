// CEP:FILE: tools/xla-run/src/main.rs
// CEP:WHAT: xla-run — the standalone JIT executor: compiles a .xir file at
//           a chosen tier and executes it with caller-supplied inputs.
// CEP:WHY: Master architecture project layout: "Standalone JIT executor".
//          Drives the full stack: parse -> verify -> tier pipeline ->
//          structurize -> lower -> interpret; prints results.
// CEP:CLASS: CEP-2
// CEP:STATUS: complete
// CEP:FAILURE: exit 1 on compile/execution failure; exit 2 on I/O or CLI
//              misuse; exit 0 on success.
// CEP:ASSUMES: trusted UTF-8 input; --input values are f64 scalars.
// CEP:COST: compilation tier cost + interpreter cost.
// CEP:EVIDENCE: tools tested via workspace integration tests (tests/cli.rs).
// CEP:SECURITY: no network; bounded parsing and execution.
// CEP:HPC-DETERMINISM: identical inputs produce identical outputs.
// CEP:WAIVER: print_stdout/print_stderr allowed in this CEP-2 binary
//              (terminal output is the tool's purpose).

// Waiver (see header).
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::env;
use std::fs;
use std::process::ExitCode;

use jit::driver::{compile_text, JitError};
use jit::tier::Tier;
use runtime::interp::execute;
use runtime::value::Value;
use xir_core::ty::Shape;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    // Arg contract: xla-run [--tier 0|1|2] [--input V]... FILE
    let mut tier = Tier::Tier1;
    let mut inputs: Vec<f64> = Vec::new();
    let mut file: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--tier" => {
                i += 1;
                let Some(t) = args.get(i) else {
                    eprintln!("xla-run: --tier requires a value (0|1|2)");
                    return ExitCode::from(2);
                };
                tier = match t.as_str() {
                    "0" => Tier::Tier0,
                    "1" => Tier::Tier1,
                    "2" => Tier::Tier2,
                    other => {
                        eprintln!("xla-run: unknown tier {}", other);
                        return ExitCode::from(2);
                    }
                };
            }
            "--input" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("xla-run: --input requires a value");
                    return ExitCode::from(2);
                };
                match v.parse::<f64>() {
                    Ok(x) => inputs.push(x),
                    Err(_) => {
                        eprintln!("xla-run: bad input value {:?}", v);
                        return ExitCode::from(2);
                    }
                }
            }
            "--help" | "-h" => {
                println!("xla-run [--tier 0|1|2] [--input V]... FILE.xir");
                println!("  Compiles FILE at the tier and executes it.");
                return ExitCode::SUCCESS;
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("xla-run: unknown flag {}", other);
                    return ExitCode::from(2);
                }
                if file.is_some() {
                    eprintln!("xla-run: exactly one input file expected");
                    return ExitCode::from(2);
                }
                file = Some(other.to_string());
            }
        }
        i += 1;
    }
    let Some(path) = file else {
        eprintln!("xla-run: missing input file (try --help)");
        return ExitCode::from(2);
    };
    let src = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("xla-run: cannot read {}: {}", path, e);
            return ExitCode::from(2);
        }
    };
    let compiled = match compile_text(&src, tier) {
        Ok(c) => c,
        Err(JitError::Parse(p)) => {
            eprintln!("xla-run: parse error {:?} in {}", p, path);
            return ExitCode::from(1);
        }
        Err(e) => {
            eprintln!("xla-run: compile failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    println!(
        "compiled: tier={} fingerprint={:016x} instrs={}",
        compiled.tier.code(),
        compiled.fingerprint,
        compiled.target.instrs.len()
    );
    // Bind scalar inputs (defaults 0.0).
    let empty_shape = Shape::scalar();
    let _ = empty_shape;
    // Only actual inputs bind parameters; the executor pads the rest of
    // the value table with zeros and instructions overwrite outputs.
    let values: Vec<Value> = inputs.iter().map(|v| Value::F64(*v)).collect();
    let results = match execute(&compiled.target, &values) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("xla-run: execution failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    for (i, r) in results.iter().enumerate() {
        match r {
            Value::F64(x) => println!("result[{}]= {:.6}", i, x),
            Value::I64(x) => println!("result[{}]= {}", i, x),
            Value::Tensor { data, shape } => {
                println!(
                    "result[{}]= tensor rank={} elems={} [{:.3} ...]",
                    i,
                    shape.rank(),
                    data.len(),
                    data.first().copied().unwrap_or(0.0)
                );
            }
        }
    }
    ExitCode::SUCCESS
}
