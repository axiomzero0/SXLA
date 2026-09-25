// CEP:FILE: tools/xla-opt/src/main.rs
// CEP:WHAT: xla-opt — CLI tool for IR manipulation, pass inspection and
//           debugging (print, canonicalize, pipeline manifests).
// CEP:WHY: Master architecture project layout: "CLI tool for IR
//          manipulation and debugging". HPC-2 offline tooling: printing is
//          this binary's job, so print_stdout is allowed HERE (and only
//          here) with an explicit, documented waiver of the lib-code ban.
// CEP:CLASS: CEP-2
// CEP:STATUS: complete
// CEP:FAILURE: exits with code 1 and a diagnostic on parse/verify/pipeline
//              failure; exit 2 on I/O failure; exit 0 on success.
// CEP:ASSUMES: input files are repository-trusted UTF-8 (22.5); unknown
//              flags are rejected loudly (never silently ignored).
// CEP:COST: process startup + compile tier cost; offline tool.
// CEP:EVIDENCE: tools tested via workspace integration tests (tests/cli.rs).
// CEP:SECURITY: no network; bounded parsing; diagnostics carry byte
//              offsets, not file contents.
// CEP:HPC-DETERMINISM: output IR is byte-stable for identical input.
// CEP:WAIVER: print_stdout/print_stderr allowed in this CEP-2 binary
//              (justified: the tool's entire purpose is terminal output);
//              lib crates keep the deny.

// Waiver (see header): terminal output is the tool's purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::env;
use std::fs;
use std::process::ExitCode;

use anvil::telemetry::TelemetryBus;
use xir_core::text::print_arena;
use xir_levels::level1::layout_infer;
use xir_levels::passman::{CanonicalizePass, PassManager, WorkerContext};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    // Arg contract: xla-opt [--explain-pipeline] [--layout] FILE
    let mut explain = false;
    let mut layout = false;
    let mut file: Option<&str> = None;
    for a in args.iter().skip(1) {
        match a.as_str() {
            "--explain-pipeline" => explain = true,
            "--layout" => layout = true,
            "--help" | "-h" => {
                println!("xla-opt [--explain-pipeline] [--layout] FILE.xir");
                println!("  Prints the optimized IR (canonicalize: GVN + DCE).");
                println!("  --explain-pipeline  print the pass manifest");
                println!("  --layout            run layout inference before printing");
                return ExitCode::SUCCESS;
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("xla-opt: unknown flag {}", other);
                    return ExitCode::from(2);
                }
                if file.is_some() {
                    eprintln!("xla-opt: exactly one input file expected");
                    return ExitCode::from(2);
                }
                file = Some(other);
            }
        }
    }
    let Some(path) = file else {
        eprintln!("xla-opt: missing input file (try --help)");
        return ExitCode::from(2);
    };
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("xla-opt: cannot read {}: {}", path, e);
            return ExitCode::from(2);
        }
    };
    let (snap, roots) = match jit::driver::parse_and_verify(&src) {
        Ok(x) => x,
        Err(jit::driver::JitError::Parse(p)) => {
            eprintln!("xla-opt: parse error {:?} in {}", p, path);
            return ExitCode::from(1);
        }
        Err(jit::driver::JitError::Verify(v)) => {
            eprintln!("xla-opt: verification failed: {:?}", v);
            return ExitCode::from(1);
        }
        Err(e) => {
            eprintln!("xla-opt: {:?}", e);
            return ExitCode::from(1);
        }
    };
    if explain {
        // Manifest of the Tier-1 pipeline this tool applies.
        println!("pipeline: sxla-tier1-2026-09");
        println!(
            "passes: verify, canonicalize-l0 (gvn+dce){}",
            if layout { ", layout-infer" } else { "" }
        );
        println!("ir-version: {}", xir_core::IR_VERSION);
        println!("compiler: {}", xir_core::COMPILER_VERSION);
        return ExitCode::SUCCESS;
    }
    // Canonicalize through the pass manager (same path as Tier-1).
    let bus = TelemetryBus::new(1);
    let mut ctx = WorkerContext {
        worker_index: 0,
        workers: 1,
        telemetry: &bus,
    };
    let pm = PassManager::new(
        "sxla-tier1-2026-09",
        vec![Box::new(CanonicalizePass::new(roots.clone()))],
    );
    let optimized = match pm.run(&mut ctx, &snap) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("xla-opt: pipeline failed: {:?}", e);
            return ExitCode::from(1);
        }
    };
    // Print through the snapshot's arena view (the layout flag re-runs
    // layout inference on a working copy).
    if layout {
        let mut arena = optimized.arena().deep_clone();
        let _ = layout_infer(&mut arena);
        print!("{}", print_arena(&arena));
    } else {
        print!("{}", print_arena(optimized.arena()));
    }
    ExitCode::SUCCESS
}
