// CEP:FILE: tests/cli.rs
// CEP:WHAT: End-to-end integration tests — tool pipelines, tier equivalence,
//           and the SPSC JIT boundary flow.
// CEP:WHY: CEP&CC 38.45 (differential testing) and 38.43 (compiler testing
//          requirements): every tier must preserve Tier-0 semantics; the
//          CLI tools are the user-visible contract.
// CEP:CLASS: CEP-2 (tests)
// CEP:STATUS: complete
// CEP:FAILURE: asserts fire on any semantic divergence.
// CEP:ASSUMES: workspace binaries built (cargo test builds them).
// CEP:COST: test-only; full pipelines per case.
// CEP:EVIDENCE: this file.
// CEP:SECURITY: trusted fixture strings.
// CEP:HPC-DETERMINISM: deterministic fixtures and outputs.

use std::process::Command;

/// Fixture: (a+b) computed at all tiers must agree.
const ADD_SRC: &str =
    "xir v1 func @main {\n  %0 = param 0\n  %1 = param 1\n  %2 = binary.add %0, %1\n}\n";

/// Fixture with foldable constants and a dead node.
const CANON_SRC: &str = "xir v1 func @main {\n  %0 = const.f64 3.0\n  %1 = const.f64 4.0\n  %2 = binary.add %0, %1\n  %3 = unary.neg %2\n  %4 = binary.mul %2, %2\n}\n";

fn run_tool(tool: &str, args: &[&str]) -> (bool, String) {
    let bin = format!("{}/../target/debug/{}", env!("CARGO_MANIFEST_DIR"), tool);
    let out = Command::new(bin).args(args).output();
    match out {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout).to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            (o.status.success(), format!("{}{}", stdout, stderr))
        }
        Err(e) => (false, format!("spawn failed: {}", e)),
    }
}

fn write_fixture(name: &str, src: &str) -> String {
    let path = format!("/tmp/sxla-test-{}.xir", name);
    std::fs::write(&path, src).ok();
    path
}

// CEP:WHAT: xla-run executes at all tiers with identical results.
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on tier divergence.
// CEP:ASSUMES: debug binaries present.
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn tiers_agree_on_add() {
    let path = write_fixture("add", ADD_SRC);
    let mut results: Vec<String> = Vec::new();
    for tier in ["0", "1", "2"] {
        let (ok, out) = run_tool(
            "xla-run",
            &["--tier", tier, "--input", "20", "--input", "22", &path],
        );
        assert!(ok, "tier {} failed: {}", tier, out);
        let line = out.lines().find(|l| l.starts_with("result[0]="));
        assert!(line.is_some(), "tier {} printed no result: {}", tier, out);
        results.push(line.unwrap_or("").to_string());
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
    assert!(results[0].contains("42"));
}

// CEP:WHAT: xla-opt canonicalizes (folds constants, drops dead code).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires if canonicalization is lost.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn xla_opt_canonicalizes() {
    let path = write_fixture("canon", CANON_SRC);
    let (ok, out) = run_tool("xla-opt", &[&path]);
    assert!(ok, "xla-opt failed: {}", out);
    // The folded const 49.0 replaces the 3+4 chain; dead neg is removed.
    assert!(
        out.contains("const.f64 49.0"),
        "expected folded const in:\n{}",
        out
    );
    assert!(!out.contains("unary.neg"), "dead node survived:\n{}", out);
}

// CEP:WHAT: Bad input produces a nonzero exit and a diagnostic.
// CEP:STATUS: complete
// CEP:FAILURE: assert fires if garbage succeeds silently.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn bad_input_is_rejected() {
    let path = write_fixture("bad", "xir v1 func @main { %0 = nosuchop }");
    let (ok, out) = run_tool("xla-run", &[&path]);
    assert!(!ok);
    assert!(out.contains("parse error"));
    // Missing file: I/O failure path.
    let (ok2, out2) = run_tool("xla-run", &["/tmp/sxla-definitely-missing.xir"]);
    assert!(!ok2);
    assert!(out2.contains("cannot read"));
}

// CEP:WHAT: The SPSC JIT boundary flow compiles and caches across the
//           thread boundary (in-process driver-level check).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on boundary deadlock or loss.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test
#[test]
fn jit_boundary_flow() {
    use jit::boundary::{CompileRequest, JitBoundary};
    use jit::cache::{cache_key, JitCache, KernelEntry};
    use jit::driver::compile_text;
    use jit::tier::Tier;

    let compiled = compile_text(ADD_SRC, Tier::Tier1);
    assert!(compiled.is_ok());
    let compiled = match compiled {
        Ok(c) => c,
        Err(_) => return,
    };
    let cache = JitCache::new(16);
    assert!(cache.is_ok());
    let cache = match cache {
        Ok(c) => c,
        Err(_) => return,
    };
    let boundary = JitBoundary::new(16);
    let ok = std::thread::scope(|s| {
        // Compiler worker.
        let _worker = s.spawn(|| {
            let req = match boundary.pop_request() {
                Ok(r) => r,
                Err(_) => return false,
            };
            // Compile for the requested fingerprint (same source; the
            // driver is deterministic so this matches the request).
            let c = match compile_text(ADD_SRC, req.tier) {
                Ok(c) => c,
                Err(_) => return false,
            };
            let key = cache_key(c.fingerprint, &[], c.tier);
            let slot = match cache.register_reader() {
                Ok(sl) => sl,
                Err(_) => return false,
            };
            let entry = KernelEntry {
                program: c.target.clone(),
                tier: c.tier,
                results: c.results.clone(),
            };
            if cache.insert(key, entry, slot).is_err() {
                return false;
            }
            boundary
                .push_response(jit::boundary::CompileResponse {
                    kernel_key: key,
                    tier: c.tier,
                    failed: false,
                    _pad: 0,
                })
                .is_ok()
        });
        // Execution thread: request Tier-1 compilation.
        let req = CompileRequest {
            fingerprint: compiled.fingerprint,
            shape_layout: 0,
            tier: Tier::Tier1,
            _pad: 0,
        };
        if boundary.push_request(req).is_err() {
            return false;
        }
        for _ in 0..100_000 {
            match boundary.pop_response() {
                Ok(resp) => return !resp.failed,
                Err(anvil::spsc::QueueError::Empty) => std::thread::yield_now(),
                Err(_) => return false,
            }
        }
        false
    });
    assert!(ok, "boundary flow failed");
    assert_eq!(cache.len(), 1);
}

// CEP:WHAT: Differential check: fused Tier-2 equals unfused Tier-0 on a
//           chain program (the fusion search must not change semantics).
// CEP:STATUS: complete
// CEP:FAILURE: assert fires on semantic divergence.
// CEP:ASSUMES: none
// CEP:COST: test-only
// CEP:EVIDENCE: this test (CEP&CC 38.45 differential testing)
#[test]
fn differential_fusion_equivalence() {
    let src = "xir v1 func @main {\n  %0 = param 0\n  %1 = const.f64 2.0\n  %2 = binary.mul %0, %1\n  %3 = const.f64 1.0\n  %4 = binary.add %2, %3\n  %5 = unary.relu %4\n}\n";
    let path = write_fixture("chain", src);
    let (ok0, out0) = run_tool("xla-run", &["--tier", "0", "--input", "7", &path]);
    let (ok2, out2) = run_tool("xla-run", &["--tier", "2", "--input", "7", &path]);
    assert!(ok0 && ok2);
    let r0 = out0.lines().find(|l| l.starts_with("result[0]="));
    let r2 = out2.lines().find(|l| l.starts_with("result[0]="));
    assert!(r0.is_some() && r2.is_some());
    assert_eq!(r0, r2);
    // relu(7*2+1) = 15.
    assert!(r2.unwrap_or("").contains("15"));
}
