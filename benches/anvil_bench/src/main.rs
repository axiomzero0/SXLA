// CEP:FILE: benches/anvil_bench.rs
// CEP:WHAT: Benchmark harness for the Anvil primitives (measurement
//           evidence for CEP:COST claims).
// CEP:WHY: CEP&CC Law 4 ("no unmeasured hot code") and 13 (measurement
//          artifacts): every CEP-0 cost claim points here or to CI runs of
//          this harness. Implementation is a portable std::time harness
//          (criterion is deliberately absent: zero-dependency supply chain,
//          CEP&CC 22.11); the harness reports ns/op with warmups and the
//          target description (13.2).
// CEP:CLASS: CEP-2
// CEP:STATUS: complete
// CEP:FAILURE: exits nonzero on harness misuse; never panics on results.
// CEP:ASSUMES: run on the reference target described in docs/targets.md.
// CEP:COST: offline measurement; results are advisory evidence, not gates
//           (the CI gate wiring is CEP-21).
// CEP:EVIDENCE: `cargo run --release --bin anvil_bench` output archived in
//           .cep/evidence/bench-<date>.txt by CI.
// CEP:SECURITY: no untrusted input.
// CEP:HPC-DETERMINISM: measurements are wall-clock (advisory only; never
//           translation inputs).
// CEP:WAIVER: print_stdout allowed in this CEP-2 harness (output IS the
//           artifact; see .cep/waivers.md).

// Waiver (see header): terminal output is the harness artifact.
#![allow(clippy::print_stdout)]

use std::time::Instant;

fn bench_ns_per_op(name: &str, iters: u64, mut f: impl FnMut()) {
    // Warmup: 10% of iterations.
    let warm = iters / 10 + 1;
    for _ in 0..warm {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / iters as f64;
    println!("{:<24} {:>10.2} ns/op   ({} iters)", name, ns, iters);
}

fn main() {
    println!("sxla anvil bench — target: {}", bench_target());
    // Bump arena: alloc small structs.
    bench_ns_per_op("bump_alloc_u32", 10_000_000, || {
        let mut a = anvil::bump::BumpArena::new(1024 * 1024);
        for _ in 0..64 {
            let _ = a.alloc(7u32);
        }
    });
    // Chase-Lev push/pop (single thread).
    bench_ns_per_op("chase_lev_push_pop", 5_000_000, || {
        let d = anvil::chase_lev::Deque::new(8192);
        let _ = d.push(1u64);
        let _ = d.pop();
    });
    // SPSC push/pop.
    bench_ns_per_op("spsc_push_pop", 5_000_000, || {
        let r = anvil::spsc::SpscRing::new(1024);
        let (p, c) = r.split();
        let _ = p.push(1u64);
        let _ = c.pop();
    });
    // EBR pin/unpin.
    bench_ns_per_op("ebr_pin_unpin", 5_000_000, || {
        let c = anvil::ebr::Collector::new();
        let slot = c.register();
        if let Ok(s) = slot {
            let _ = c.pin(s);
        }
    });
    // FNV hashing (black_box keeps the result observable).
    bench_ns_per_op("fnv64_u64", 10_000_000, || {
        let mut h = xir_core::hash::Fnv64::new();
        h.write_u64(0xABCD_EF01_2345_6789);
        std::hint::black_box(h.finish());
    });
    // CEP-3 (Law 4 evidence): region setup cost — scoped spawns vs the
    // persistent pool, over MANY small Gear-1 regions. This is the exact
    // workload shape of per-round e-graph saturation and fusion universe
    // scoring (dozens of tiny regions per compilation).
    {
        let inputs: Vec<u64> = (0..256).collect();
        let mut outputs: Vec<u64> = vec![0; 256];
        let workers = anvil::default_worker_count().max(1);
        bench_ns_per_op("region_scoped", 2_000, || {
            let _ = anvil::run_partitioned_scoped(&inputs, &mut outputs, workers, |x| {
                x.wrapping_mul(3)
            });
            std::hint::black_box(outputs[255]);
        });
        bench_ns_per_op("region_pooled", 2_000, || {
            let _ = anvil::run_partitioned_pooled(&inputs, &mut outputs, workers, |x| {
                x.wrapping_mul(3)
            });
            std::hint::black_box(outputs[255]);
        });
    }
    println!("done.");
}

/// CEP:WHAT: Human-readable benchmark environment description (13.2).
/// CEP:STATUS: complete
fn bench_target() -> String {
    format!(
        "rustc {} {} {}",
        option_env!("CFG_RELEASE").unwrap_or("stable"),
        std::env::consts::ARCH,
        std::env::consts::OS
    )
}
