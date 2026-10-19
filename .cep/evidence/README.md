# Evidence Register (CEP&CC 10.9 / 13.1)

- Unit/integration tests: `cargo test --workspace` (143 tests at time of
  writing; CI log is the canonical artifact).
- Benchmarks: `benches/anvil_bench` output archived per CI run
  (bench-<sha>.txt); cross-system kernel bench evidence archived per
  reference run (kernel-bench-2026-10-19.txt, kernel-bench-2026-10-20.txt —
  see docs/benchmarks.md).
- Determinism: `tests/cli.rs::tiers_agree_on_add` and
  `differential_fusion_equivalence` (38.45 differential testing).
- Unsafe-audit: clippy `undocumented_unsafe_blocks = deny` (mechanical
  enforcement of CEP:UNSAFE blocks).
- Header/field lint: `python3 scripts/cep_lint.py crates tools benches tests`.
