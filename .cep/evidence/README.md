# Evidence Register (CEP&CC 10.9 / 13.1)

- Unit/integration tests: `cargo test --workspace` (279 tests at time of
  writing; CI log is the canonical artifact) plus the sanitizer matrix
  (ASan; TSan with -Zbuild-std, halt_on_error=1 — hard gates since
  session 6).
- Benchmarks: ALWAYS-ON since session 6 — `.github/workflows/bench.yml`
  runs anvil_bench + kernel_bench (smoke scale) on every push to main, every
  PR, daily, and on dispatch, archiving per-run artifacts (90-day retention);
  `scripts/bench.sh` reproduces the same evidence locally. Cross-system
  kernel bench reference runs remain archived as dated files
  (kernel-bench-2026-10-19.txt, kernel-bench-2026-10-20.txt — see
  docs/benchmarks.md).
- Determinism: `tests/cli.rs::tiers_agree_on_add` and
  `differential_fusion_equivalence` (38.45 differential testing).
- Unsafe-audit: clippy `undocumented_unsafe_blocks = deny` (mechanical
  enforcement of CEP:UNSAFE blocks).
- Header/field lint: `python3 scripts/cep_lint.py crates tools benches tests`.
