# CEP Lint Configuration

- Tool: `scripts/cep_lint.py` (CEP-2, deterministic, no network).
- Scope: `crates tools benches tests` (first-party only).
- Required per file: `CEP:FILE`, `CEP:WHAT`, `CEP:WHY`, `CEP:CLASS`,
  `CEP:STATUS`, `CEP:FAILURE`, `CEP:ASSUMES`, `CEP:COST`, `CEP:EVIDENCE`.
- Required on unsafe blocks: `CEP:UNSAFE` (safety word recognized by clippy
  via `undocumented_unsafe_blocks` + clippy.toml accept-comments settings).
- Required on TODOs: `CEP:TODO(owner): CEP-<n>: text` (10.10).
- Violations are reported with file/line and a severity class (34.2);
  the CI job fails on Severity >= 2.
