# CEP&CC Waivers (34.7)

All waivers are temporary, owner-assigned, and carry expiry + review date.

| ID | Waiver | Owner | Expires | Justification |
|----|--------|-------|---------|---------------|
| W-1 | clippy::print_stdout/print_stderr in CEP-2 binaries (tools, bench harness) | main-agent | 2027-09 (review 2027-03) | Terminal output is the tool's purpose; lib crates keep the deny |
| W-2 | clippy::arithmetic_side_effects not denied globally | main-agent | 2027-09 (review 2027-03) | Overflow checks ON in all profiles + explicit bounds checks; hot-path wrap uses wrapping forms |
| W-3 | dyn Pass inside the HPC-1 pass manager | main-agent | 2027-09 (review 2027-03) | CEP&CC 38.3.2 classifies compile orchestration as HPC-1; passes themselves are monomorphic |
| W-4 | RESOLVED 2026-10-20 — .github/workflows/ci.yml committed and pushed with a workflow-capable credential; the CI gates are live. CEP:EVIDENCE fields citing the CI workflow describe the executing pipeline from that date. | main-agent | closed 2026-10-20 | Activation succeeded this session; the reference copy (docs/ci-workflow.reference.yml) documents the mirror and the keep-in-sync rule |
