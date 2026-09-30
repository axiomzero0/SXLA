# CEP&CC Waivers (34.7)

All waivers are temporary, owner-assigned, and carry expiry + review date.

| ID | Waiver | Owner | Expires | Justification |
|----|--------|-------|---------|---------------|
| W-1 | clippy::print_stdout/print_stderr in CEP-2 binaries (tools, bench harness) | main-agent | 2027-09 (review 2027-03) | Terminal output is the tool's purpose; lib crates keep the deny |
| W-2 | clippy::arithmetic_side_effects not denied globally | main-agent | 2027-09 (review 2027-03) | Overflow checks ON in all profiles + explicit bounds checks; hot-path wrap uses wrapping forms |
| W-3 | dyn Pass inside the HPC-1 pass manager | main-agent | 2027-09 (review 2027-03) | CEP&CC 38.3.2 classifies compile orchestration as HPC-1; passes themselves are monomorphic |
| W-4 | CI workflow (.github/workflows/ci.yml) present in the working tree but NOT committed/pushed — CEP:EVIDENCE fields citing the CI workflow describe the reference pipeline (docs/ci-workflow.reference.yml), not an executing CI | main-agent | 2026-12 (review 2026-11) | The pushing credential lacks `workflow` scope (GitHub rejects .github/workflows/ pushes and contents-API writes); committing the file locally would make main unpushable. Install via a workflow-scoped credential or the Actions web UI, then expire this waiver |
