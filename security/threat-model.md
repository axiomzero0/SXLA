# SXLA Threat Model (CEP&CC 22.2)

## Assets

1. Correctness of compiled programs (miscompilation = Severity 0).
2. Build determinism and reproducibility of artifacts.
3. Host memory integrity during JIT compilation and execution.
4. Repository/toolchain integrity (supply chain).

## Trust boundaries

| Boundary | Trust | Controls |
|----------|-------|----------|
| .xir input files → parser | repository-trusted only (22.5) | bounded lexer, byte-offset diagnostics, no guessing; verifier gates the pipeline (38.18) |
| IR passes ↔ snapshots | internal | transactional commit + verify before publication |
| Worker threads ↔ Anvil primitives | internal | unsafe blocks documented + clippy-enforced safety comments; bounds-checked slot math |
| JIT cache reads (execution threads) | internal | EBR pin/unpin; lock-free reads never touch freed memory (two-epoch lag proof in anvil::ebr) |
| SPSC boundary | internal | POD-only messages; single-producer/single-consumer enforced by handle types |
| Telemetry | internal, best-effort | POD events, no strings/pointers/secrets; overflow counted, never blocking |

## Out of scope (this release)

- Executing untrusted .xir from the network (documented policy: not
  supported; inputs are repository files).
- GPU code generation (CEP-16) — no W^X surfaces exist yet; JIT patching
  policy (38.40) applies when it lands.
- Compiler plugins (38.41) — none loadable in this release.

## Supply chain

Zero external crate dependencies. `Cargo.lock` committed. Deterministic
build profile (no time/locale/env inputs). CI pins the toolchain via
rust-toolchain.toml.

## Secrets

The codebase contains no secrets. (Repository access tokens are operator
credentials, not code artifacts; none appear in sources.)
