// CEP:FILE: crates/jit/src/lib.rs
// CEP:WHAT: jit — the tiered dispatcher, EBR-sharded kernel cache, SPSC
//           compilation boundary, and the full pipeline driver.
// CEP:WHY: Master architecture section 6: the JIT is isolated from
//          execution threads (SPSC boundary), dispatches tiers 0-3, and
//          caches kernels in the EBR-sharded JIT cache; section 8 Tiered
//          Compilation defines the latency budgets.
// CEP:CLASS: CEP-1 (orchestration) / CEP-0 (cache reads)
// CEP:STATUS: partial
// CEP:FAILURE: JitError codes; conservative Tier-0 fallback on any
//             compilation failure (never a wrong result).
// CEP:ASSUMES: text IR files are repository-trusted (22.5).
// CEP:COST: Tier-1 < 5ms budget on reference graphs (documented; the
//           benchmark harness is CEP-21).
// CEP:EVIDENCE: per-module tests; xla-run end-to-end; differential tests.
// CEP:SECURITY: untrusted IR validated by the parser + verifier; cache
//             bounded by insertion policy (eviction = remove, EBR-safe).
// CEP:HPC-CLASS: HPC-1 (dispatcher), HPC-0 (cache get path).
// CEP:HPC-DETERMINISM: deterministic compilation; cache hits return the
//             SAME kernel the deterministic pipeline would rebuild.
// CEP:TODO(main-agent): CEP-25: Tier-3 PGO recompilation from hardware
//             counters; speculative compilation of likely shapes.
//! # jit
//!
//! Tiered compilation with the SPSC boundary and the EBR kernel cache.

pub mod boundary;
pub mod cache;
pub mod driver;
pub mod tier;

pub use boundary::{CompileRequest, CompileResponse, JitBoundary};
pub use cache::{JitCache, KernelEntry};
pub use driver::{compile, compile_text, CompiledProgram};
pub use tier::{Tier, TIER1_BUDGET_MS};
