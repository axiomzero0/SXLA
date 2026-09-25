// CEP:FILE: crates/xir-graph/src/lib.rs
// CEP:WHAT: xir-graph — sea-of-nodes infrastructure: IR verifier, dominance,
//           GVN/CSE, DCE, canonical scheduling.
// CEP:WHY: Master architecture Level 0 ("Preserves mathematical semantics
//          and side-effect ordering via token edges") runs global
//          CSE/GVN/algebraic simplification on the pure sea-of-nodes via
//          Anvil Gear 1. CEP&CC 38.18 makes IR verification mandatory after
//          every HPC-0 pass — the verifier lives here and the pass manager
//          calls it at every commit.
// CEP:CLASS: CEP-0 (analysis passes) / CEP-1 (orchestration helpers)
// CEP:STATUS: complete
// CEP:FAILURE: VerifierError codes identify the exact broken invariant;
//           passes return PassError; no panics.
// CEP:ASSUMES: input snapshots passed a previous verify (frontends verify
//           after parsing — HPC-IR contract).
// CEP:COST: verifier O(nodes + edges); GVN O(nodes log nodes); DCE
//           O(nodes * uses) fixpoint; scheduler O(nodes + edges).
// CEP:EVIDENCE: per-module tests; workspace integration tests.
// CEP:SECURITY: verifier treats IR as untrusted (CEP&CC 38.13: internal
//           compiler errors on invalid input are Severity 1) — every check
//           is explicit and bounded.
// CEP:HPC-CLASS: HPC-0 (passes), HPC-1 (verifier driver).
// CEP:HPC-IR: Owns the Level-0 form checks.
// CEP:HPC-DETERMINISM: all passes iterate canonical slot order; results are
//           scheduling-independent (Gear-1 compatible).
//! # xir-graph
//!
//! Sea-of-nodes infrastructure for XIR Level 0.

pub mod dce;
pub mod dominance;
pub mod fold;
pub mod gvn;
pub mod schedule;
pub mod verifier;
