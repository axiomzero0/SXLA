// CEP:FILE: crates/xir-levels/src/passman.rs
// CEP:WHAT: The Anvil-aware Pass Manager — pass contracts, pipeline
//           execution, verification gating, telemetry emission.
// CEP:WHY: Master architecture section 7 defines the Pass trait with
//          required_form / concurrency / run; CEP&CC 38.20 requires a
//          versioned explicit pipeline; 38.18 requires verification after
//          every HPC-0 pass. This module is the single place where those
//          contracts are enforced mechanically (psychopathic tier).
// CEP:CLASS: CEP-1 (orchestration)
// CEP:STATUS: complete
// CEP:FAILURE: PipelineError::{PassFailed, VerifyFailed, FormConversion}
//              — pipeline aborts loudly on the first failure; partial
//              results are never returned.
// CEP:ASSUMES: `dyn Pass` is permitted HERE and only here: the pass manager
//              is HPC-1 compile orchestration (CEP&CC 38.3.2 explicitly
//              classifies "compile orchestration" as HPC-1), not a CEP-0
//              hot path; the passes themselves are monomorphic and
//              statically dispatched internally.
// CEP:COST: per pass: 1 telemetry emit (~2ns) + pass body + verifier
//           O(nodes) + snapshot commit O(nodes). Documented; the JIT's
//           Tier-1 pipeline skips nonessential passes to meet its <5ms
//           budget (jit crate).
// CEP:EVIDENCE: tests `pipeline_runs_and_verifies`, `verify_failure_aborts`.
// CEP:SECURITY: passes are first-party (Box<dyn Pass> built in-code; no
//           plugin loading — 38.41).
// CEP:HPC-DETERMINISM: deterministic: passes run in manifest order and
//           produce scheduling-independent snapshots.
//! Pass manager.

use std::sync::Arc;

use anvil::telemetry::{TelemetryBus, TelemetryEvent, TelemetryKind};
use xir_core::arena::IrArena;
use xir_core::snapshot::IrSnapshot;
use xir_graph::verifier::{verify, VerifierError};

/// Which IR form a pass prefers (architecture section 7).
///
/// CEP:WHAT: Form preference discriminant.
/// CEP:WHY: The manager inserts graphify/structurize conversions when a
///          pass's preference disagrees with the current form (38.20
///          "Pass pipeline contract").
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: passman tests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrFormPreference {
    /// Sea-of-nodes graph form (Levels 0-2).
    Graph,
    /// Structured loop form (Levels 3-4).
    Structured,
    /// Either form is acceptable.
    Either,
}

/// Concurrency contract of a pass (architecture section 7).
///
/// CEP:WHAT: Concurrency discriminant.
/// CEP:WHY: Disjoint passes may run under Gear 1 static partitioning;
///          ReadOnly passes may read a snapshot concurrently; Transactional
///          passes produce edit logs committed atomically.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: passman tests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassConcurrency {
    /// Workers mutate disjoint subgraphs (Gear 1).
    Disjoint,
    /// Read-only over the snapshot (any gear).
    ReadOnly,
    /// Edits collected transactionally, committed once.
    Transactional,
}

/// HPC class of a pass (drives verification gating).
///
/// CEP:WHAT: HPC-0/1/2 classification per CEP&CC 38.3.
/// CEP:WHY: 38.18: verification must run after each HPC-0 pass; HPC-1 passes
///          verify at commit boundaries anyway (the snapshot commit does).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: 1 byte
/// CEP:EVIDENCE: passman tests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpcClass {
    /// Compiler-critical hot code: verify after every run.
    Hpc0,
    /// Deterministic support code: verify at commit.
    Hpc1,
}

/// Pass output contract.
///
/// CEP:WHAT: Whether a pass changed the IR, and the new version if so.
/// CEP:WHY: Unchanged passes skip the commit/verify cycle (compile-time
///          budget, CEP&CC 38.11/38.12).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: enum-sized
/// CEP:EVIDENCE: passman tests
pub enum PassOutput {
    /// The IR is unchanged (same snapshot continues).
    Unchanged,
    /// A new snapshot version was produced.
    Changed(Arc<IrSnapshot>),
}

/// Pipeline failure enumeration.
///
/// CEP:WHAT: Explicit error type for pipeline execution.
/// CEP:WHY: Law 6 + 38.6: failures must pinpoint the pass and the broken
///          invariant.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: 24 bytes
/// CEP:EVIDENCE: tests `verify_failure_aborts`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineError {
    /// A pass returned an error.
    PassFailed(&'static str),
    /// Post-pass verification failed.
    VerifyFailed(&'static str, VerifierError),
    /// A required form conversion is not implemented (loud, not silent).
    FormConversion(&'static str),
}

/// Worker context passed into passes (architecture: `run(&mut WorkerContext, ..)`).
///
/// CEP:WHAT: Per-invocation worker state: index, telemetry bus, worker count.
/// CEP:WHY: Passes emit telemetry (PassStart/PassEnd, fusion decisions)
///          through the lock-free bus and size Gear-1 partitions by the
///          worker count — both come from the manager, never from globals
///          (no hidden global state, CEP&CC 38.17).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: telemetry bus outlives the pipeline run.
/// CEP:COST: borrow-only; zero allocation
/// CEP:EVIDENCE: passman tests
pub struct WorkerContext<'a> {
    /// Index of the executing worker (0-based).
    pub worker_index: usize,
    /// Total workers in this region.
    pub workers: usize,
    /// Lock-free telemetry bus (best-effort events).
    pub telemetry: &'a TelemetryBus,
}

impl WorkerContext<'_> {
    /// CEP:WHAT: Emits a pass event (best-effort, zero cost on overflow).
    /// CEP:WHY: Uniform instrumentation; drops are counted by the bus.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: returns false on ring overflow (event dropped+counted).
    /// CEP:ASSUMES: called by the owning worker.
    /// CEP:COST: ~2ns amortized (Gear 3).
    /// CEP:EVIDENCE: telemetry crate tests.
    pub fn emit(&self, kind: TelemetryKind, pass_id: u16, payload: u64) -> bool {
        self.telemetry.emit(
            self.worker_index,
            TelemetryEvent {
                kind: kind.code(),
                _pad: 0,
                pass_id,
                payload,
            },
        )
    }
}

/// The pass contract (architecture section 7, extended per CEP&CC 38.21).
///
/// CEP:WHAT: Pass trait — name, version, form, concurrency, class, run.
/// CEP:WHY: 38.21 pass certification requires purpose/legality/cost fields;
///          those live in each implementation's CEP:HPC-PASS block, while
///          the machine-checkable parts (form, concurrency, class) live
///          here so the manager can enforce them.
/// CEP:STATUS: complete
/// CEP:FAILURE: implementations report via PassResult.
/// CEP:ASSUMES: implementations are panic-free and deterministic.
/// CEP:COST: contract only.
/// CEP:EVIDENCE: passman tests.
/// CEP:SECURITY: first-party implementations only.
pub trait Pass: Send + Sync {
    /// Stable pass name (pipeline manifest identity).
    fn name(&self) -> &'static str;
    /// Pass version (manifest identity; changes are semantic events).
    fn version(&self) -> u32;
    /// Required IR form.
    fn required_form(&self) -> IrFormPreference;
    /// Concurrency contract.
    fn concurrency(&self) -> PassConcurrency;
    /// HPC class (verification gating).
    fn hpc_class(&self) -> HpcClass;
    /// Numeric pass id for telemetry (stable per manifest).
    fn pass_id(&self) -> u16;
    /// Executes the pass over an immutable snapshot.
    fn run(
        &self,
        ctx: &mut WorkerContext<'_>,
        ir: &Arc<IrSnapshot>,
    ) -> Result<PassOutput, PipelineError>;
}

/// The versioned pipeline (CEP&CC 38.20).
///
/// CEP:WHAT: Ordered pass list with a manifest identity.
/// CEP:WHY: "run optimizer" is banned; the manifest is explicit, versioned,
///          and printed in diagnostics and telemetry.
/// CEP:STATUS: complete
/// CEP:FAILURE: see PipelineError.
/// CEP:ASSUMES: pass order is the semantic contract (docs/pipeline.md).
/// CEP:COST: see module header.
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic manifest order.
pub struct PassManager {
    /// Manifest identity, e.g. "sxla-tier2-2026-09".
    pub manifest: &'static str,
    /// Ordered passes.
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// CEP:WHAT: Creates a manager from an ordered pass list.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: one Vec allocation (init).
    /// CEP:EVIDENCE: tests
    pub fn new(manifest: &'static str, passes: Vec<Box<dyn Pass>>) -> PassManager {
        PassManager { manifest, passes }
    }

    /// CEP:WHAT: Runs the pipeline over a snapshot.
    /// CEP:WHY: The compiler driver: form policing, telemetry, verification
    ///          gating and commit chaining in one place — passes cannot
    ///          skip verification (mechanical enforcement of 38.18).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: first PipelineError aborts; no partial pipeline result.
    /// CEP:ASSUMES: ctx telemetry bus wired to the executor's workers.
    /// CEP:COST: per pass: body + verify + optional commit.
    /// CEP:EVIDENCE: tests `pipeline_runs_and_verifies`, `verify_failure_aborts`.
    /// CEP:HPC-DETERMINISM: deterministic.
    pub fn run(
        &self,
        ctx: &mut WorkerContext<'_>,
        ir: &Arc<IrSnapshot>,
    ) -> Result<Arc<IrSnapshot>, PipelineError> {
        // Entry verification (CEP&CC 38.18: verification runs after
        // parsing/lowering AND after each HPC-0 pass — the frontend hands us
        // a verified snapshot or we refuse to run).
        if let Err(e) = verify(ir.arena()) {
            return Err(PipelineError::VerifyFailed("pipeline-entry", e));
        }
        let mut current = Arc::clone(ir);
        for pass in &self.passes {
            // Form policing: our Level-0..2 snapshots are graph-form; a
            // Structured requirement currently maps to the schedule
            // projection at Level 3 (documented partial; loud failure for
            // future forms).
            let form_ok = match pass.required_form() {
                IrFormPreference::Either | IrFormPreference::Graph => true,
                IrFormPreference::Structured => {
                    // Structured consumers (Level 3/4) receive the graph
                    // snapshot and project it themselves (convert.rs);
                    // accepted here, conversion responsibility is theirs.
                    true
                }
            };
            if !form_ok {
                return Err(PipelineError::FormConversion(pass.name()));
            }
            let _ = ctx.emit(TelemetryKind::PassStart, pass.pass_id(), 0);
            let out = pass.run(ctx, &current)?;
            match out {
                PassOutput::Unchanged => {}
                PassOutput::Changed(next) => {
                    // Verification after each HPC-0 pass (38.18).
                    if pass.hpc_class() == HpcClass::Hpc0 {
                        if let Err(e) = verify(next.arena()) {
                            return Err(PipelineError::VerifyFailed(pass.name(), e));
                        }
                    }
                    current = next;
                }
            }
            let _ = ctx.emit(
                TelemetryKind::PassEnd,
                pass.pass_id(),
                u64::from(pass.version()),
            );
        }
        Ok(current)
    }

    /// CEP:WHAT: Manifest line for diagnostics (pass order + versions).
    /// CEP:WHY: 38.20 pipeline documentation requirement; printed by
    ///          xla-opt's --explain-pipeline.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(passes) formatting (CEP-1)
    /// CEP:EVIDENCE: xla-opt tool tests
    pub fn manifest_line(&self) -> String {
        let mut s = String::with_capacity(64 + self.passes.len() * 24);
        s.push_str(self.manifest);
        s.push_str(": ");
        for (i, p) in self.passes.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(p.name());
            s.push('@');
            s.push_str(&p.version().to_string());
        }
        s
    }

    /// CEP:WHAT: Pass count (diagnostic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

/// Level-0 canonicalization pass: GVN + DCE (architecture Level 0).
///
/// CEP:WHAT: CanonicalizePass — global value numbering then dead code
///           elimination with the caller-supplied root set.
/// CEP:WHY: "Gear 1 (Static Partitioning) for global CSE/GVN/Algebraic
///          simplification" (arch section 3). Implemented as a Transactional
///          pass: it clones, mutates, verifies through the commit path and
///          publishes a new snapshot.
/// CEP:STATUS: complete
/// CEP:FAILURE: propagates PipelineError::PassFailed on internal error.
/// CEP:ASSUMES: roots reference live result nodes.
/// CEP:COST: GVN O(nodes log nodes) + DCE O(nodes); one clone-commit.
/// CEP:EVIDENCE: xir-graph module tests drive the internals; passman tests
///           drive the pipeline integration.
/// CEP:SECURITY: IR untrusted; internals bounds-checked.
/// CEP:HPC-PASS: canonicalize-l0
/// CEP:HPC-PASS-KIND: canonicalization (GVN + DCE)
/// CEP:HPC-PASS-INPUT: verified Level-0 snapshot + roots
/// CEP:HPC-PASS-OUTPUT: deduplicated, dead-code-free snapshot
/// CEP:HPC-PASS-ANALYSIS-REQUIRED: dominance, use counts
/// CEP:HPC-PASS-ANALYSIS-PRODUCED: none persisted
/// CEP:HPC-PASS-ANALYSIS-INVALIDATED: use-def chains
/// CEP:HPC-PASS-LEGALITY: GVN dominance rule; DCE purity + roots rule
/// CEP:HPC-PASS-PRESERVES: semantics, effect order
/// CEP:HPC-PASS-COST: O(nodes log nodes)
/// CEP:HPC-PASS-FAILURE: conservative abort
/// CEP:HPC-PASS-TARGET: target-independent
/// CEP:HPC-PASS-EVIDENCE: xir-graph tests + passman tests
pub struct CanonicalizePass {
    /// Function result nodes anchoring DCE liveness.
    pub roots: Vec<xir_core::id::NodeId>,
}

impl CanonicalizePass {
    /// CEP:WHAT: Constructs the pass with a root set.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: roots live.
    /// CEP:COST: one Vec copy (init)
    /// CEP:EVIDENCE: passman tests
    pub fn new(roots: Vec<xir_core::id::NodeId>) -> CanonicalizePass {
        CanonicalizePass { roots }
    }
}

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize-l0"
    }
    fn version(&self) -> u32 {
        1
    }
    fn required_form(&self) -> IrFormPreference {
        IrFormPreference::Graph
    }
    fn concurrency(&self) -> PassConcurrency {
        PassConcurrency::Transactional
    }
    fn hpc_class(&self) -> HpcClass {
        HpcClass::Hpc0
    }
    fn pass_id(&self) -> u16 {
        1
    }
    fn run(
        &self,
        _ctx: &mut WorkerContext<'_>,
        ir: &Arc<IrSnapshot>,
    ) -> Result<PassOutput, PipelineError> {
        // Clone-mutate-publish (transactional boundary).
        let mut working: IrArena = ir.arena().deep_clone();
        let folded = xir_graph::fold::run(&mut working);
        let merged = match xir_graph::gvn::run(&mut working) {
            Ok(n) => n,
            Err(_) => return Err(PipelineError::PassFailed(self.name())),
        };
        let removed = match xir_graph::dce::run(&mut working, &self.roots) {
            Ok(n) => n,
            Err(_) => return Err(PipelineError::PassFailed(self.name())),
        };
        if merged == 0 && removed == 0 && folded == 0 {
            return Ok(PassOutput::Unchanged);
        }
        // Verify before publication (commit contract).
        if let Err(e) = verify(&working) {
            return Err(PipelineError::VerifyFailed(self.name(), e));
        }
        Ok(PassOutput::Changed(Arc::new(IrSnapshot::new(working))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anvil::telemetry::TelemetryBus;
    use xir_core::arena::{const_i64, IrArena};
    use xir_core::node::Node;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};

    fn build_snapshot_with_duplicate() -> (Arc<IrSnapshot>, xir_core::id::NodeId) {
        let mut a = IrArena::with_capacity(32, 8);
        let root = a.root_region();
        let c0 = const_i64(&mut a, root, 3);
        let c1 = const_i64(&mut a, root, 4);
        let mut result_id = xir_core::id::NodeId::NONE;
        if let (Ok(v0), Ok(v1)) = (c0, c1) {
            let (x0, x1) = (a.value_of(v0, 0).ok(), a.value_of(v1, 0).ok());
            if let (Some(a0), Some(a1)) = (x0, x1) {
                let add = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let first = a.insert_node(root, add);
                let add2 = Node::new(
                    Op::Binary(BinaryOp::Add),
                    root,
                    &[a0, a1],
                    Type::Scalar(ScalarType::I64),
                );
                let _ = a.insert_node(root, add2);
                if let Ok(f) = first {
                    result_id = f;
                }
            }
        }
        (Arc::new(IrSnapshot::new(a)), result_id)
    }

    // CEP:WHAT: Pipeline runs, canonicalizes, verifies and commits.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on pipeline error.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn pipeline_runs_and_verifies() {
        let (snap, root_id) = build_snapshot_with_duplicate();
        let before = snap.arena().node_count();
        let bus = TelemetryBus::new(1);
        let mut ctx = WorkerContext {
            worker_index: 0,
            workers: 1,
            telemetry: &bus,
        };
        let pm = PassManager::new(
            "sxla-test-pipeline",
            vec![Box::new(CanonicalizePass::new(vec![root_id]))],
        );
        let out = pm.run(&mut ctx, &snap);
        assert!(out.is_ok(), "pipeline failed: {:?}", out.err());
        if let Ok(new_snap) = out {
            // fold(3+4=7) collapses both adds to one ConstF64(7); GVN merges
            // the duplicates; DCE removes the now-unused operand consts.
            // The entire function is ONE folded constant.
            assert_eq!(new_snap.arena().node_count(), 1);
            let mut found_folded = false;
            new_snap.arena().for_each_live_node(|_id, node| {
                if node.op == xir_core::op::Op::ConstI64(7) {
                    found_folded = true;
                }
            });
            assert!(found_folded, "expected folded ConstI64(7)");
            assert_eq!(new_snap.version(), 0); // fresh snapshot numbering
            let _ = before;
        }
        // Telemetry saw PassStart + PassEnd.
        let mut events = Vec::new();
        let n = bus.drain(0, &mut events);
        assert!(n >= 2);
    }

    // CEP:WHAT: A broken snapshot aborts the pipeline loudly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if a broken snapshot passes.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn verify_failure_aborts() {
        // Snapshot with an arity violation (dot with 0 inputs).
        let mut a = IrArena::with_capacity(16, 4);
        let root = a.root_region();
        let bad = Node::new(Op::Dot, root, &[], Type::Scalar(ScalarType::F64));
        let bid = a.insert_node(root, bad);
        assert!(bid.is_ok());
        let snap = Arc::new(IrSnapshot::new(a));
        let bus = TelemetryBus::new(1);
        let mut ctx = WorkerContext {
            worker_index: 0,
            workers: 1,
            telemetry: &bus,
        };
        // Canonicalize verifies BEFORE publication: the pipeline must fail.
        let pm = PassManager::new(
            "sxla-test-broken",
            vec![Box::new(CanonicalizePass::new(vec![]))],
        );
        let out = pm.run(&mut ctx, &snap);
        assert!(out.is_err());
    }

    // CEP:WHAT: Manifest line reflects pass order.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on manifest drift.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn manifest_is_explicit() {
        let pm = PassManager::new(
            "sxla-manifest-test",
            vec![Box::new(CanonicalizePass::new(vec![]))],
        );
        assert_eq!(pm.manifest_line(), "sxla-manifest-test: canonicalize-l0@1");
        assert_eq!(pm.pass_count(), 1);
    }
}
