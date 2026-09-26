// CEP:FILE: crates/jit/src/driver.rs
// CEP:WHAT: The compile driver — text to compiled kernel through the
//           versioned pipeline (verify, canonicalize, layout, fusion,
//           structurize, lower).
// CEP:WHY: Master architecture sections 7-8: the pass manager runs the
//          manifest; the JIT tiers select the manifest depth. Every tier
//          shares the same entry verification (CEP&CC 38.18) and the same
//          deterministic lowering — a cache hit and a rebuild produce the
//          same kernel.
// CEP:CLASS: CEP-1 (driver)
// CEP:STATUS: complete
// CEP:FAILURE: JitError codes (parse, verify, pipeline, lowering); Tier-0
//             fallback is the caller's policy, not a silent swap.
// CEP:ASSUMES: repository-trusted UTF-8 source text.
// CEP:COST: Tier-1 O(nodes log nodes); Tier-2 adds saturation + universe
//           search (documented budgets in docs/pipeline.md).
// CEP:EVIDENCE: tests `compile_tier1_end_to_end`, `compile_tier2_matches
//             _tier0_semantics`, `bad_input_fails_loudly`.
// CEP:SECURITY: parser validates input (bounded offsets); verifier gates
//             the pipeline.
// CEP:HPC-CLASS: HPC-1.
// CEP:HPC-DETERMINISM: deterministic manifests per tier.
//! The compile driver.

use std::sync::Arc;

use anvil::telemetry::TelemetryBus;
use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::snapshot::IrSnapshot;
use xir_core::text::{parse_arena, ParseError};
use xir_graph::verifier::{verify, VerifierError};
use xir_levels::level3::{project, LoopProgram};
use xir_levels::level4::{lower, TargetError, TargetProgram};
use xir_levels::passman::{PassManager, PassOutput, WorkerContext};

use crate::tier::Tier;

/// Driver failure enumeration.
///
/// CEP:WHAT: Explicit error type for compilation.
/// CEP:WHY: Law 6 + 38.6: invalid input must produce clear diagnostics,
///          never a crash and never a wrong kernel.
/// CEP:STATUS: complete
/// CEP:FAILURE: n/a — this IS the failure report.
/// CEP:ASSUMES: none
/// CEP:COST: 24 bytes
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitError {
    /// The textual IR failed to parse (byte offset carried).
    Parse(ParseError),
    /// Entry verification failed.
    Verify(VerifierError),
    /// A pipeline pass failed.
    Pipeline(&'static str),
    /// Target lowering rejected an op.
    Lower(TargetError),
}

/// A compiled program bundle.
///
/// CEP:WHAT: Target program + results + provenance.
/// CEP:WHY: The cache value and the runtime launch argument.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: built by compile().
/// CEP:COST: program size.
/// CEP:EVIDENCE: tests in this module.
pub struct CompiledProgram {
    /// The lowered CPU target program.
    pub target: TargetProgram,
    /// Result value slots.
    pub results: Vec<u32>,
    /// Tier that produced this program.
    pub tier: Tier,
    /// Structural fingerprint of the source snapshot (cache key base).
    pub fingerprint: u64,
    /// Level-3 bufferization record: (value slot, bytes, address space).
    /// CEP:WHAT: The fusion-aware buffer plan (CEP-22 half-closing).
    /// CEP:WHY: The projection's materialization decisions are part of
    ///          the compiled artifact: at Tier 2, tensor intermediates
    ///          consumed entirely inside their fusion cluster carry
    ///          Register space (they never materialize to global memory);
    ///          cross-cluster values and results stay Global. Device
    ///          runtimes read this plan for allocation; the differential
    ///          tier test asserts the Register deltas.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none (plain data).
    /// CEP:ASSUMES: built by compile().
    /// CEP:COST: O(tensor ops).
    /// CEP:EVIDENCE: `tier2_fusion_changes_bufferization`.
    pub buffers: Vec<(u32, u32, xir_core::ty::AddressSpace)>,
}

/// CEP:WHAT: Parses text into a verified snapshot.
/// CEP:WHY: The shared entry: parse (bounded) then verify (38.18 "after
///          parsing/lowering") — unverified IR never enters the pipeline.
/// CEP:STATUS: complete
/// CEP:FAILURE: Parse / Verify errors with offsets.
/// CEP:ASSUMES: trusted UTF-8 source.
/// CEP:COST: O(bytes) + O(nodes).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic.
pub fn parse_and_verify(src: &str) -> Result<(Arc<IrSnapshot>, Vec<NodeId>), JitError> {
    let arena: IrArena = parse_arena(src, 4096).map_err(JitError::Parse)?;
    verify(&arena).map_err(JitError::Verify)?;
    // Roots: the LAST live node OF THE ROOT REGION is the function result
    // (single-result v1 text format). Audit F-11: the last SLOT overall is
    // an else-BODY node when the program ends in an if (the parser inserts
    // the If first, then the block statements) — the If's value is the
    // result, and a root-region scan selects it.
    let root = arena.root_region();
    let mut last = NodeId::NONE;
    arena.for_each_live_node(|id, node| {
        if node.region == root {
            last = id;
        }
    });
    let roots = if last.is_none() { vec![] } else { vec![last] };
    Ok((Arc::new(IrSnapshot::new(arena)), roots))
}

/// CEP:WHAT: Compiles a snapshot at a tier.
/// CEP:WHY: The tier manifests:
///          Tier 0: verify + project + lower (correctness path).
///          Tier 1: + canonicalize (GVN/DCE) + layout inference.
///          Tier 2: + e-graph saturation + fusion universe search.
///          Tier 3: placeholder — loud failure (CEP-25).
/// CEP:STATUS: complete
/// CEP:FAILURE: see JitError.
/// CEP:ASSUMES: verified snapshot + roots.
/// CEP:COST: see manifests (docs/pipeline.md).
/// CEP:EVIDENCE: tests in this module (fold divergence test proves tiers
///           actually change the output — audit F-3 regression).
/// CEP:HPC-DETERMINISM: deterministic manifests.
pub fn compile(
    snap: &Arc<IrSnapshot>,
    roots: &[NodeId],
    tier: Tier,
) -> Result<CompiledProgram, JitError> {
    if tier == Tier::Tier3 {
        // CEP:STATUS: placeholder — Tier 3 requires the PGO harness (CEP-25);
        // failing loudly beats silently degrading (CEP&CC 10.5).
        return Err(JitError::Pipeline("tier3-pgo-unavailable"));
    }
    let bus = TelemetryBus::new(1);
    let mut ctx = WorkerContext {
        worker_index: 0,
        workers: 1,
        telemetry: &bus,
    };
    let current: Arc<IrSnapshot> = match tier {
        Tier::Tier0 => Arc::clone(snap),
        Tier::Tier1 | Tier::Tier2 => {
            let manifest: &'static str = if tier == Tier::Tier1 {
                "sxla-tier1-2026-09"
            } else {
                "sxla-tier2-2026-09"
            };
            let mut passes: Vec<Box<dyn xir_levels::passman::Pass>> = Vec::with_capacity(2);
            passes.push(Box::new(xir_levels::passman::CanonicalizePass::new(
                roots.to_vec(),
            )));
            if tier == Tier::Tier2 {
                // Tier 2: e-graph saturation + extraction application
                // (transactional, CEP-17) then the fusion search.
                passes.push(Box::new(EgraphPass {
                    roots: roots.to_vec(),
                }));
            }
            let pm = PassManager::new(manifest, passes);
            match pm.run(&mut ctx, snap) {
                Ok(next) => next,
                Err(e) => return Err(JitError::Pipeline(pass_name(&e))),
            }
        }
        Tier::Tier3 => Arc::clone(snap),
    };
    // Tier-1 layout inference: run on a working clone and republish through
    // the verify-gated snapshot discipline (layout is metadata for the
    // fusion legality engine; the manifest contract includes it).
    let mut current = current;
    if tier == Tier::Tier1 || tier == Tier::Tier2 {
        let mut working = current.arena().deep_clone();
        let _ = xir_levels::level1::layout_infer(&mut working);
        if let Err(e) = xir_graph::verifier::verify(&working) {
            return Err(JitError::Verify(e));
        }
        current = Arc::new(IrSnapshot::new(working));
    }
    // Tier-2 fusion search runs on the CANONICALIZED arena — the optimized
    // snapshot the pipeline produced (audit F-3: previously discarded; the
    // search is a scheduling transformation whose ClusterSet feeds the
    // level-3 tiling decisions; semantic equivalence is enforced by the
    // differential integration test).
    //
    // CEP-22 half-closing (this search's outcome is now CONSUMED): the
    // winning ClusterSet threads into project_with_fusion — cluster-aware
    // scheduling (member adjacency) + fusion-aware bufferization
    // (intra-cluster tensor intermediates become Register space). The
    // arena itself is untouched (fusion is a scheduling transformation);
    // values are enforced equal by the differential tier test.
    let mut fused_clusters: Option<xir_levels::level2::ClusterSet> = None;
    if tier == Tier::Tier2 {
        let arena = current.arena();
        let outcome = fusion::search::search(arena, anvil::default_worker_count().max(1))
            .map_err(|_| JitError::Pipeline("fusion-search"))?;
        fused_clusters = Some(outcome.clusters);
    }
    // Structurize + lower — FROM THE OPTIMIZED SNAPSHOT (audit F-3); at
    // Tier 2 the projection honors the fusion decisions (CEP-22).
    let prog: LoopProgram = match fused_clusters {
        Some(clusters) => {
            xir_levels::level3::project_with_fusion(current.arena(), roots, &clusters)
                .map_err(|_| JitError::Pipeline("structurize"))?
        }
        None => project(current.arena(), roots).map_err(|_| JitError::Pipeline("structurize"))?,
    };
    let target = lower(&prog).map_err(JitError::Lower)?;
    Ok(CompiledProgram {
        results: target.results.clone(),
        fingerprint: current.fingerprint(),
        tier,
        target,
        buffers: prog.buffers.clone(),
    })
}

/// CEP:WHAT: Parses and compiles text in one call (tool entry).
/// CEP:STATUS: complete
/// CEP:FAILURE: see JitError.
/// CEP:ASSUMES: trusted source.
/// CEP:COST: parse + tier cost.
/// CEP:EVIDENCE: xla-run end-to-end tests.
pub fn compile_text(src: &str, tier: Tier) -> Result<CompiledProgram, JitError> {
    let (snap, roots) = parse_and_verify(src)?;
    compile(&snap, &roots, tier)
}

/// Tier-2 e-graph pass (extraction application, CEP-17).
///
/// CEP:WHAT: EgraphPass — saturates the pure subgraph, extracts the
///           cheapest program, and APPLIES the selected rewrites through
///           the transactional clone-mutate-verify-publish discipline.
/// CEP:WHY: Architecture section 4: saturation discovers equalities,
///          extraction selects, application lands the rewrites in the IR.
///          Until v2 this pass was analysis-only (ReadOnly, results
///          discarded) — the audit's "dead analysis" gap.
/// CEP:STATUS: complete
/// CEP:FAILURE: PipelineError::PassFailed on saturation failure;
///              VerifyFailed on post-application verification.
/// CEP:ASSUMES: verified, canonicalized input (manifest order runs
///              canonicalize-l0 first).
/// CEP:COST: saturation + extraction + application (see egraph::apply).
/// CEP:EVIDENCE: jit driver tier-2 divergence test (instruction count
///           drops below Tier-1 while values agree differentially).
struct EgraphPass {
    /// Function result nodes anchoring the closing DCE sweep.
    roots: Vec<NodeId>,
}

impl xir_levels::passman::Pass for EgraphPass {
    fn name(&self) -> &'static str {
        "egraph-apply"
    }
    fn version(&self) -> u32 {
        2
    }
    fn required_form(&self) -> xir_levels::passman::IrFormPreference {
        xir_levels::passman::IrFormPreference::Graph
    }
    fn concurrency(&self) -> xir_levels::passman::PassConcurrency {
        xir_levels::passman::PassConcurrency::Transactional
    }
    fn hpc_class(&self) -> xir_levels::passman::HpcClass {
        xir_levels::passman::HpcClass::Hpc0
    }
    fn pass_id(&self) -> u16 {
        2
    }
    fn run(
        &self,
        _ctx: &mut WorkerContext<'_>,
        ir: &Arc<IrSnapshot>,
    ) -> Result<PassOutput, xir_levels::passman::PipelineError> {
        // Clone-mutate-verify-publish (transactional boundary).
        let mut working: IrArena = ir.arena().deep_clone();
        let outcome = egraph::apply::apply(&mut working, &self.roots, 8192)
            .map_err(|_| xir_levels::passman::PipelineError::PassFailed(self.name()))?;
        if outcome.rewrites_applied == 0 {
            return Ok(PassOutput::Unchanged);
        }
        // Verify before publication (commit contract; the pass manager
        // re-verifies HPC-0 passes as well — both gates run).
        if let Err(e) = verify(&working) {
            return Err(xir_levels::passman::PipelineError::VerifyFailed(
                self.name(),
                e,
            ));
        }
        Ok(PassOutput::Changed(Arc::new(IrSnapshot::new(working))))
    }
}

/// CEP:WHAT: Maps pipeline errors to stable diagnostic strings.
/// CEP:WHY: JitError::Pipeline carries &'static str; the mapping keeps the
///          failure report loud and allocation-free.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: none
/// CEP:COST: branch
/// CEP:EVIDENCE: driver tests
fn pass_name(e: &xir_levels::passman::PipelineError) -> &'static str {
    match e {
        xir_levels::passman::PipelineError::PassFailed(p) => p,
        xir_levels::passman::PipelineError::VerifyFailed(p, _) => p,
        xir_levels::passman::PipelineError::FormConversion(p) => p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Foldable redundancy: (3.0 + 4.0) folds at Tier 1+, stays unfolded at
    // Tier 0 — tiers MUST diverge in instruction count while agreeing on
    // results (audit F-3 regression).
    const SRC: &str = "xir v1 func @main {\n  %0 = const.f64 3.0\n  %1 = const.f64 4.0\n  %2 = binary.add %0, %1\n  %3 = param 0\n  %4 = binary.mul %2, %3\n}\n";

    // CEP:WHAT: Tier-1 compiles text end to end.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on any pipeline stage.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn compile_tier1_end_to_end() {
        let out = compile_text(SRC, Tier::Tier1);
        assert!(out.is_ok(), "compile failed: {:?}", out.err());
        if let Ok(p) = out {
            assert_eq!(p.tier, Tier::Tier1);
            assert!(!p.results.is_empty());
            assert!(p.fingerprint != 0);
        }
    }

    // CEP:WHAT: Tiers preserve semantics AND Tier 1/2 actually optimize.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if results diverge OR if optimization is a
    //               no-op (audit F-3: the old test could not detect that).
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn compile_tier2_matches_tier0_semantics() {
        let t0 = compile_text(SRC, Tier::Tier0);
        let t1 = compile_text(SRC, Tier::Tier1);
        let t2 = compile_text(SRC, Tier::Tier2);
        assert!(t0.is_ok() && t1.is_ok() && t2.is_ok());
        if let (Ok(a), Ok(b), Ok(c)) = (t0, t1, t2) {
            // Each tier produces exactly one result (slot indices differ
            // legitimately: folding removes value slots).
            assert_eq!(a.results.len(), 1);
            assert_eq!(b.results.len(), 1);
            assert_eq!(c.results.len(), 1);
            // Tier 0 carries the unfolded chain (2 consts + add + mul =
            // 4 instructions); Tier 1/2 fold 3+4 into one const
            // (const + mul = 2 instructions).
            assert_eq!(a.target.instrs.len(), 4);
            assert_eq!(b.target.instrs.len(), 2);
            assert_eq!(c.target.instrs.len(), 2);
            // Structural divergence: the optimized IR differs.
            assert_ne!(a.fingerprint, b.fingerprint);
            // Semantic preservation: execute all three with param=2 and
            // compare VALUES (7 * 2 = 14 at every tier).
            use runtime::interp::execute;
            use runtime::value::Value;
            let va = execute(&a.target, &[Value::F64(2.0)]);
            let vb = execute(&b.target, &[Value::F64(2.0)]);
            let vc = execute(&c.target, &[Value::F64(2.0)]);
            assert!(va.is_ok() && vb.is_ok() && vc.is_ok());
            if let (Ok(xa), Ok(xb), Ok(xc)) = (va, vb, vc) {
                assert_eq!(xa[0], Value::F64(14.0));
                assert_eq!(xb[0], Value::F64(14.0));
                assert_eq!(xc[0], Value::F64(14.0));
            }
        }
    }

    // CEP:WHAT: Invalid input fails loudly with an offset.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if garbage compiles.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn bad_input_fails_loudly() {
        let bad = compile_text("xir v1 func @main { %0 = nosuchop }", Tier::Tier1);
        assert!(matches!(bad, Err(JitError::Parse(_))));
        let t3 = compile_text(SRC, Tier::Tier3);
        assert_eq!(t3.err(), Some(JitError::Pipeline("tier3-pgo-unavailable")));
    }

    // Integer identity: param + 0 survives Tier 1 (fold needs both operands
    // constant) but the Tier-2 e-graph eliminates it (saturation discovers
    // x+0 == x; extraction selects the operand; application rewires the
    // consumer) — tiers MUST diverge in instruction count while agreeing on
    // values (CEP-17 closing regression).
    // CEP:WHAT: Tier-2 e-graph application optimizes beyond Tier-1.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the identity is not eliminated or if
    //               values diverge.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier2_egraph_identity_diverges_from_tier1() {
        const SRC2: &str = "xir v1 func @main {\n  %0 = param 0\n  %1 = const.i64 0\n  %2 = binary.add %0, %1\n  %3 = const.i64 7\n  %4 = binary.mul %2, %3\n}\n";
        let t1 = compile_text(SRC2, Tier::Tier1);
        let t2 = compile_text(SRC2, Tier::Tier2);
        assert!(t1.is_ok() && t2.is_ok());
        if let (Ok(a), Ok(b)) = (t1, t2) {
            // Tier 1: const-zero + add + const-7 + mul = 4 instrs (params
            // are function arguments, not instructions; no fold — param is
            // not constant; no GVN — no duplicates).
            // Tier 2: e-graph eliminates x+0 -> x, DCE drops the dead zero
            // and add: const-7 + mul = 2 instrs.
            assert_eq!(a.target.instrs.len(), 4);
            assert_eq!(b.target.instrs.len(), 2);
            // Differential values: (5 + 0) * 7 == 5 * 7 == 35.
            use runtime::interp::execute;
            use runtime::value::Value;
            let va = execute(&a.target, &[Value::I64(5)]);
            let vb = execute(&b.target, &[Value::I64(5)]);
            assert!(va.is_ok() && vb.is_ok());
            if let (Ok(xa), Ok(xb)) = (va, vb) {
                assert_eq!(xa[0], Value::I64(35));
                assert_eq!(xb[0], Value::I64(35));
            }
        }
    }

    // CEP:WHAT: Tier-2's fusion search CHANGES the compiled artifact (the
    //           CEP-22 half-closing regression for audit F-3): the tensor
    //           chain's intra-cluster intermediates become Register buffers
    //           at Tier 2 only, while Tier 1 materializes everything to
    //           Global — AND the executed values agree differentially.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the search outcome stops feeding the
    //               projection (the dead-analysis regression) or if values
    //               diverge.
    // CEP:ASSUMES: elementwise tensor chain (param -> mul -> add -> relu).
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier2_fusion_changes_bufferization() {
        const SRC4: &str = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : tensor<f64>[2,2] row\n",
            "  %1 = const.f64 2.0 : scalar<f64>\n",
            "  %2 = binary.mul %0, %1 : tensor<f64>[2,2] row\n",
            "  %3 = const.f64 3.0 : scalar<f64>\n",
            "  %4 = binary.add %2, %3 : tensor<f64>[2,2] row\n",
            "  %5 = unary.relu %4 : tensor<f64>[2,2] row\n",
            "}\n",
        );
        let t1 = compile_text(SRC4, Tier::Tier1);
        let t2 = compile_text(SRC4, Tier::Tier2);
        assert!(
            t1.is_ok() && t2.is_ok(),
            "t1={:?} t2={:?}",
            t1.err(),
            t2.err()
        );
        if let (Ok(a), Ok(b)) = (t1, t2) {
            // Three tensor ops -> three buffer records at both tiers.
            assert_eq!(a.buffers.len(), 3);
            assert_eq!(b.buffers.len(), 3);
            // Tier 1: everything Global (no fusion decisions).
            for rec in a.buffers.iter() {
                assert_eq!(rec.2, xir_core::ty::AddressSpace::Global);
            }
            // Tier 2: the two intra-cluster intermediates are Register;
            // the result root stays Global.
            assert_eq!(b.buffers[0].2, xir_core::ty::AddressSpace::Register);
            assert_eq!(b.buffers[1].2, xir_core::ty::AddressSpace::Register);
            assert_eq!(b.buffers[2].2, xir_core::ty::AddressSpace::Global);
            // Differential: relu((x*2)+3) elementwise, identical at both
            // tiers.
            use runtime::interp::execute;
            use runtime::value::Value;
            use xir_core::ty::Shape;
            let shape = Shape::from_dims(&[2, 2]).ok().unwrap_or(Shape::scalar());
            let input = Value::tensor(vec![1.0, -4.0, 2.0, -5.0], shape)
                .ok()
                .unwrap_or(Value::F64(0.0));
            let va = execute(&a.target, std::slice::from_ref(&input));
            let vb = execute(&b.target, std::slice::from_ref(&input));
            assert!(va.is_ok() && vb.is_ok());
            if let (Ok(xa), Ok(xb)) = (va, vb) {
                assert_eq!(xa[0], xb[0]);
                if let Value::Tensor { data, .. } = &xa[0] {
                    // relu([2, -8, 4, -10] + 3) = [5, 0, 7, 0].
                    assert_eq!(data, &vec![5.0, 0.0, 7.0, 0.0]);
                }
            }
        }
    }

    // CEP:WHAT: The CEP-18 integer annihilation rules diverge Tier-2 from
    //           Tier-1: param-param is NOT foldable at Tier 1 (operands are
    //           not constants) but the e-graph proves x-x == 0 by CLASS
    //           EQUALITY and the chain collapses to a constant.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the annihilation rules regress.
    // CEP:ASSUMES: i64 program (param - param) * 5 + 3.
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier2_annihilates_self_subtraction() {
        // NOTE: explicit scalar<i64> annotations — the parser's untyped
        // default is scalar<f64>, and FLOAT x-x must never annihilate
        // (x-x is NaN at x=inf); the integer gate is the point.
        const SRC5: &str = concat!(
            "xir v1 func @main {\n",
            "  %0 = param 0 : scalar<i64>\n",
            "  %1 = binary.sub %0, %0 : scalar<i64>\n",
            "  %2 = const.i64 5 : scalar<i64>\n",
            "  %3 = binary.mul %1, %2 : scalar<i64>\n",
            "  %4 = const.i64 3 : scalar<i64>\n",
            "  %5 = binary.add %3, %4 : scalar<i64>\n",
            "}\n",
        );
        let t1 = compile_text(SRC5, Tier::Tier1);
        let t2 = compile_text(SRC5, Tier::Tier2);
        assert!(t1.is_ok() && t2.is_ok());
        if let (Ok(a), Ok(b)) = (t1, t2) {
            // Tier 1: sub + const5 + mul + const3 + add = 5 instrs.
            // Tier 2: x-x -> 0, 0*5 -> 0, 0+3 -> 3: ONE instr.
            assert_eq!(a.target.instrs.len(), 5);
            assert_eq!(b.target.instrs.len(), 1);
            // Differential: any param value gives 3 at both tiers.
            use runtime::interp::execute;
            use runtime::value::Value;
            let va = execute(&a.target, &[Value::I64(41)]);
            let vb = execute(&b.target, &[Value::I64(41)]);
            assert!(va.is_ok() && vb.is_ok());
            if let (Ok(xa), Ok(xb)) = (va, vb) {
                assert_eq!(xa[0], Value::I64(3));
                assert_eq!(xb[0], Value::I64(3));
            }
        }
    }

    // Float identity: param(f64) + 0.0 must survive EVERY tier (38.24
    // signed-zero discipline — x + 0.0 differs from x at x = -0.0).
    // CEP:WHAT: The e-graph never eliminates float zero identities.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if the float identity is eliminated.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier2_preserves_float_zero_identity() {
        const SRC3: &str = "xir v1 func @main {\n  %0 = param 0\n  %1 = const.f64 0.0\n  %2 = binary.add %0, %1\n}\n";
        let t1 = compile_text(SRC3, Tier::Tier1);
        let t2 = compile_text(SRC3, Tier::Tier2);
        assert!(t1.is_ok() && t2.is_ok());
        if let (Ok(a), Ok(b)) = (t1, t2) {
            assert_eq!(a.target.instrs.len(), b.target.instrs.len());
            // Differential: -0.0 + 0.0 == +0.0 (NOT -0.0) at both tiers.
            use runtime::interp::execute;
            use runtime::value::Value;
            let va = execute(&a.target, &[Value::F64(-0.0)]);
            let vb = execute(&b.target, &[Value::F64(-0.0)]);
            assert!(va.is_ok() && vb.is_ok());
            if let (Ok(xa), Ok(xb)) = (va, vb) {
                let expect = Value::F64(-0.0 + 0.0);
                assert_eq!(xa[0], expect);
                assert_eq!(xb[0], expect);
            }
        }
    }
}
