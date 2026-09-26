// CEP:FILE: crates/egraph/src/lift.rs
// CEP:WHAT: Shared saturation internals — pure-subgraph lift into the
//           e-graph with XIR<->class mapping tables, plus progressive
//           local-rule saturation rounds.
// CEP:WHY: CEP-17 (extraction application) needs the node->class map and
//           the e-node->XIR map to rebuild the arena from extraction
//           choices; saturation must also propagate folded constants
//           through class_const so later rounds can fire on parents
//           (progressive saturation on ONE graph — the old driver re-lifted
//           from the unchanged arena each round, which discarded merges).
//           The lift also tightens the impure-edge discipline: a node with
//           ANY impure value input is NOT lifted at all. The previous
//           analysis-only lift dropped impure edges, corrupting e-node
//           arity (add(impure, 0) lifted as a one-child add). That was
//           benign only because the saturated graph was discarded; with
//           apply() consuming extraction choices it would be unsound.
// CEP:CLASS: CEP-0 (lift) / CEP-1 (rounds)
// CEP:STATUS: complete
// CEP:FAILURE: EgraphError propagation (Full on budget, BadClass on
//              inconsistent state).
// CEP:ASSUMES: verified arena (SSA use-def); only PURE ops lifted.
// CEP:COST: lift O(nodes) worklist; rounds O(rounds * records), bounded
//           by MAX_ROUNDS.
// CEP:EVIDENCE: tests via apply.rs and saturate.rs; regression
//           `apply_skips_impure_inputs` pins the impure-edge discipline.
// CEP:SECURITY: IR treated as untrusted; all lookups checked.
// CEP:HPC-DETERMINISM: deterministic — records in slot order, rules in
//           fixed order, merges union-by-min-id; class_const propagation
//           is value-preserving (sound rewrites only).
// CEP:TODO(main-agent): CEP-18: congruence rebuild after merges (stale
//           child class ids are canonicalized at use; full re-keying of
//           e-node children after merge is the e-graph rebuild step).
//! Shared lift + saturation internals.

use xir_core::arena::IrArena;
use xir_core::id::NodeId;
use xir_core::node::MAX_INPUTS;
use xir_core::op::Op;

use crate::egraph::{EGraph, EgraphError};
use crate::rules::{local_rules, ConstVal, Rewrite};
use crate::saturate::MAX_ROUNDS;

/// One lifted XIR node record.
///
/// CEP:WHAT: The per-node saturation unit.
/// CEP:WHY: Records drive rule rounds; `node` links the e-class back to
///          the XIR node for extraction application.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: children were canonical at record time; canon() at use.
/// CEP:COST: 40 bytes.
/// CEP:EVIDENCE: apply tests.
#[derive(Clone, Copy)]
pub(crate) struct LiftRecord {
    /// XIR node this record lifted.
    pub node: NodeId,
    /// Opcode at lift time.
    pub op: Op,
    /// Child e-classes (canonical at record time).
    pub children: [u32; 4],
    /// Live child count.
    pub n_children: u8,
    /// Element type of the node's result (rule legality gates, CEP-18).
    pub elem: Option<xir_core::ty::ScalarType>,
}

/// The lifted e-graph plus mapping tables.
///
/// CEP:WHAT: Saturation state with XIR correspondence.
/// CEP:WHY: Extraction application needs class->chosen-node (extraction)
///          plus node->class (this) and e-node->XIR (this) to translate
///          choices back into arena rewrites.
/// CEP:STATUS: complete
/// CEP:FAILURE: see EgraphError.
/// CEP:ASSUMES: built by lift().
/// CEP:COST: O(nodes) tables.
/// CEP:EVIDENCE: apply tests.
pub(crate) struct LiftedGraph {
    /// The e-graph.
    pub g: EGraph,
    /// Lifted records (lift order).
    pub recs: Vec<LiftRecord>,
    /// XIR slot -> e-class (canonical at record time; canon() at use).
    pub node_class: Vec<Option<u32>>,
    /// e-node seq -> XIR node (None for rule-generated e-nodes).
    ///
    /// CEP:WHAT: Alignment invariant: xir_of.len() == g.node_count() at
    ///           all times — one entry per APPENDED e-node (structural
    ///           dedup hits append nothing and push nothing).
    pub xir_of: Vec<Option<NodeId>>,
    /// Canonical class -> folded constant value (propagated on merge).
    pub class_const: Vec<Option<ConstVal>>,
}

/// CEP:WHAT: Lifts the arena's liftable pure nodes into an e-graph.
/// CEP:WHY: "Pure subgraphs are lifted into the E-graph engine" (arch
///          section 3 Level 0). A node is liftable iff it is pure AND every
///          value input's producer is pure (impure producers poison the
///          node — see module header). Constants seed class_const so rule
///          rounds can fold without arena re-probing.
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError propagation (Full on budget exhaustion).
/// CEP:ASSUMES: verified arena.
/// CEP:COST: O(nodes) classification + worklist insert.
/// CEP:EVIDENCE: apply/saturate tests.
/// CEP:HPC-DETERMINISM: deterministic — for_each_live_node slot order.
pub(crate) fn lift(arena: &IrArena, budget: usize) -> Result<LiftedGraph, EgraphError> {
    let mut g = EGraph::new(budget);
    let mut node_class: Vec<Option<u32>> = vec![None; arena.slot_count()];
    let mut xir_of: Vec<Option<NodeId>> = Vec::with_capacity(budget);
    // Class ids are monotonic and bounded by the node budget (one class per
    // appended e-node; dedup hits create none).
    let mut class_const: Vec<Option<ConstVal>> = vec![None; budget + 1];
    let mut recs: Vec<LiftRecord> = Vec::new();

    // Pass 1: classify liftable nodes (pure op; all value-input producers
    // pure). Deterministic slot order.
    let mut liftable: Vec<NodeId> = Vec::new();
    arena.for_each_live_node(|id, node| {
        if !node.op.is_pure() {
            return;
        }
        let mut ok = true;
        for i in 0..node.n_inputs as usize {
            if i >= MAX_INPUTS {
                break;
            }
            let def = node.inputs[i].node();
            if def.is_none() {
                continue;
            }
            match arena.node(def) {
                Ok(p) if p.op.is_pure() => {}
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            liftable.push(id);
        }
    });

    // Pass 2: dependency-order worklist insert (SSA inputs before users).
    let mut remaining = liftable;
    let mut passes = 0usize;
    let bound = arena.slot_count() + 1;
    while !remaining.is_empty() && passes <= bound {
        let mut next: Vec<NodeId> = Vec::new();
        let mut progressed = false;
        for id in remaining.iter() {
            let node = arena.node(*id).map_err(|_| EgraphError::BadClass)?;
            let mut children = [0u32; 4];
            let mut n = 0u8;
            let mut ready = true;
            for i in 0..node.n_inputs as usize {
                if i >= MAX_INPUTS {
                    break;
                }
                let def = node.inputs[i].node();
                if def.is_none() {
                    continue;
                }
                match node_class.get(def.index() as usize).copied().flatten() {
                    Some(c) if n < 4 => {
                        children[n as usize] = c;
                        n += 1;
                    }
                    Some(_) => {
                        // More than 4 value inputs: unliftable (e-node
                        // children are bounded at 4).
                        ready = false;
                        break;
                    }
                    None => {
                        ready = false;
                        break;
                    }
                }
            }
            if !ready {
                next.push(*id);
                continue;
            }
            let before = g.node_count();
            let class = g.add(node.op, &children[..n as usize])?;
            // Alignment invariant: one xir_of entry per appended e-node.
            if g.node_count() > before {
                xir_of.push(Some(*id));
            }
            node_class[id.index() as usize] = Some(class);
            if let Op::ConstI64(v) = node.op {
                if (class as usize) < class_const.len() {
                    class_const[class as usize] = Some(ConstVal::I(v));
                }
            } else if let Op::ConstF64(v) = node.op {
                if (class as usize) < class_const.len() {
                    class_const[class as usize] = Some(ConstVal::F(v));
                }
            }
            recs.push(LiftRecord {
                node: *id,
                op: node.op,
                children,
                n_children: n,
                elem: elem_of(&node.ty),
            });
            progressed = true;
        }
        remaining = next;
        passes += 1;
        if !progressed {
            break;
        }
    }
    Ok(LiftedGraph {
        g,
        recs,
        node_class,
        xir_of,
        class_const,
    })
}

/// CEP:WHAT: The const payload of an op, if any.
/// CEP:WHY: Fold rewrites produce Const ops; class_const stores their
///          value for later rounds.
/// CEP:STATUS: complete
/// CEP:FAILURE: none.
/// CEP:ASSUMES: none.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: round tests via apply.
fn const_of_op(op: Op) -> Option<ConstVal> {
    match op {
        Op::ConstI64(v) => Some(ConstVal::I(v)),
        Op::ConstF64(v) => Some(ConstVal::F(v)),
        _ => None,
    }
}

/// CEP:WHAT: Propagates a folded constant from a merged-away class to the
///           canonical winner class.
/// CEP:WHY: class_const is indexed by class id; merges invalidate loser
///          ids, so the value must survive on the winner for later rounds.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (bounds-checked; silent only when ids are out of
///              table range, which cannot happen for in-budget graphs).
/// CEP:ASSUMES: winner/loser are canonical post-merge.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: progressive fold tests (parent folds after child).
fn propagate_const(lg: &mut LiftedGraph, winner: u32, loser: u32) {
    let loser_val = lg.class_const.get(loser as usize).copied().flatten();
    if let Some(v) = loser_val {
        if let Some(slot) = lg.class_const.get_mut(winner as usize) {
            if slot.is_none() {
                *slot = Some(v);
            }
        }
    }
}

/// One record's rule-evaluation snapshot (plain value; Gear-1 partitionable).
///
/// CEP:WHAT: The pure input bundle of local_rules for one record.
/// CEP:WHY: Rule evaluation must be a pure function of a COPYABLE value so
///          `anvil::run_partitioned` can slice it across workers (arch
///          section 4: "workers apply rewrite rules locally without
///          synchronization"); union-find canonicalization happens in the
///          sequential resolve step, not inside the partition body.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data).
/// CEP:ASSUMES: children canonical at snapshot time.
/// CEP:COST: 56 bytes.
/// CEP:EVIDENCE: saturation tests.
#[derive(Clone, Copy)]
struct RuleSnap {
    op: Op,
    children: [u32; 4],
    consts: [Option<ConstVal>; 4],
    elem: Option<xir_core::ty::ScalarType>,
}

/// CEP:WHAT: Runs progressive local-rule saturation rounds on ONE graph.
/// CEP:WHY: Local rules (constant folding, identity elimination) are
///          synchronization-free (arch section 4). Each round: (1) resolve
///          every record's child classes + folded constants sequentially
///          (union-find canonicalization is stateful); (2) evaluate the
///          rules as a PURE function per record under Anvil Gear 1
///          (`run_partitioned` — pooled since CEP-3, deterministic merge by
///          index); (3) apply the fired rewrites sequentially (fold = new
///          const e-node merged into the class + class_const propagation so
///          PARENT records can fold in later rounds; identity = class
///          merge). Rounds stop at fixpoint or MAX_ROUNDS (bounded compile
///          time, CEP&CC 38.11).
/// CEP:STATUS: complete
/// CEP:FAILURE: EgraphError propagation; sequential fallback if partition
///              sizing fails (documented degradation, same results).
/// CEP:ASSUMES: lift() built the graph; records reference live classes.
/// CEP:COST: O(rounds * (records + partition dispatch)).
/// CEP:EVIDENCE: saturate tests (folding fires); apply tests (chained
///           parent folds across rounds).
/// CEP:HPC-DETERMINISM: deterministic — the partition computes pure
///           per-record functions and the driver merges by index; rule
///           order and union-by-min-id merges are scheduling-independent.
pub(crate) fn run_rounds(lg: &mut LiftedGraph) -> Result<u32, EgraphError> {
    let mut applied = 0u32;
    for _round in 0..MAX_ROUNDS {
        let mut round_fired = 0u32;
        let n_recs = lg.recs.len();
        // (1) Resolve rule inputs: canonical child classes + folded consts.
        let mut snaps: Vec<RuleSnap> = Vec::with_capacity(n_recs);
        for i in 0..n_recs {
            let (op, children, n_children) =
                (lg.recs[i].op, lg.recs[i].children, lg.recs[i].n_children);
            let mut cc = [0u32; 4];
            let mut consts = [None; 4];
            for j in 0..n_children as usize {
                let c = lg.g.canon(children[j])?;
                cc[j] = c;
                consts[j] = lg.class_const.get(c as usize).copied().flatten();
            }
            snaps.push(RuleSnap {
                op,
                children: cc,
                consts,
                elem: lg.recs[i].elem,
            });
        }
        // (2) Gear 1: partitioned pure rule evaluation (CEP-3: routed
        // through the persistent pool when the topology matches).
        let mut results: Vec<Option<Rewrite>> = vec![None; n_recs];
        let workers = anvil::default_worker_count().max(1);
        let partition_ok = anvil::run_partitioned(&snaps, &mut results, workers, |s: &RuleSnap| {
            local_rules(s.op, &s.children, &s.consts, s.elem)
        })
        .is_ok();
        if !partition_ok {
            // Sequential fallback (worker bounds; same results).
            for (i, s) in snaps.iter().enumerate() {
                results[i] = local_rules(s.op, &s.children, &s.consts, s.elem);
            }
        }
        // (3) Deterministic sequential merge by record index.
        for (i, r) in results.iter().enumerate() {
            let Some(rw) = r else { continue };
            let node = lg.recs[i].node;
            let own =
                lg.g.canon(lg.node_class[node.index() as usize].ok_or(EgraphError::BadClass)?)?;
            if rw.n_children == 0 {
                // Constant fold: insert the folded const e-node and merge
                // it into the node's class (extraction can now choose it).
                let before = lg.g.node_count();
                let nc = lg.g.add(rw.op, &[])?;
                if lg.g.node_count() > before {
                    lg.xir_of.push(None);
                }
                let winner = lg.g.merge(own, nc)?;
                if let Some(v) = const_of_op(rw.op) {
                    if let Some(slot) = lg.class_const.get_mut(winner as usize) {
                        *slot = Some(v);
                    }
                }
                round_fired += 1;
            } else if rw.n_children == 1 && rw.op == (Op::Param { index: u32::MAX }) {
                // Identity pass-through: the node's class merges with the
                // surviving operand's class (extraction can now choose the
                // operand). The marker op is the reserved Param{u32::MAX}
                // convention of rules.rs.
                let child = rw.children[0];
                let winner = lg.g.merge(own, child)?;
                let loser = if winner == own { child } else { own };
                propagate_const(lg, winner, loser);
                round_fired += 1;
            } else {
                // Unknown rewrite shape (future rules): conservative skip.
                continue;
            }
        }
        // Commutativity saturation (CEP-18): for every commutative binary
        // record, add the operand-swapped e-node and merge the classes —
        // add(b, a) and add(a, b) become provably equal, so PARENTS that
        // differ only by operand order become congruent (the rebuild below
        // then dedups them). Legality: commutative() gates floats off
        // Add/Mul (38.24 rounding; Max/Min ride the documented F-18 note).
        let mut commute_fired = 0u32;
        for i in 0..n_recs {
            let (op, elem, n_children) = (lg.recs[i].op, lg.recs[i].elem, lg.recs[i].n_children);
            if n_children != 2 || !crate::rules::commutative(op, elem) {
                continue;
            }
            // Canonicalize BEFORE the swapped add: records carry
            // at-lift-time child ids; stale ids would hash to a different
            // key and mint duplicate swapped nodes every round.
            let c0 = lg.g.canon(lg.recs[i].children[0])?;
            let c1 = lg.g.canon(lg.recs[i].children[1])?;
            if c0 == c1 {
                continue;
            }
            let own = lg.g.canon(
                lg.node_class[lg.recs[i].node.index() as usize].ok_or(EgraphError::BadClass)?,
            )?;
            // Alignment invariant (audit round 4, F-2): one xir_of entry
            // per APPENDED e-node — mirror the fold path's before/after
            // check.
            let before = lg.g.node_count();
            let swapped = lg.g.add(op, &[c1, c0])?;
            if lg.g.node_count() > before {
                lg.xir_of.push(None);
            }
            let swapped_c = lg.g.canon(swapped)?;
            if own != swapped_c {
                let winner = lg.g.merge(own, swapped_c)?;
                // class_const rides the canonical winner (audit F-6).
                propagate_const(lg, winner, own);
                propagate_const(lg, winner, swapped_c);
                commute_fired += 1;
            }
        }
        // Congruence rebuild (CEP-18): canonicalize children, rebuild the
        // lookup map, merge nodes that became congruent through this
        // round's merges. class_const is re-homed onto canonical ids after
        // the rebuild's merges (audit F-6); the merge count participates in
        // quiescence so a congruence cascade gets its extra round (audit F-5).
        let merged = lg.g.rebuild()?;
        for i in 0..lg.class_const.len() {
            if lg.class_const[i].is_some() {
                let canon = lg.g.canon(i as u32)?;
                propagate_const(lg, canon, i as u32);
            }
        }
        applied += round_fired + commute_fired;
        if round_fired == 0 && commute_fired == 0 && merged == 0 {
            break;
        }
    }
    Ok(applied)
}

/// CEP:WHAT: Element type of a node's result for rule legality gates.
/// CEP:WHY: The 38.24 discipline is element-typed: integer rewrites are
///          exact and always legal; float rewrites need per-rule proofs.
///          Scalars carry their type directly; tensors carry the element.
/// CEP:STATUS: complete
/// CEP:FAILURE: none (None for untracked forms — rules treat it
///              conservatively).
/// CEP:ASSUMES: none.
/// CEP:COST: O(1).
/// CEP:EVIDENCE: `sub_self_is_zero_int_only`, `mul_by_zero_int_only`.
fn elem_of(ty: &xir_core::ty::Type) -> Option<xir_core::ty::ScalarType> {
    match ty {
        xir_core::ty::Type::Scalar(s) => Some(*s),
        xir_core::ty::Type::Tensor(t) => Some(t.elem),
        xir_core::ty::Type::MemRef(t, _) => Some(t.elem),
        _ => None,
    }
}
