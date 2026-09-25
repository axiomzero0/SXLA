// CEP:FILE: crates/anvil/src/fuel.rs
// CEP:WHAT: FuelMeter — deterministic task-fuel accounting for Gear 2.
// CEP:WHY: The architecture's Task Fuel rule ("if a search branch takes <500ns,
//          it is executed synchronously; only heavy search nodes are pushed to
//          the deque") needs a cost signal. A wall-clock read would violate HPC
//          determinism (CEP&CC 38.10 bans time as a translation input) and pull
//          std::time into CEP-0 (25.3.1). We therefore measure fuel in
//          deterministic abstract work units — one unit per loop step —
//          calibrated so TASK_FUEL_UNITS approximates 500ns on a modern core
//          while keeping compilation bit-deterministic across machines.
// CEP:CLASS: CEP-0
// CEP:STATUS: complete
// CEP:FAILURE: none; fuel exhaustion is a *query*, not a failure.
// CEP:ASSUMES: callers burn 1 unit per search step of comparable weight;
//              calibration documented in config::TASK_FUEL_UNITS.
// CEP:COST: burn: 1 add + 1 compare. should_defer: 1 compare.
// CEP:EVIDENCE: tests `fuel_exhausts_and_refills`, `fuel_is_deterministic`;
//           fusion crate search tests consume FuelMeter end-to-end.
// CEP:SECURITY: no untrusted input.
// CEP:HPC-DETERMINISM: deterministic by construction — the entire point.
//! Deterministic task fuel metering.

use crate::config::TASK_FUEL_UNITS;

/// Deterministic fuel budget for one synchronous search expansion.
///
/// CEP:WHAT: Counter of abstract work units consumed since the last refill.
/// CEP:WHY: Decides inline-vs-dequeue for search branches (Gear 2 task fuel)
///          without wall-clock nondeterminism.
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: one unit == one bounded search step (caller contract).
/// CEP:COST: 4 bytes of state; burn = add+cmp.
/// CEP:EVIDENCE: tests in this module
/// CEP:SECURITY: none
#[derive(Debug, Clone, Copy)]
pub struct FuelMeter {
    spent: u32,
    budget: u32,
}

impl Default for FuelMeter {
    /// CEP:WHAT: Defaults to the configured task-fuel budget.
    /// CEP:WHY: clippy::new_without_default parity; same semantics as new().
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 stores
    /// CEP:EVIDENCE: test `fuel_exhausts_and_refills`
    fn default() -> FuelMeter {
        FuelMeter::new()
    }
}

impl FuelMeter {
    /// CEP:WHAT: Creates a meter with the configured budget.
    /// CEP:WHY: Central config, not a call-site literal (Law 7).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 2 stores
    /// CEP:EVIDENCE: tests in this module
    #[inline]
    pub fn new() -> FuelMeter {
        FuelMeter {
            spent: 0,
            budget: TASK_FUEL_UNITS,
        }
    }

    /// CEP:WHAT: Consumes `units` of fuel; returns true if now exhausted.
    /// CEP:WHY: Search loops call this once per step; the boolean drives the
    ///           push-to-deque decision.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: saturates at u32::MAX instead of overflowing (checked add).
    /// CEP:ASSUMES: units > 0 typically 1.
    /// CEP:COST: 1 add + 1 compare
    /// CEP:EVIDENCE: test `fuel_exhausts_and_refills`
    /// CEP:SECURITY: none
    #[inline]
    pub fn burn(&mut self, units: u32) -> bool {
        self.spent = self.spent.saturating_add(units);
        self.spent >= self.budget
    }

    /// CEP:WHAT: Reports whether the current branch should be deferred to the
    ///           worker deque instead of executing synchronously.
    /// CEP:WHY: The Task Fuel rule: cheap branches run inline; heavy nodes get
    ///           stolen by idle cores.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 compare
    /// CEP:EVIDENCE: tests in this module
    /// CEP:SECURITY: none
    #[inline]
    pub fn should_defer(&self) -> bool {
        self.spent >= self.budget
    }

    /// CEP:WHAT: Resets consumed fuel (start of a new branch).
    /// CEP:WHY: Each dequeued task restarts with a fresh budget.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 store
    /// CEP:EVIDENCE: test `fuel_exhausts_and_refills`
    /// CEP:SECURITY: none
    #[inline]
    pub fn refill(&mut self) {
        self.spent = 0;
    }

    /// CEP:WHAT: Units consumed so far (diagnostic).
    /// CEP:WHY: Telemetry of inline/deferred balance.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: 1 load
    /// CEP:EVIDENCE: tests in this module
    #[inline]
    pub fn spent(&self) -> u32 {
        self.spent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Exhaustion and refill cycle.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on miscount.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fuel_exhausts_and_refills() {
        let mut f = FuelMeter::new();
        assert!(!f.should_defer());
        for _ in 0..TASK_FUEL_UNITS {
            assert!(!f.should_defer());
            f.burn(1);
        }
        assert!(f.should_defer());
        f.refill();
        assert!(!f.should_defer());
        assert_eq!(f.spent(), 0);
    }

    // CEP:WHAT: Fuel accounting is independent of interleaving (determinism).
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires if state leaks between meters.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn fuel_is_deterministic() {
        let mut a = FuelMeter::new();
        let mut b = FuelMeter::new();
        for _ in 0..10 {
            a.burn(1);
            b.burn(1);
        }
        assert_eq!(a.spent(), b.spent());
    }
}
