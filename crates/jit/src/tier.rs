// CEP:FILE: crates/jit/src/tier.rs
// CEP:WHAT: Compilation tiers 0-3 and their budgets.
// CEP:WHY: Master architecture section 8: Tier 0 fallback interpreter,
//          Tier 1 fast (<5ms), Tier 2 full search, Tier 3 profile-guided.
//          Tiers are explicit, budgeted values — never implicit policy.
// CEP:CLASS: CEP-1
// CEP:STATUS: partial
// CEP:FAILURE: none (discriminants).
// CEP:ASSUMES: budgets documented in docs/pipeline.md.
// CEP:COST: 1 byte per tier value.
// CEP:EVIDENCE: driver tests exercise Tiers 0-2.
// CEP:SECURITY: none.
// CEP:HPC-DETERMINISM: deterministic tier selection by input flags/state.
// CEP:TODO(main-agent): CEP-25: Tier 3 (PGO) is a loud placeholder.
//! Compilation tiers.

// The compilation tier.
//
// CEP:WHAT: Tier selector.
// CEP:WHY: The dispatcher's policy axis: Tier 0 correctness-first, Tier 1
//          greedy+fast, Tier 2 full search (fusion universes + e-graph).
// CEP:STATUS: partial
// CEP:FAILURE: none
// CEP:ASSUMES: none
// CEP:COST: 1 byte
// CEP:EVIDENCE: driver tests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Tier 0: reference interpreter path (no optimization).
    Tier0,
    /// Tier 1: canonicalization + layout + greedy lowering (< 5ms budget).
    Tier1,
    /// Tier 2: Tier 1 + e-graph saturation + fusion universe search.
    Tier2,
    /// Tier 3: profile-guided recompilation (placeholder — loud failure).
    Tier3,
}

impl Tier {
    /// CEP:WHAT: Telemetry code for the tier.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: branch
    /// CEP:EVIDENCE: cache tests
    pub const fn code(self) -> u64 {
        match self {
            Tier::Tier0 => 0,
            Tier::Tier1 => 1,
            Tier::Tier2 => 2,
            Tier::Tier3 => 3,
        }
    }
}

/// Tier-1 latency budget (milliseconds).
///
/// CEP:WHAT: The architecture's "<5ms" Tier-1 compile budget.
/// CEP:WHY: Named constant (Law 7); the driver measures against it in
///          telemetry and the CI benchmark harness gates regressions
///          (CEP-21 wires the gate; the budget itself lives here).
/// CEP:STATUS: complete
/// CEP:FAILURE: none
/// CEP:ASSUMES: reference machine documented in docs/targets.md.
/// CEP:COST: compile-time constant
/// CEP:EVIDENCE: docs/pipeline.md; CEP-21 harness placeholder.
pub const TIER1_BUDGET_MS: u32 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    // CEP:WHAT: Tier codes are unique and stable.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on collision.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn tier_codes_unique() {
        assert_ne!(Tier::Tier0.code(), Tier::Tier1.code());
        assert_ne!(Tier::Tier1.code(), Tier::Tier2.code());
        assert_ne!(Tier::Tier2.code(), Tier::Tier3.code());
        assert_eq!(TIER1_BUDGET_MS, 5);
    }
}
