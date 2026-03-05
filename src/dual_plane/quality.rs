//! Quality ladder and callback-budget governor.

use std::time::Duration;

/// Fixed quality tiers used by the RT governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityTier {
    Q0,
    Q1,
    Q2,
    Q3,
    Q4,
}

impl QualityTier {
    #[inline]
    pub fn upgraded(self) -> Self {
        match self {
            QualityTier::Q0 => QualityTier::Q1,
            QualityTier::Q1 => QualityTier::Q2,
            QualityTier::Q2 => QualityTier::Q3,
            QualityTier::Q3 => QualityTier::Q4,
            QualityTier::Q4 => QualityTier::Q4,
        }
    }

    #[inline]
    pub fn downgraded(self) -> Self {
        match self {
            QualityTier::Q0 => QualityTier::Q0,
            QualityTier::Q1 => QualityTier::Q0,
            QualityTier::Q2 => QualityTier::Q1,
            QualityTier::Q3 => QualityTier::Q2,
            QualityTier::Q4 => QualityTier::Q3,
        }
    }

    /// Base lane blend weights `[transient, tonal, residual]`.
    #[inline]
    pub fn lane_weights(self) -> [f32; 3] {
        match self {
            QualityTier::Q0 => [0.05, 0.95, 0.0],
            QualityTier::Q1 => [0.12, 0.83, 0.05],
            QualityTier::Q2 => [0.18, 0.72, 0.10],
            QualityTier::Q3 => [0.24, 0.64, 0.12],
            QualityTier::Q4 => [0.30, 0.58, 0.12],
        }
    }
}

/// End-user latency/quality products.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyProfile {
    Scratch,
    Mix,
    Render,
}

impl LatencyProfile {
    #[inline]
    pub fn initial_tier(self) -> QualityTier {
        match self {
            LatencyProfile::Scratch => QualityTier::Q1,
            LatencyProfile::Mix => QualityTier::Q2,
            LatencyProfile::Render => QualityTier::Q4,
        }
    }

    #[inline]
    pub fn tier_crossfade_blocks(self) -> usize {
        match self {
            LatencyProfile::Scratch => 2,
            LatencyProfile::Mix => 4,
            LatencyProfile::Render => 8,
        }
    }
}

/// Runtime governor configuration.
#[derive(Debug, Clone, Copy)]
pub struct RtGovernorConfig {
    /// Soft callback budget for a processing block.
    pub callback_budget: Duration,
    /// Promote quality when load ratio stays below this level.
    pub upgrade_load_ratio: f32,
    /// Demote quality when load ratio stays above this level.
    pub downgrade_load_ratio: f32,
    /// Consecutive low-load blocks required to promote.
    pub promote_after_blocks: u32,
    /// Consecutive overload blocks required to demote.
    pub demote_after_blocks: u32,
    /// Lower tier bound.
    pub min_tier: QualityTier,
    /// Upper tier bound.
    pub max_tier: QualityTier,
}

impl Default for RtGovernorConfig {
    fn default() -> Self {
        Self {
            callback_budget: Duration::from_micros(2_500),
            upgrade_load_ratio: 0.55,
            downgrade_load_ratio: 0.90,
            promote_after_blocks: 24,
            demote_after_blocks: 2,
            min_tier: QualityTier::Q0,
            max_tier: QualityTier::Q4,
        }
    }
}

/// Callback-time quality governor.
#[derive(Debug, Clone)]
pub struct QualityGovernor {
    cfg: RtGovernorConfig,
    tier: QualityTier,
    promote_streak: u32,
    demote_streak: u32,
}

impl QualityGovernor {
    pub fn new(initial_tier: QualityTier, cfg: RtGovernorConfig) -> Self {
        let tier = clamp_tier(initial_tier, cfg.min_tier, cfg.max_tier);
        Self {
            cfg,
            tier,
            promote_streak: 0,
            demote_streak: 0,
        }
    }

    #[inline]
    pub fn tier(&self) -> QualityTier {
        self.tier
    }

    /// Observes callback time and returns the (possibly updated) tier.
    pub fn observe_block(&mut self, elapsed: Duration) -> QualityTier {
        let budget = self.cfg.callback_budget.as_secs_f64().max(1e-9);
        let load = (elapsed.as_secs_f64() / budget) as f32;

        if load >= self.cfg.downgrade_load_ratio {
            self.demote_streak = self.demote_streak.saturating_add(1);
            self.promote_streak = 0;
            if self.demote_streak >= self.cfg.demote_after_blocks {
                self.demote_streak = 0;
                self.tier =
                    clamp_tier(self.tier.downgraded(), self.cfg.min_tier, self.cfg.max_tier);
            }
            return self.tier;
        }

        if load <= self.cfg.upgrade_load_ratio {
            self.promote_streak = self.promote_streak.saturating_add(1);
            self.demote_streak = 0;
            if self.promote_streak >= self.cfg.promote_after_blocks {
                self.promote_streak = 0;
                self.tier = clamp_tier(self.tier.upgraded(), self.cfg.min_tier, self.cfg.max_tier);
            }
            return self.tier;
        }

        self.promote_streak = 0;
        self.demote_streak = 0;
        self.tier
    }

    pub fn force_demote_once(&mut self) -> QualityTier {
        self.promote_streak = 0;
        self.demote_streak = 0;
        self.tier = clamp_tier(self.tier.downgraded(), self.cfg.min_tier, self.cfg.max_tier);
        self.tier
    }
}

#[inline]
fn clamp_tier(tier: QualityTier, min: QualityTier, max: QualityTier) -> QualityTier {
    if tier < min {
        min
    } else if tier > max {
        max
    } else {
        tier
    }
}

#[cfg(test)]
mod tests {
    use super::{LatencyProfile, QualityGovernor, QualityTier, RtGovernorConfig};
    use std::time::Duration;

    #[test]
    fn profile_defaults_match_expected_tiers() {
        assert_eq!(LatencyProfile::Scratch.initial_tier(), QualityTier::Q1);
        assert_eq!(LatencyProfile::Mix.initial_tier(), QualityTier::Q2);
        assert_eq!(LatencyProfile::Render.initial_tier(), QualityTier::Q4);
    }

    #[test]
    fn governor_demotes_on_overload() {
        let mut gov = QualityGovernor::new(
            QualityTier::Q3,
            RtGovernorConfig {
                demote_after_blocks: 2,
                ..RtGovernorConfig::default()
            },
        );
        assert_eq!(
            gov.observe_block(Duration::from_micros(10_000)),
            QualityTier::Q3
        );
        assert_eq!(
            gov.observe_block(Duration::from_micros(10_000)),
            QualityTier::Q2
        );
    }

    #[test]
    fn governor_promotes_on_sustained_headroom() {
        let mut gov = QualityGovernor::new(
            QualityTier::Q1,
            RtGovernorConfig {
                promote_after_blocks: 3,
                ..RtGovernorConfig::default()
            },
        );
        assert_eq!(
            gov.observe_block(Duration::from_micros(100)),
            QualityTier::Q1
        );
        assert_eq!(
            gov.observe_block(Duration::from_micros(100)),
            QualityTier::Q1
        );
        assert_eq!(
            gov.observe_block(Duration::from_micros(100)),
            QualityTier::Q2
        );
    }
}
