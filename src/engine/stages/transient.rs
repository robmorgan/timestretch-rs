//! Artifact-first transient control: the cursor that maps pre-analysis
//! onset/beat positions onto the engine's stage timeline.
//!
//! The [`PreAnalysisArtifact`] is the engine's primary control signal
//! (ROADMAP Stage 4): its positions are absolute track frames, produced
//! offline with full lookahead. This cursor threads them through two
//! timeline transforms — the host's track anchor (track → ring frames,
//! surviving seeks and loop wraps without resets) and the varispeed
//! timeline map (ring → stage frames, exact under tempo rides) — and
//! publishes the events near the current block as [`OnsetEvent`]s in
//! [`StageCtx`](crate::engine::stage::StageCtx), where the correctors
//! consume them: SOLA steers splices away from onsets, the PV schedules
//! strength-gated per-band phase resets.

use std::sync::Arc;

use crate::core::preanalysis::PreAnalysisArtifact;
use crate::engine::stage::OnsetEvent;

/// How far behind the current stage position events are retained, in
/// frames: covers the corrector latency plus splice-fade spans so a
/// protection window around a just-passed onset is still visible.
const KEEP_BEHIND_FRAMES: f64 = 1_024.0;

/// Scheduling lookahead, in stage frames.
const HORIZON_FRAMES: f64 = 2_048.0;

/// Maximum events published per block (fixed storage, no allocation).
pub(crate) const MAX_EVENTS: usize = 16;

/// Cursor over the artifact's onset and beat lists.
#[derive(Debug)]
pub(crate) struct TransientCursor {
    artifact: Arc<PreAnalysisArtifact>,
    /// Index of the first onset not yet behind the keep window.
    onset_cursor: usize,
    /// Index of the first beat not yet behind the keep window.
    beat_cursor: usize,
    /// Fixed event storage republished every block.
    events: [OnsetEvent; MAX_EVENTS],
    len: usize,
}

impl TransientCursor {
    pub(crate) fn new(artifact: Arc<PreAnalysisArtifact>) -> Self {
        Self {
            artifact,
            onset_cursor: 0,
            beat_cursor: 0,
            events: [OnsetEvent::default(); MAX_EVENTS],
            len: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.onset_cursor = 0;
        self.beat_cursor = 0;
        self.len = 0;
    }

    /// Rebuilds the published window for the current block.
    ///
    /// `map_track_to_stage` converts an absolute track frame to a stage
    /// timeline frame (`None` while the position is unmappable — not yet
    /// fed, or behind the anchored history). `stage_now` is the stage frame
    /// of the block being processed.
    pub(crate) fn advance(
        &mut self,
        stage_now: f64,
        mut map_track_to_stage: impl FnMut(u64) -> Option<f64>,
    ) -> &[OnsetEvent] {
        self.len = 0;

        // Drop onsets that have fallen behind the keep window. Positions
        // are re-mapped every block (cheap: a handful of interpolations),
        // so tempo rides move scheduled events instead of stale copies.
        let onsets = &self.artifact.transient_onsets;
        while self.onset_cursor < onsets.len() {
            match map_track_to_stage(onsets[self.onset_cursor] as u64) {
                Some(stage_frame) if stage_frame < stage_now - KEEP_BEHIND_FRAMES => {
                    self.onset_cursor += 1;
                }
                _ => break,
            }
        }
        let beats = &self.artifact.beat_positions;
        while self.beat_cursor < beats.len() {
            match map_track_to_stage(beats[self.beat_cursor] as u64) {
                Some(stage_frame) if stage_frame < stage_now - KEEP_BEHIND_FRAMES => {
                    self.beat_cursor += 1;
                }
                _ => break,
            }
        }

        // Publish onsets inside [now - keep, now + horizon].
        let mut idx = self.onset_cursor;
        while idx < onsets.len() && self.len < MAX_EVENTS {
            let Some(stage_frame) = map_track_to_stage(onsets[idx] as u64) else {
                break;
            };
            if stage_frame > stage_now + HORIZON_FRAMES {
                break;
            }
            self.events[self.len] = OnsetEvent {
                stage_frame,
                strength: self.artifact.strength_at(idx),
                beat: false,
            };
            self.len += 1;
            idx += 1;
        }
        // Beats too (marked): SOLA prefers splicing right after beats where
        // the mix is most masked; beats that coincide with onsets simply
        // appear twice with different flags — consumers treat them alike
        // for protection purposes.
        let mut idx = self.beat_cursor;
        while idx < beats.len() && self.len < MAX_EVENTS {
            let Some(stage_frame) = map_track_to_stage(beats[idx] as u64) else {
                break;
            };
            if stage_frame > stage_now + HORIZON_FRAMES {
                break;
            }
            self.events[self.len] = OnsetEvent {
                stage_frame,
                strength: 1.0,
                beat: true,
            };
            self.len += 1;
            idx += 1;
        }

        &self.events[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(onsets: Vec<usize>, beats: Vec<usize>) -> Arc<PreAnalysisArtifact> {
        Arc::new(PreAnalysisArtifact {
            transient_onsets: onsets.clone(),
            transient_strengths: vec![0.9; onsets.len()],
            beat_positions: beats,
            sample_rate: 44_100,
            bpm: 128.0,
            confidence: 0.9,
            ..Default::default()
        })
    }

    #[test]
    fn publishes_events_in_window_and_drops_passed_ones() {
        let mut cursor = TransientCursor::new(artifact(vec![1_000, 3_000, 50_000], vec![]));
        // Identity mapping (track == stage). Keep window is 1_024 behind,
        // horizon 2_048 ahead.
        let events = cursor.advance(1_500.0, |track| Some(track as f64));
        // 1_000 (recent past) and 3_000 (upcoming) are in the window;
        // 50_000 is beyond the horizon.
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].stage_frame, 1_000.0);
        assert_eq!(events[1].stage_frame, 3_000.0);

        // Far past both: they must drop out and never come back.
        let events = cursor.advance(10_000.0, |track| Some(track as f64));
        assert!(events.is_empty());
    }

    #[test]
    fn remapping_moves_events_with_the_tempo_ride() {
        // Onsets fire exactly once at MAPPED positions: the same onset's
        // stage position shifts as the (rate-dependent) map changes, and
        // consumers keyed to `stage_frame <= now` still see one event.
        let mut cursor = TransientCursor::new(artifact(vec![10_000], vec![]));
        let events = cursor.advance(8_000.0, |track| Some(track as f64 / 1.05));
        assert_eq!(events.len(), 1);
        let first = events[0].stage_frame;
        assert!((first - 9_523.8).abs() < 1.0);

        let events = cursor.advance(8_032.0, |track| Some(track as f64 / 1.02));
        assert_eq!(events.len(), 1);
        assert!((events[0].stage_frame - 9_803.9).abs() < 1.0);
    }

    #[test]
    fn beats_are_flagged_and_unmappable_positions_hold_the_cursor() {
        let mut cursor = TransientCursor::new(artifact(vec![2_000], vec![3_000]));
        let events = cursor.advance(1_000.0, |track| {
            if track <= 2_500 {
                Some(track as f64)
            } else {
                None // not yet fed
            }
        });
        assert_eq!(events.len(), 1);
        assert!(!events[0].beat);
        // Once mappable, the beat appears with its flag.
        let events = cursor.advance(1_032.0, |track| Some(track as f64));
        assert_eq!(events.len(), 2);
        assert!(events[1].beat);
        assert_eq!(events[1].stage_frame, 3_000.0);
    }

    #[test]
    fn seek_skips_permanently_passed_events() {
        // After a forward seek the anchor maps earlier onsets to -inf;
        // the cursor must skip them and go on publishing later ones.
        let mut cursor = TransientCursor::new(artifact(vec![1_000, 90_000], vec![]));
        let events = cursor.advance(100.0, |track| {
            if track < 88_000 {
                Some(f64::NEG_INFINITY)
            } else {
                Some(track as f64 - 88_000.0)
            }
        });
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].stage_frame, 2_000.0);
    }
}
