//! Feed/control thread for the pull-based deck engine (ROADMAP new Stage 1).
//!
//! The pull engine inverts the desktop's data flow: the audio callback owns
//! the [`EngineProcessor`](timestretch::engine::EngineProcessor) and pulls,
//! while this thread keeps the engine's source ring topped up, forwards
//! tempo control, and publishes the playhead. Seeks are cold restarts at
//! the target position for now (warm-start priming is roadmap Stage 5);
//! loop wraps feed straight across the seam, exactly like the old path's
//! `notify_source_jump` timeline re-anchor.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use timestretch::engine::{EngineController, SourceProducer};

use crate::state::{AtomicPosition, SharedStateHandle, StopFlag, Transport};

const CHANNELS: usize = 2;
/// Interleaved samples pushed per feed batch.
const FEED_BATCH_SAMPLES: usize = 2048 * CHANNELS;
/// Ring occupancy (frames) the feeder tops up to — several large callbacks
/// at the fastest supported rate.
const TARGET_OCCUPANCY_FRAMES: usize = 16_384;
/// Occupancy (frames) required before output unmutes after start/seek.
const PREROLL_FRAMES: usize = 4_096;
const RATE_UPDATE_EPSILON: f64 = 1e-6;

/// One source-timeline discontinuity: at cumulative consumed frame
/// `anchor`, playback continued from source frame `target`.
#[derive(Debug, Clone, Copy)]
struct Jump {
    anchor: f64,
    target: f64,
}

/// Maps the engine's cumulative consumed-source position to an absolute
/// source frame across feed-cursor jumps (loop wraps, seeks).
#[derive(Debug)]
struct JumpMap {
    jumps: Vec<Jump>,
}

impl JumpMap {
    fn starting_at(source_frame: f64) -> Self {
        Self {
            jumps: vec![Jump {
                anchor: 0.0,
                target: source_frame,
            }],
        }
    }

    fn record(&mut self, anchor: f64, target: f64) {
        self.jumps.push(Jump { anchor, target });
    }

    fn map(&self, cumulative: f64) -> f64 {
        let jump = self
            .jumps
            .iter()
            .rev()
            .find(|j| j.anchor <= cumulative)
            .or(self.jumps.first())
            .copied()
            .unwrap_or(Jump {
                anchor: 0.0,
                target: 0.0,
            });
        jump.target + (cumulative - jump.anchor)
    }

    /// Drops jumps well behind the playhead, keeping one anchor.
    fn prune(&mut self, cumulative: f64) {
        while self.jumps.len() >= 2 && self.jumps[1].anchor <= cumulative {
            self.jumps.remove(0);
        }
    }
}

/// Starts the pull-deck feed thread. Returns its join handle.
#[allow(clippy::too_many_arguments)]
pub fn start_pull_deck_thread(
    state: SharedStateHandle,
    source_audio: Arc<Vec<f32>>,
    mut source: SourceProducer,
    controller: EngineController,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    stop_flag: Arc<StopFlag>,
    reset_request: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let total_frames = source_audio.len() / CHANNELS;
        // Interleaved read offset into the source.
        let mut cursor: usize = 0;
        let mut jumps = JumpMap::starting_at(0.0);
        // Frames fed to the engine since the last reset (jump anchors).
        let mut fed_frames: f64 = 0.0;
        let mut last_rate = f64::NAN;
        let mut finished = false;
        let mut prerolled = false;
        let mut last_underruns = 0u64;

        // The pull chain has zero pipeline delay and resampler-lookahead
        // control latency; publish once for the UI latency chip.
        {
            let mut st = state.lock().unwrap();
            st.reported_latency_secs = 0.0;
            st.reported_control_latency_secs = 0.0;
        }

        stream_active.store(false, Ordering::Relaxed);

        loop {
            if stop_flag.is_set() {
                break;
            }

            let (transport, stretch_ratio, seek_req, loop_region) = {
                let mut st = state.lock().unwrap();
                let t = st.transport;
                let r = st.stretch_ratio;
                let s = st.seek_request.take();
                let lr = st.loop_region;
                (t, r, s, lr)
            };

            // Tempo: the deck's stretch ratio is output/input length, the
            // engine's tempo rate is playback speed — reciprocal. The
            // engine clamps to its supported range.
            let rate = 1.0 / stretch_ratio.clamp(0.25, 4.0);
            if last_rate.is_nan() || (rate - last_rate).abs() > RATE_UPDATE_EPSILON {
                controller.set_tempo_rate(rate);
                last_rate = rate;
            }

            if let Some(seek_frame) = seek_req {
                // Cold restart at the target (Stage 5 brings warm starts):
                // mute, have the audio callback reset the engine (which
                // also discards in-flight source), then refeed from there.
                stream_active.store(false, Ordering::Relaxed);
                prerolled = false;
                reset_request.store(true, Ordering::Release);
                let mut spins = 0;
                while reset_request.load(Ordering::Acquire) && spins < 500 {
                    thread::sleep(Duration::from_millis(1));
                    spins += 1;
                }
                let target = seek_frame.min(total_frames);
                cursor = target * CHANNELS;
                fed_frames = 0.0;
                jumps = JumpMap::starting_at(target as f64);
                finished = false;
            }

            if transport != Transport::Playing {
                stream_active.store(false, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Loop wrap: jump the feed cursor and re-anchor the timeline.
            // The engine streams straight across the seam — no reset.
            if let Some((loop_start, loop_end)) = loop_region {
                if cursor >= loop_end * CHANNELS {
                    cursor = loop_start * CHANNELS;
                    jumps.record(fed_frames, loop_start as f64);
                    finished = false;
                }
            }

            // End of stream: flush the resampler lookahead once, then stop
            // the transport when the buffered tail has drained.
            if cursor >= source_audio.len() && loop_region.is_none() {
                if !finished {
                    finished = source.finish();
                } else if source.occupied_frames() == 0 {
                    thread::sleep(Duration::from_millis(100));
                    stream_active.store(false, Ordering::Relaxed);
                    {
                        let mut st = state.lock().unwrap();
                        st.transport = Transport::Stopped;
                        st.position_frames = 0;
                    }
                    position.store(0);
                    continue;
                }
            } else if source.occupied_frames() < TARGET_OCCUPANCY_FRAMES {
                // Top up the ring, clamping each batch to the loop end (the
                // wrap above fires on the next iteration) and to EOF.
                let mut end = (cursor + FEED_BATCH_SAMPLES).min(source_audio.len());
                if let Some((_, loop_end)) = loop_region {
                    let loop_end = loop_end * CHANNELS;
                    if cursor < loop_end {
                        end = end.min(loop_end);
                    }
                }
                if end > cursor {
                    let accepted = source.push(&source_audio[cursor..end]);
                    cursor += accepted * CHANNELS;
                    fed_frames += accepted as f64;
                }
            }

            if !prerolled
                && (source.occupied_frames() >= PREROLL_FRAMES || cursor >= source_audio.len())
            {
                prerolled = true;
            }
            stream_active.store(prerolled, Ordering::Relaxed);

            // Playhead: map the engine's cumulative consumed-source position
            // through the jump timeline to an absolute source frame.
            let consumed = controller.source_position();
            jumps.prune(consumed);
            let playhead = jumps.map(consumed).clamp(0.0, total_frames as f64);
            position.store(playhead as usize);

            let underruns = controller.underrun_frames();
            if underruns > last_underruns && !finished {
                log::warn!(
                    "pull deck: {} underrun frames (total {underruns})",
                    underruns - last_underruns
                );
                last_underruns = underruns;
            }

            thread::sleep(Duration::from_millis(2));
        }

        stream_active.store(false, Ordering::Relaxed);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_map_identity_without_jumps() {
        let map = JumpMap::starting_at(0.0);
        assert_eq!(map.map(0.0), 0.0);
        assert_eq!(map.map(1234.5), 1234.5);
    }

    #[test]
    fn jump_map_seek_start_offsets_position() {
        let map = JumpMap::starting_at(44_100.0);
        assert_eq!(map.map(0.0), 44_100.0);
        assert_eq!(map.map(100.0), 44_200.0);
    }

    #[test]
    fn jump_map_loop_wrap_re_anchors() {
        // Fed 1000 frames, then wrapped back to source frame 200.
        let mut map = JumpMap::starting_at(0.0);
        map.record(1000.0, 200.0);
        assert_eq!(map.map(999.0), 999.0); // pre-wrap audio still playing
        assert_eq!(map.map(1000.0), 200.0); // seam
        assert_eq!(map.map(1300.0), 500.0); // inside the loop
    }

    #[test]
    fn jump_map_prune_keeps_active_anchor() {
        let mut map = JumpMap::starting_at(0.0);
        map.record(1000.0, 200.0);
        map.record(2000.0, 200.0);
        map.prune(2500.0);
        assert_eq!(map.map(2500.0), 700.0);
    }
}
