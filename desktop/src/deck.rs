//! Feed/control thread for the deck engine.
//!
//! The audio callback owns the
//! [`EngineProcessor`](timestretch::engine::EngineProcessor) and reads from
//! it; this thread keeps the engine's source ring topped up, forwards tempo
//! control, and publishes the playhead. Seeks feed the preroll preceding the
//! target and request warm-start priming; loop wraps feed straight across the
//! seam via a timeline re-anchor.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use timestretch::engine::{EngineController, SourceProducer};

use crate::state::{
    AtomicPosition, DeckEngine, ScrubPhase, ScrubState, SharedStateHandle, StopFlag, Transport,
};

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

/// Starts the deck feed thread. Returns its join handle.
#[allow(clippy::too_many_arguments)]
pub fn start_deck_thread(
    state: SharedStateHandle,
    source_audio: Arc<Vec<f32>>,
    mut source: SourceProducer,
    controller: EngineController,
    position: Arc<AtomicPosition>,
    stream_active: Arc<AtomicBool>,
    stop_flag: Arc<StopFlag>,
    reset_request: Arc<AtomicBool>,
    scrub: Arc<ScrubState>,
    pipeline_latency_secs: f64,
    warm_start_preroll: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let total_frames = source_audio.len() / CHANNELS;
        // Interleaved read offset into the source.
        let mut cursor: usize = 0;
        let mut jumps = JumpMap::starting_at(0.0);
        // Frames fed to the engine since the last reset (jump anchors).
        let mut fed_frames: f64 = 0.0;
        let mut last_rate = f64::NAN;
        let mut last_keylock: Option<bool> = None;
        let mut finished = false;
        let mut prerolled = false;
        let mut last_underruns = 0u64;

        // Publish the chain's constant pipeline delay (the keylock chain's
        // constant — the same in both deck modes) for the UI latency chip;
        // control-to-audio stays at resampler lookahead either way.
        {
            let mut st = state.lock().unwrap();
            st.reported_latency_secs = pipeline_latency_secs;
            st.reported_control_latency_secs = 0.0;
        }

        stream_active.store(false, Ordering::Relaxed);
        // Anchor the artifact timeline: the first pushed frame is track 0.
        source.set_track_position(0);

        loop {
            if stop_flag.is_set() {
                break;
            }

            let (transport, stretch_ratio, seek_req, loop_region, deck_engine) = {
                let mut st = state.lock().unwrap();
                let t = st.transport;
                let r = st.stretch_ratio;
                let s = st.seek_request.take();
                let lr = st.loop_region;
                let de = st.deck_engine;
                (t, r, s, lr, de)
            };

            // Tempo: the deck's stretch ratio is output/input length, the
            // engine's tempo rate is playback speed — reciprocal. The
            // engine clamps to its supported range.
            let rate = 1.0 / stretch_ratio.clamp(0.25, 4.0);
            if last_rate.is_nan() || (rate - last_rate).abs() > RATE_UPDATE_EPSILON {
                controller.set_tempo_rate(rate);
                last_rate = rate;
            }

            // Deck mode: the engine always runs the keylock chain; Tape is
            // its delay-matched varispeed bypass. Forwarded on change (and
            // once at start, before output unmutes) — the stage crossfades,
            // so a mid-play switch is instant and click-free.
            let keylock = deck_engine == DeckEngine::Keylock;
            if last_keylock != Some(keylock) {
                controller.set_keylock(keylock);
                last_keylock = Some(keylock);
            }

            if let Some(seek_frame) = seek_req {
                // Warm-start seek: mute, have the audio callback reset the
                // engine (which discards in-flight source), then feed the
                // preroll PRECEDING the target and request priming — the
                // graph runs the history through and resumes converged.
                stream_active.store(false, Ordering::Relaxed);
                prerolled = false;
                reset_request.store(true, Ordering::Release);
                let mut spins = 0;
                while reset_request.load(Ordering::Acquire) && spins < 500 {
                    thread::sleep(Duration::from_millis(1));
                    spins += 1;
                }
                let target = seek_frame.min(total_frames);
                let preroll = warm_start_preroll.min(target);
                let feed_from = target - preroll;
                cursor = feed_from * CHANNELS;
                fed_frames = 0.0;
                jumps = JumpMap::starting_at(feed_from as f64);
                source.set_track_position(feed_from as u64);
                controller.warm_start(preroll as u32);
                finished = false;
            }

            if transport != Transport::Playing {
                stream_active.store(false, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Audible scrub: while the pointer holds the platter (`Active`)
            // the audio callback plays its own varispeed voice and leaves
            // the engine unconsumed, and the UI owns the displayed position
            // — don't feed, don't publish a stale engine playhead, don't
            // drive EOF logic. During the release glide (`Settling`) the
            // loop must keep running so the landing seek (handled above)
            // resets, feeds preroll, and primes the engine in parallel with
            // the glide — only the playhead publish stays yielded.
            let scrub_phase = scrub.phase();
            if scrub_phase == ScrubPhase::Active {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Loop wrap: jump the feed cursor and re-anchor the timeline.
            // The engine streams straight across the seam — no reset.
            if let Some((loop_start, loop_end)) = loop_region
                && cursor >= loop_end * CHANNELS
            {
                cursor = loop_start * CHANNELS;
                jumps.record(fed_frames, loop_start as f64);
                source.set_track_position(loop_start as u64);
                finished = false;
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
            // through the jump timeline to an absolute source frame. The
            // glide display belongs to the scrub voice, so don't fight it.
            let consumed = controller.source_position();
            jumps.prune(consumed);
            let playhead = jumps.map(consumed).clamp(0.0, total_frames as f64);
            if scrub_phase == ScrubPhase::Idle {
                position.store(playhead as usize);
            }

            let underruns = controller.underrun_frames();
            if underruns > last_underruns && !finished {
                log::warn!(
                    "deck: {} underrun frames (total {underruns})",
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
