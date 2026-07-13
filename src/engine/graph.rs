//! The engine graph and its audio-thread half, [`EngineProcessor`].
//!
//! The processor owns a fixed chain: source ring → varispeed head → DSP
//! stages → output FIFO. `process` is the entire real-time API: it fills
//! exactly the requested frames, never allocates, never blocks, and never
//! returns an error. A block scheduler adapts arbitrary caller callback
//! sizes to the internal fixed block so per-callback stage work stays
//! bounded.

use std::sync::Arc;

use crate::core::preanalysis::PreAnalysisArtifact;
use crate::core::ring_buffer::RingBuffer;
use crate::engine::control::{clamp_tempo_rate, EngineShared, Param};
use crate::engine::source::{SourceRing, TimelineMap};
use crate::engine::stage::{BlockBuf, OnsetEvent, Stage, StageCtx, BLOCK_FRAMES};
use crate::engine::stages::transient::{TransientCursor, MAX_EVENTS};
use crate::engine::stages::varispeed::{VarispeedHead, FEED_CHUNK_FRAMES, MAX_OUT_PER_FEED};

/// Blocks a fast tempo gesture holds off disruptive stage maintenance
/// (~46 ms at 44.1 kHz) — the graph-level modulation-hold policy.
const MODULATION_HOLD_BLOCKS: u32 = 64;
/// Per-call rate change that counts as a fast gesture.
const MODULATION_HOLD_TRIGGER: f64 = 1e-3;

/// Audio-thread half of the engine.
///
/// Single consumer: exactly one thread (the audio callback) may call
/// [`process`](Self::process).
pub struct EngineProcessor {
    shared: Arc<EngineShared>,
    ring: Arc<SourceRing>,
    varispeed: VarispeedHead,
    stages: Vec<Box<dyn Stage>>,
    block: BlockBuf,
    /// Interleaved varispeed output awaiting fixed-block stage processing.
    stage_fifo: RingBuffer<f32>,
    /// Interleaved processed output awaiting delivery to the caller.
    out_fifo: RingBuffer<f32>,
    timeline: TimelineMap,
    /// Varispeed output frames emitted into the pipeline.
    emitted_frames: u64,
    /// Varispeed output frames consumed by the stage chain (block-aligned).
    stage_in_frames: u64,
    /// Frames delivered to the caller (includes underrun silence).
    delivered_frames: u64,
    /// Silence frames delivered due to source underrun.
    underrun_total: u64,
    channels: usize,
    sample_rate: u32,
    rate: f64,
    max_block_frames: usize,
    /// Scratch for popping source frames (one feed chunk, interleaved).
    feed_scratch: Vec<f32>,
    /// Scratch for interleaving varispeed output (one feed chunk's worth).
    interleave_scratch: Vec<f32>,
    /// Scratch for moving one fixed block through the stage chain.
    block_scratch: Vec<f32>,
    pipeline_latency_frames: usize,
    /// Artifact cursor (None = artifact-less stream; stages fall back to
    /// online heuristics).
    transient: Option<TransientCursor>,
    /// Last coherent track anchor (ring frame, track frame).
    anchor: (u64, u64),
    /// Ring frames already consumed when the engine (re)started; converts
    /// the ring's monotonic timeline to this run's fed-source coordinates.
    ring_frames_at_reset: u64,
    /// Remaining blocks of modulation hold.
    modulation_hold_blocks: u32,
}

impl std::fmt::Debug for EngineProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineProcessor")
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("rate", &self.rate)
            .field("stages", &self.stages.len())
            .field("delivered_frames", &self.delivered_frames)
            .finish_non_exhaustive()
    }
}

impl EngineProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        shared: Arc<EngineShared>,
        ring: Arc<SourceRing>,
        stages: Vec<Box<dyn Stage>>,
        channels: usize,
        sample_rate: u32,
        initial_rate: f64,
        max_block_frames: usize,
        artifact: Option<Arc<PreAnalysisArtifact>>,
    ) -> Self {
        let out_fifo_frames = max_block_frames + 2 * MAX_OUT_PER_FEED + BLOCK_FRAMES;
        let pipeline_latency_frames = stages.iter().map(|s| s.latency_frames()).sum();
        Self {
            shared,
            ring,
            varispeed: VarispeedHead::new(channels),
            block: BlockBuf::new(channels),
            stage_fifo: RingBuffer::with_capacity((BLOCK_FRAMES + 2 * MAX_OUT_PER_FEED) * channels),
            out_fifo: RingBuffer::with_capacity(out_fifo_frames * channels),
            timeline: TimelineMap::with_capacity(out_fifo_frames / (FEED_CHUNK_FRAMES / 4) + 16),
            emitted_frames: 0,
            stage_in_frames: 0,
            delivered_frames: 0,
            underrun_total: 0,
            channels,
            sample_rate,
            rate: clamp_tempo_rate(initial_rate),
            max_block_frames,
            feed_scratch: vec![0.0; FEED_CHUNK_FRAMES * channels],
            interleave_scratch: vec![0.0; MAX_OUT_PER_FEED * channels],
            block_scratch: vec![0.0; BLOCK_FRAMES * channels],
            stages,
            pipeline_latency_frames,
            transient: artifact.map(TransientCursor::new),
            anchor: (0, 0),
            ring_frames_at_reset: 0,
            modulation_hold_blocks: 0,
        }
    }

    /// Fills `out` with exactly `out.len() / channels` interleaved frames.
    ///
    /// Infallible and allocation-free: if the source ring runs dry the
    /// shortfall is delivered as silence and counted as underrun. `out`
    /// lengths that are not a whole frame count are truncated to one
    /// (debug builds assert).
    pub fn process(&mut self, out: &mut [f32]) {
        debug_assert_eq!(out.len() % self.channels, 0);
        let whole = out.len() / self.channels * self.channels;
        self.apply_pending_control();

        let max_chunk = self.max_block_frames * self.channels;
        let mut offset = 0;
        while offset < whole {
            let end = (offset + max_chunk).min(whole);
            self.render_chunk(&mut out[offset..end]);
            offset = end;
        }
        out[whole..].fill(0.0);

        self.timeline.evict_before(self.media_delivered_frames());
        let position = self
            .timeline
            .map_to_source(self.position_query_frame())
            .unwrap_or(0.0);
        self.shared
            .publish_position(position, self.delivered_frames);
    }

    /// Constant pipeline delay of the stage chain, in frames. The varispeed
    /// head is zero-delay (its kernel is cursor-aligned), so tape mode
    /// reports 0: the first delivered frame is source frame 0.
    pub fn pipeline_latency_frames(&self) -> usize {
        self.pipeline_latency_frames
    }

    /// Worst-case frames between a control write and its first audible
    /// output at the current rate: pending output backlog plus one feed
    /// chunk of retarget ramp.
    pub fn control_to_audio_bound_frames(&self) -> usize {
        let feed_out = (FEED_CHUNK_FRAMES as f64 / self.rate).ceil() as usize + 1;
        let blocking = if self.stages.is_empty() {
            0
        } else {
            BLOCK_FRAMES - 1
        };
        2 * feed_out + blocking
    }

    /// Kernel lookahead of the varispeed head at unity/pitch-down, in
    /// source frames: how far input availability leads output emission.
    /// This is source-side buffering, not pipeline delay — the first
    /// delivered frame is still source frame 0.
    pub fn varispeed_lookahead_frames(&self) -> usize {
        self.varispeed.lookahead_frames()
    }

    /// Cold reset back to stream start: clears the resamplers, stage chain,
    /// FIFOs, timeline, and any in-flight source in the ring — a reset is a
    /// hard stream restart, so stale source must not play after it. The host
    /// should pause feeding while a reset is pending. (Warm-start priming
    /// arrives in Stage 5.)
    pub fn reset(&mut self) {
        self.varispeed.reset();
        for stage in &mut self.stages {
            stage.reset();
        }
        self.stage_fifo.clear();
        self.out_fifo.clear();
        self.timeline.clear();
        self.emitted_frames = 0;
        self.stage_in_frames = 0;
        self.delivered_frames = 0;
        self.underrun_total = 0;
        if let Some(cursor) = self.transient.as_mut() {
            cursor.reset();
        }
        self.modulation_hold_blocks = 0;
        // Bounded drain (allocation-free): a concurrently pushing producer
        // could otherwise extend this loop indefinitely.
        let max_drains = self.ring.capacity_samples() / self.feed_scratch.len() + 4;
        for _ in 0..max_drains {
            if self.ring.pop_slice(&mut self.feed_scratch) == 0 {
                break;
            }
        }
        // This run's fed-source coordinates restart at the ring's current
        // (monotonic) consumption cursor.
        self.ring_frames_at_reset = self.ring.head_frames();
        self.shared.publish_position(0.0, 0);
    }

    /// Engine channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Engine sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Current tempo rate in effect on the audio thread.
    pub fn current_tempo_rate(&self) -> f64 {
        self.rate
    }

    /// Frames of real (non-underrun) media delivered so far.
    fn media_delivered_frames(&self) -> u64 {
        self.delivered_frames - self.underrun_total
    }

    /// Output-timeline frame whose source position is "now": the next frame
    /// to deliver, shifted back by the stage chain's constant delay.
    fn position_query_frame(&self) -> f64 {
        (self.media_delivered_frames() as f64 - self.pipeline_latency_frames as f64).max(0.0)
    }

    /// Drains the mailbox and applies control at the block boundary.
    /// (Timestamped sample-offset application lands in Stage 5.)
    fn apply_pending_control(&mut self) {
        while let Some(event) = self.shared.pop_event() {
            match event.param {
                Param::TempoRate => {}
            }
        }
        // The latest-value register is authoritative for ASAP semantics and
        // also covers any events dropped on overflow.
        let new_rate = clamp_tempo_rate(self.shared.tempo_latest());
        if (new_rate - self.rate).abs() > MODULATION_HOLD_TRIGGER {
            self.modulation_hold_blocks = MODULATION_HOLD_BLOCKS;
        }
        self.rate = new_rate;

        // Refresh the track anchor (kept when a concurrent write tears).
        if let Some(anchor) = self.ring.anchor.load() {
            self.anchor = anchor;
        }
    }

    /// Renders one caller chunk (at most `max_block_frames` frames).
    fn render_chunk(&mut self, out: &mut [f32]) {
        let needed_samples = out.len();
        self.fill_out_fifo(needed_samples);

        let popped = self.out_fifo.pop_slice(out);
        if popped < needed_samples {
            out[popped..].fill(0.0);
            let missing_frames = ((needed_samples - popped) / self.channels) as u64;
            self.underrun_total += missing_frames;
            self.shared.add_underrun_frames(missing_frames);
        }
        self.delivered_frames += (needed_samples / self.channels) as u64;
    }

    /// Pulls source through the graph until the output FIFO holds at least
    /// `needed_samples`, or the source ring runs dry.
    fn fill_out_fifo(&mut self, needed_samples: usize) {
        // Bounded iterations by construction: each pass either consumes
        // source or proves none is available. The guard only backstops a
        // logic error (const-bounded loop culture from the old engine).
        let guard = needed_samples / FEED_CHUNK_FRAMES + needed_samples + 64;
        for _ in 0..guard {
            if self.out_fifo.len() >= needed_samples {
                return;
            }
            if self.stages.is_empty() {
                if !self.feed_once_into_out() {
                    return;
                }
            } else if !self.advance_stage_pipeline() {
                return;
            }
        }
        debug_assert!(false, "fill_out_fifo guard exhausted");
    }

    /// Feeds one source chunk through varispeed straight into the output
    /// FIFO (tape mode: no stages). Returns false when out of source.
    fn feed_once_into_out(&mut self) -> bool {
        let produced = self.feed_varispeed_once();
        match produced {
            None => false,
            Some(produced) => {
                self.push_varispeed_output(produced, true);
                true
            }
        }
    }

    /// Keeps the stage FIFO topped up and runs whole fixed blocks through
    /// the stage chain. Returns false when no forward progress is possible.
    fn advance_stage_pipeline(&mut self) -> bool {
        let block_samples = BLOCK_FRAMES * self.channels;
        if self.stage_fifo.len() < block_samples {
            let produced = self.feed_varispeed_once();
            match produced {
                None => return false,
                Some(produced) => self.push_varispeed_output(produced, false),
            }
        }
        if self.stage_fifo.len() >= block_samples {
            let popped = self
                .stage_fifo
                .pop_slice(&mut self.block_scratch[..block_samples]);
            debug_assert_eq!(popped, block_samples);
            self.block
                .fill_deinterleaved(&self.block_scratch[..block_samples], BLOCK_FRAMES);
            self.stage_in_frames += BLOCK_FRAMES as u64;

            // Artifact events near this block, mapped track → ring → stage.
            let mut events: [OnsetEvent; MAX_EVENTS] = [OnsetEvent::default(); MAX_EVENTS];
            let mut event_count = 0usize;
            if let Some(cursor) = self.transient.as_mut() {
                let (anchor_ring, anchor_track) = self.anchor;
                let ring_base = self.ring_frames_at_reset;
                let timeline = &self.timeline;
                let mapped = cursor.advance(self.stage_in_frames as f64, |track_frame| {
                    if track_frame < anchor_track {
                        // Behind the anchored region: permanently passed.
                        return Some(f64::NEG_INFINITY);
                    }
                    let ring_frame = anchor_ring + (track_frame - anchor_track);
                    if ring_frame < ring_base {
                        return Some(f64::NEG_INFINITY);
                    }
                    let fed_source = (ring_frame - ring_base) as f64;
                    timeline.map_to_output(fed_source)
                });
                event_count = mapped.len();
                events[..event_count].copy_from_slice(mapped);
            }

            if self.modulation_hold_blocks > 0 {
                self.modulation_hold_blocks -= 1;
            }
            // Delay-matched control: the rate embedded at the END of this
            // block's span on the varispeed timeline, so corrections track
            // the audio being consumed rather than the control target.
            let ctx = StageCtx {
                embedded_rate: self
                    .timeline
                    .rate_at(self.stage_in_frames as f64)
                    .unwrap_or(self.rate),
                onsets: &events[..event_count],
                modulation_hold: self.modulation_hold_blocks > 0,
                has_artifact: self.transient.is_some(),
            };
            for stage in &mut self.stages {
                stage.process(&mut self.block, &ctx);
            }
            self.block
                .write_interleaved(&mut self.block_scratch[..block_samples], BLOCK_FRAMES);
            let pushed = self
                .out_fifo
                .push_slice(&self.block_scratch[..block_samples]);
            debug_assert_eq!(pushed, block_samples);
        }
        true
    }

    /// Pops up to one feed chunk from the source ring and resamples it.
    /// `None` means the ring is empty (underrun); `Some(n)` means the head
    /// now holds `n` output frames (possibly 0 while the kernel lookahead
    /// fills).
    fn feed_varispeed_once(&mut self) -> Option<usize> {
        let popped = self.ring.pop_slice(&mut self.feed_scratch);
        if popped == 0 {
            return None;
        }
        debug_assert_eq!(popped % self.channels, 0);
        let produced = self.varispeed.feed(&self.feed_scratch[..popped], self.rate);
        self.emitted_frames += produced as u64;
        self.timeline
            .push(self.emitted_frames, self.varispeed.source_pos(), self.rate);
        Some(produced)
    }

    /// Interleaves the head's most recent output into the chosen FIFO.
    fn push_varispeed_output(&mut self, produced: usize, direct_to_out: bool) {
        if produced == 0 {
            return;
        }
        let samples = produced * self.channels;
        for ch in 0..self.channels {
            let src = self.varispeed.output(ch);
            for (f, &sample) in src.iter().enumerate().take(produced) {
                self.interleave_scratch[f * self.channels + ch] = sample;
            }
        }
        let fifo = if direct_to_out {
            &mut self.out_fifo
        } else {
            &mut self.stage_fifo
        };
        let pushed = fifo.push_slice(&self.interleave_scratch[..samples]);
        debug_assert_eq!(pushed, samples, "engine FIFO sized too small");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig, EngineProfile};

    fn sine(freq: f32, sample_rate: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn tape_unity_is_exact_passthrough_from_frame_zero() {
        let handles = Engine::build(EngineConfig {
            channels: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        let (mut processor, mut source) = (handles.processor, handles.source);

        let input = sine(440.0, 44_100.0, 8192);
        source.push(&input);

        let mut out = vec![0.0f32; 512];
        let mut collected = Vec::new();
        for _ in 0..8 {
            processor.process(&mut out);
            collected.extend_from_slice(&out);
        }
        assert_eq!(processor.pipeline_latency_frames(), 0);
        for (i, (&got, &want)) in collected.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "sample {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn stereo_channels_stay_aligned() {
        let handles = Engine::build(EngineConfig::default()).unwrap();
        let (mut processor, mut source) = (handles.processor, handles.source);

        let frames = 4096;
        let mut interleaved = Vec::with_capacity(frames * 2);
        let mono = sine(300.0, 44_100.0, frames);
        for &s in &mono {
            interleaved.push(s);
            interleaved.push(-s);
        }
        source.push(&interleaved);

        let mut out = vec![0.0f32; 256 * 2];
        for _ in 0..8 {
            processor.process(&mut out);
            for frame in out.chunks(2) {
                assert!(
                    (frame[0] + frame[1]).abs() < 1e-5,
                    "stereo misalignment: {frame:?}"
                );
            }
        }
    }

    #[test]
    fn underrun_fills_silence_and_counts_then_recovers() {
        let handles = Engine::build(EngineConfig {
            channels: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        let (controller, mut processor, mut source) =
            (handles.controller, handles.processor, handles.source);

        // Only 100 frames available for a 256-frame request.
        source.push(&vec![0.5f32; 100]);
        let mut out = vec![1.0f32; 256];
        processor.process(&mut out);
        assert!(controller.underrun_frames() > 0);
        assert_eq!(out[255], 0.0, "shortfall must be silence");

        // Refill: engine must resume without error.
        source.push(&vec![0.5f32; 4096]);
        processor.process(&mut out);
        assert!(out.iter().any(|&s| s != 0.0), "must recover after refill");
    }

    #[test]
    fn odd_callback_sizes_fill_exactly() {
        let handles = Engine::build(EngineConfig {
            channels: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        let (mut processor, mut source) = (handles.processor, handles.source);
        source.push(&sine(440.0, 44_100.0, 44_100));

        // Non-power-of-two and larger-than-max-block requests both fill.
        for size in [64usize, 96, 100, 333, 1024, 1500, 4096] {
            let mut out = vec![9.9f32; size];
            processor.process(&mut out);
            assert!(
                out.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
                "bad output at callback size {size}"
            );
        }
    }

    /// A stage that inverts polarity — validates the fixed-block path.
    struct Invert;
    impl Stage for Invert {
        fn process(&mut self, block: &mut BlockBuf, ctx: &StageCtx) {
            assert!(ctx.embedded_rate > 0.0, "ctx must carry a usable rate");
            for ch in 0..block.channels() {
                for s in block.channel_mut(ch) {
                    *s = -*s;
                }
            }
        }
        fn latency_frames(&self) -> usize {
            0
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn stage_chain_processes_fixed_blocks() {
        let config = EngineConfig {
            channels: 1,
            profile: EngineProfile::Tape,
            ..EngineConfig::default()
        };
        let handles = Engine::build_with_stages(config, vec![Box::new(Invert)]).unwrap();
        let (mut processor, mut source) = (handles.processor, handles.source);

        let input = sine(440.0, 44_100.0, 8192);
        source.push(&input);
        let mut out = vec![0.0f32; 512];
        let mut collected = Vec::new();
        for _ in 0..8 {
            processor.process(&mut out);
            collected.extend_from_slice(&out);
        }
        for (i, (&got, &want)) in collected.iter().zip(input.iter()).enumerate() {
            assert!(
                (got + want).abs() < 1e-4,
                "sample {i}: got {got}, want inverted {want}"
            );
        }
    }

    /// Zero-crossing frequency estimate over a window.
    fn measure_freq(window: &[f32], sample_rate: f64) -> f64 {
        let (mut first, mut last, mut count) = (None, None, 0usize);
        for i in 1..window.len() {
            let (a, b) = (window[i - 1] as f64, window[i] as f64);
            if a <= 0.0 && b > 0.0 {
                let t = (i - 1) as f64 + a / (a - b);
                if first.is_none() {
                    first = Some(t);
                }
                last = Some(t);
                count += 1;
            }
        }
        match (first, last) {
            (Some(f), Some(l)) if count >= 2 => (count - 1) as f64 * sample_rate / (l - f),
            _ => 0.0,
        }
    }

    fn run_profile_at_rate(
        profile: EngineProfile,
        freq: f32,
        rate: f64,
        seconds: usize,
    ) -> Vec<f32> {
        let handles = Engine::build(EngineConfig {
            channels: 1,
            profile,
            initial_tempo_rate: rate,
            ..EngineConfig::default()
        })
        .unwrap();
        let (mut processor, mut source) = (handles.processor, handles.source);
        let input = sine(freq, 44_100.0, 44_100 * (seconds + 1));
        let mut feed = 0usize;
        let mut out = vec![0.0f32; 256];
        let mut collected = Vec::new();
        for _ in 0..(44_100 * seconds / 256) {
            while feed < input.len() && source.occupied_frames() < 8_192 {
                let end = (feed + 8_192).min(input.len());
                feed += source.push(&input[feed..end]);
            }
            processor.process(&mut out);
            collected.extend_from_slice(&out);
        }
        collected
    }

    #[test]
    fn keylock_profile_holds_pitch_while_tape_follows_tempo() {
        let rate = 1.06;
        let tape = run_profile_at_rate(EngineProfile::Tape, 440.0, rate, 3);
        let keylock = run_profile_at_rate(EngineProfile::Keylock, 440.0, rate, 3);

        let scan = 44_100..88_200;
        let tape_freq = measure_freq(&tape[scan.clone()], 44_100.0);
        let keylock_freq = measure_freq(&keylock[scan], 44_100.0);

        // Tape mode: pitch follows tempo (440 * 1.06 = 466.4 Hz).
        assert!(
            (tape_freq - 440.0 * rate).abs() < 2.0,
            "tape pitch {tape_freq:.1} Hz should follow tempo"
        );
        // Keylock: pitch corrected back to the source (440 Hz), cents-level.
        let cents = 1200.0 * (keylock_freq / 440.0).log2();
        assert!(
            cents.abs() < 10.0,
            "keylock pitch off by {cents:.1} cents ({keylock_freq:.2} Hz)"
        );
    }

    #[test]
    fn keylock_pipeline_latency_within_budget_and_reported_exactly() {
        let handles = Engine::build(EngineConfig {
            channels: 1,
            profile: EngineProfile::Keylock,
            ..EngineConfig::default()
        })
        .unwrap();
        let (mut processor, mut source) = (handles.processor, handles.source);

        let latency = processor.pipeline_latency_frames();
        // ≤ 15 ms at 44.1 kHz (ROADMAP Stage 2 budget).
        assert!(
            latency as f64 / 44_100.0 <= 0.015,
            "keylock pipeline latency {latency} frames exceeds 15 ms"
        );

        // A silence-then-tone onset must first appear exactly at the
        // reported latency (within one internal block of smear tolerance).
        let mut input = vec![0.0f32; 32_768];
        for (i, s) in input.iter_mut().enumerate().skip(4_096) {
            *s = 0.6 * (2.0 * std::f32::consts::PI * 900.0 * (i - 4_096) as f32 / 44_100.0).sin();
        }
        source.push(&input);
        let mut out = vec![0.0f32; 16_384];
        processor.process(&mut out);

        let onset = out
            .iter()
            .position(|s| s.abs() > 1e-3)
            .expect("onset must appear");
        let expected = 4_096 + latency;
        assert!(
            (onset as i64 - expected as i64).abs() <= BLOCK_FRAMES as i64,
            "onset at {onset}, expected {expected} (reported latency {latency})"
        );
    }

    #[test]
    fn source_position_tracks_delivery_at_constant_rate() {
        let handles = Engine::build(EngineConfig {
            channels: 1,
            initial_tempo_rate: 1.25,
            ..EngineConfig::default()
        })
        .unwrap();
        let (controller, mut processor, mut source) =
            (handles.controller, handles.processor, handles.source);

        source.push(&vec![0.25f32; 32_768]);
        let mut out = vec![0.0f32; 256];
        let mut delivered = 0u64;
        for _ in 0..32 {
            processor.process(&mut out);
            delivered += 256;
        }
        let expected = delivered as f64 * 1.25;
        let got = controller.source_position();
        assert!(
            (got - expected).abs() <= 32.0 * 1.25 + 1.0,
            "source position {got} not within one feed chunk of {expected}"
        );
        assert_eq!(controller.delivered_frames(), delivered);
    }
}
