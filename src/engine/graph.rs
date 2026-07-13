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
/// Warm-start priming work per `process` call, as a multiple of the
/// callback's own frame count (clamped below): priming costs at most ~2x
/// a normal callback — keeping priming callbacks well inside the p99.9
/// WCET bound at every callback size — while typical prerolls still
/// complete within a handful of callbacks (< 40 ms of muted output).
const PRIME_BUDGET_CALLBACK_MULTIPLE: u64 = 2;
/// Priming budget floor (tiny callbacks still make brisk progress).
const PRIME_BUDGET_MIN_FRAMES: u64 = 256;
/// Priming budget ceiling.
const PRIME_BUDGET_MAX_FRAMES: u64 = 2_048;
/// Declick fade-in length after priming completes, in frames.
const DECLICK_FRAMES: u32 = 64;
/// Maximum timestamped retargets held in flight.
const MAX_PENDING_RETARGETS: usize = 8;

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
    /// Source frames of warm-start preroll still to prime (0 = not priming).
    prime_source_frames: f64,
    /// Output frames discarded so far (priming; a separate axis from
    /// `delivered_frames`, which stays host-visible time).
    discarded_frames: u64,
    /// Remaining declick fade-in frames after priming.
    declick_remaining: u32,
    /// Timestamped retargets awaiting their output frame, sorted ascending.
    pending_retargets: [(u64, f64); MAX_PENDING_RETARGETS],
    pending_retarget_len: usize,
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
            prime_source_frames: 0.0,
            discarded_frames: 0,
            declick_remaining: 0,
            pending_retargets: [(0, 1.0); MAX_PENDING_RETARGETS],
            pending_retarget_len: 0,
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

        // Warm-start priming: run preroll through the graph with output
        // discarded, emitting silence until the seek target is reached.
        if self.prime_source_frames > 0.0 {
            let budget = ((whole / self.channels) as u64 * PRIME_BUDGET_CALLBACK_MULTIPLE)
                .clamp(PRIME_BUDGET_MIN_FRAMES, PRIME_BUDGET_MAX_FRAMES);
            self.run_priming(budget);
            if self.prime_source_frames > 0.0 {
                out.fill(0.0);
                self.shared
                    .publish_position(self.priming_position(), self.delivered_frames);
                return;
            }
        }

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
        self.prime_source_frames = 0.0;
        self.discarded_frames = 0;
        self.declick_remaining = 0;
        self.pending_retarget_len = 0;
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

    /// Preroll frames the host should feed ahead of a warm-start target so
    /// every stage resumes converged: the chain's constant latency plus a
    /// full analysis window of history (IIR splits and correctors settle
    /// well within it). Tape chains only need the resampler margin.
    pub fn warm_start_preroll_frames(&self) -> usize {
        if self.stages.is_empty() {
            64
        } else {
            self.pipeline_latency_frames + 1_024
        }
    }

    /// Runs warm-start priming within the per-callback budget: pull the
    /// graph as normal but discard output, until the discarded output
    /// reaches the frame that plays preroll's end (computed through the
    /// timeline map, so it is exact at any tempo rate).
    fn run_priming(&mut self, budget: u64) {
        let mut budget = budget;
        loop {
            // Output frame at which the preroll's last source frame plays.
            let done_at = self
                .timeline
                .map_to_output(self.prime_source_frames)
                .map(|out_frame| out_frame.floor().max(0.0) as u64);
            if let Some(done_at) = done_at {
                if self.discarded_frames >= done_at {
                    self.prime_source_frames = 0.0;
                    self.declick_remaining = DECLICK_FRAMES;
                    return;
                }
            }
            if budget == 0 {
                return;
            }
            let want = match done_at {
                Some(done_at) => (done_at - self.discarded_frames)
                    .min(BLOCK_FRAMES as u64)
                    .min(budget) as usize,
                None => BLOCK_FRAMES.min(budget as usize),
            };
            let want_samples = want * self.channels;
            self.fill_out_fifo(want_samples);
            let popped = self
                .out_fifo
                .pop_slice(&mut self.block_scratch[..want_samples]);
            if popped == 0 {
                return; // ring dry: wait for the host to push more preroll
            }
            let frames = (popped / self.channels) as u64;
            self.discarded_frames += frames;
            budget = budget.saturating_sub(frames);
        }
    }

    /// Track position to publish while priming (playhead sweeps through
    /// the preroll; sub-30 ms in practice).
    fn priming_position(&self) -> f64 {
        self.timeline
            .map_to_source(self.discarded_frames as f64)
            .unwrap_or(0.0)
    }

    /// Frames of real (non-underrun) media delivered so far.
    fn media_delivered_frames(&self) -> u64 {
        self.delivered_frames + self.discarded_frames - self.underrun_total
    }

    /// Output-timeline frame whose source position is "now": the next frame
    /// to deliver, shifted back by the stage chain's constant delay.
    fn position_query_frame(&self) -> f64 {
        (self.media_delivered_frames() as f64 - self.pipeline_latency_frames as f64).max(0.0)
    }

    /// Drains the mailbox and applies control at the block boundary.
    /// (Timestamped sample-offset application lands in Stage 5.)
    fn apply_pending_control(&mut self) {
        let mut saw_asap_tempo = false;
        while let Some(event) = self.shared.pop_event() {
            match event.param {
                Param::TempoRate => {
                    if event.at_frame == crate::engine::control::APPLY_ASAP {
                        saw_asap_tempo = true;
                    } else if self.pending_retarget_len < MAX_PENDING_RETARGETS {
                        // Insert sorted by frame (stable for equal stamps).
                        let mut i = self.pending_retarget_len;
                        while i > 0 && self.pending_retargets[i - 1].0 > event.at_frame {
                            self.pending_retargets[i] = self.pending_retargets[i - 1];
                            i -= 1;
                        }
                        self.pending_retargets[i] = (event.at_frame, event.value);
                        self.pending_retarget_len += 1;
                    } else {
                        // Queue full: degrade to ASAP rather than lose it.
                        saw_asap_tempo = true;
                        self.shared
                            .store_tempo_latest(clamp_tempo_rate(event.value));
                    }
                }
                Param::WarmStart => {
                    self.prime_source_frames = event.value.max(0.0);
                    self.discarded_frames = 0;
                    self.declick_remaining = 0;
                }
            }
        }
        if saw_asap_tempo {
            // The latest-value register is authoritative for ASAP semantics
            // and also covers any events dropped on overflow.
            let new_rate = clamp_tempo_rate(self.shared.tempo_latest());
            if (new_rate - self.rate).abs() > MODULATION_HOLD_TRIGGER {
                self.modulation_hold_blocks = MODULATION_HOLD_BLOCKS;
            }
            self.rate = new_rate;
        }

        // Refresh the track anchor (kept when a concurrent write tears).
        if let Some(anchor) = self.ring.anchor.load() {
            self.anchor = anchor;
        }
    }

    /// Applies retargets that are due and returns the emission cap keeping
    /// the next feed from crossing the earliest still-pending timestamp.
    fn due_retargets_and_cap(&mut self) -> usize {
        while self.pending_retarget_len > 0 && self.pending_retargets[0].0 <= self.emitted_frames {
            let (_, rate) = self.pending_retargets[0];
            let rate = clamp_tempo_rate(rate);
            if (rate - self.rate).abs() > MODULATION_HOLD_TRIGGER {
                self.modulation_hold_blocks = MODULATION_HOLD_BLOCKS;
            }
            self.rate = rate;
            self.pending_retargets
                .copy_within(1..self.pending_retarget_len, 0);
            self.pending_retarget_len -= 1;
        }
        if self.pending_retarget_len > 0 {
            (self.pending_retargets[0].0 - self.emitted_frames) as usize
        } else {
            usize::MAX
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
        // Post-priming declick: a short linear fade-in on the first real
        // frames after a warm start.
        if self.declick_remaining > 0 {
            for frame in 0..needed_samples / self.channels {
                if self.declick_remaining == 0 {
                    break;
                }
                let gain = 1.0 - self.declick_remaining as f32 / DECLICK_FRAMES as f32;
                for ch in 0..self.channels {
                    out[frame * self.channels + ch] *= gain;
                }
                self.declick_remaining -= 1;
            }
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
        // Timestamped retargets: apply any that are due and cap this feed's
        // emissions so the next one lands exactly on its boundary.
        let cap = self.due_retargets_and_cap();
        let popped = self.ring.pop_slice(&mut self.feed_scratch);
        if popped == 0 {
            return None;
        }
        debug_assert_eq!(popped % self.channels, 0);
        let produced = self
            .varispeed
            .feed_capped(&self.feed_scratch[..popped], self.rate, cap);
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
    fn warm_start_resumes_at_steady_level_immediately() {
        // Ported warm-start semantics: after reset + preroll priming, the
        // FIRST audible output (past the declick fade) is at steady level —
        // no cold-start silence gap, no converging warble.
        let handles = Engine::build(EngineConfig {
            channels: 1,
            profile: EngineProfile::Keylock,
            initial_tempo_rate: 1.05,
            ..EngineConfig::default()
        })
        .unwrap();
        let (controller, mut processor, mut source) =
            (handles.controller, handles.processor, handles.source);

        // Play from the top for a while.
        let track = sine(500.0, 44_100.0, 44_100 * 6);
        let mut feed = 0usize;
        let mut out = vec![0.0f32; 256];
        for _ in 0..64 {
            while feed < track.len() && source.occupied_frames() < 8_192 {
                let end = (feed + 8_192).min(track.len());
                feed += source.push(&track[feed..end]);
            }
            processor.process(&mut out);
        }

        // Seek to 3 s: reset, re-anchor, request priming, feed preroll
        // then content (the host protocol).
        processor.reset();
        let target = 44_100 * 3;
        let preroll = processor.warm_start_preroll_frames();
        source.set_track_position((target - preroll) as u64);
        controller.warm_start(preroll as u32);
        let mut feed = target - preroll;
        let mut post_seek: Vec<f32> = Vec::new();
        for _ in 0..64 {
            while feed < track.len() && source.occupied_frames() < 8_192 {
                let end = (feed + 8_192).min(track.len());
                feed += source.push(&track[feed..end]);
            }
            processor.process(&mut out);
            post_seek.extend_from_slice(&out);
        }

        // Find the end of the priming silence.
        let first_sound = post_seek
            .iter()
            .position(|s| s.abs() > 1e-4)
            .expect("audio must resume after priming");
        assert!(
            first_sound < 256 * 8,
            "priming took too long: first sound at {first_sound}"
        );
        // Right after the 64-frame declick, the level must already be
        // steady (converged correctors, primed filters): compare the first
        // audible 1024 frames' RMS against a later settled span.
        let early = &post_seek[first_sound + 64..first_sound + 64 + 1_024];
        let late = &post_seek[first_sound + 8_192..first_sound + 8_192 + 4_096];
        let rms = |s: &[f32]| {
            (s.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / s.len() as f64).sqrt()
        };
        let ratio = rms(early) / rms(late);
        assert!(
            (0.85..1.15).contains(&ratio),
            "post-seek level not steady immediately: early/late RMS ratio {ratio:.3}"
        );
        // And everything past the declick fade is click-free (the fade
        // region itself carries the fade's own slope on top of the slew).
        let scan = &post_seek[first_sound + 64..];
        // The test sine helper is amplitude 1.0.
        let bound = 2.0 * std::f32::consts::PI * 500.0 * 1.05 / 44_100.0 * 1.5;
        for (i, w) in scan.windows(2).enumerate() {
            assert!(
                (w[1] - w[0]).abs() <= bound,
                "post-seek click at {i}: {}",
                (w[1] - w[0]).abs()
            );
        }
    }

    #[test]
    fn keylock_fades_to_varispeed_at_extreme_rates() {
        // Old Stage 13 semantics: beyond the fade band the deck behaves as
        // plain varispeed — pitch follows tempo, no corrector artifacts.
        let rate = 1.3; // deviation 30% >> fade end (22%)
        let out = run_profile_at_rate(EngineProfile::Keylock, 440.0, rate, 3);
        let freq = measure_freq(&out[44_100..88_200], 44_100.0);
        assert!(
            (freq - 440.0 * rate).abs() < 3.0,
            "extreme rate must be uncorrected varispeed: measured {freq:.1} Hz, \
             expected {:.1}",
            440.0 * rate
        );
    }

    #[test]
    fn keylock_gates_hold_at_other_sample_rates() {
        // ROADMAP Stage 5 exit criterion: 48/96 kHz produce equivalent
        // gated metrics — pitch held to cents-level at a DJ rate.
        for sample_rate in [48_000u32, 96_000] {
            let handles = Engine::build(EngineConfig {
                channels: 1,
                profile: EngineProfile::Keylock,
                sample_rate,
                initial_tempo_rate: 1.06,
                ..EngineConfig::default()
            })
            .unwrap();
            let (mut processor, mut source) = (handles.processor, handles.source);
            let sr = sample_rate as f64;
            // Raw 440 Hz source: the engine's varispeed shifts it, the
            // keylock chain must bring it back.
            let track: Vec<f32> = (0..(sr as usize * 4))
                .map(|i| 0.6 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr).sin() as f32)
                .collect();
            let mut feed = 0usize;
            let mut out = vec![0.0f32; 256];
            let mut collected = Vec::new();
            for _ in 0..(sr as usize * 3 / 256) {
                while feed < track.len() && source.occupied_frames() < 8_192 {
                    let end = (feed + 8_192).min(track.len());
                    feed += source.push(&track[feed..end]);
                }
                processor.process(&mut out);
                collected.extend_from_slice(&out);
            }
            let scan = &collected[sr as usize..sr as usize * 2];
            let freq = measure_freq(scan, sr);
            let cents = 1_200.0 * (freq / 440.0).log2();
            assert!(
                cents.abs() < 10.0,
                "{sample_rate} Hz: keylock pitch off by {cents:.1} cents"
            );
        }
    }

    #[test]
    fn fine_fader_steps_are_artifact_and_drift_free() {
        // Old Stage 13: 0.02% pitch-fader steps. Step the rate by 0.0002
        // every callback for 4 s; output must be click-free and the source
        // position must track the rate integral (no drift).
        let handles = Engine::build(EngineConfig {
            channels: 1,
            profile: EngineProfile::Keylock,
            ..EngineConfig::default()
        })
        .unwrap();
        let (controller, mut processor, mut source) =
            (handles.controller, handles.processor, handles.source);
        let input = sine(700.0, 44_100.0, 44_100 * 6);
        let mut feed = 0usize;
        let mut out = vec![0.0f32; 256];
        let mut collected = Vec::new();
        let mut rate_integral = 0.0f64;
        for cb in 0..(44_100 * 4 / 256) {
            let rate = 1.0 + 0.0002 * ((cb % 400) as f64 - 200.0).abs() / 200.0 * 100.0;
            controller.set_tempo_rate(rate);
            while feed < input.len() && source.occupied_frames() < 8_192 {
                let end = (feed + 8_192).min(input.len());
                feed += source.push(&input[feed..end]);
            }
            processor.process(&mut out);
            collected.extend_from_slice(&out);
            rate_integral += 256.0 * rate;
        }
        assert_eq!(controller.underrun_frames(), 0);
        // Click-free at the tone's slew (amplitude 1.0 helper).
        let bound = 2.0 * std::f32::consts::PI * 700.0 * 1.03 / 44_100.0 * 1.5;
        for (i, w) in collected[16_384..].windows(2).enumerate() {
            assert!(
                (w[1] - w[0]).abs() <= bound,
                "fine-step click at {i}: {}",
                (w[1] - w[0]).abs()
            );
        }
        // Drift-free: consumed source within one feed chunk + retarget
        // granularity of the rate integral.
        let expected = rate_integral;
        let got = controller.source_position();
        assert!(
            (got - expected).abs() < 320.0,
            "source position drifted: {got:.0} vs integral {expected:.0}"
        );
    }

    #[test]
    fn loop_wrap_is_gapless_across_ratios() {
        // Deck loop: the host feeds across the wrap (re-anchoring the
        // track position) with no engine reset. With a loop whose length
        // is a whole number of periods, the seam must be click-free at
        // every DJ ratio — the engine adds nothing at the boundary.
        for rate in [0.92f64, 1.0, 1.08] {
            let handles = Engine::build(EngineConfig {
                channels: 1,
                profile: EngineProfile::Keylock,
                initial_tempo_rate: rate,
                ..EngineConfig::default()
            })
            .unwrap();
            let (mut processor, mut source) = (handles.processor, handles.source);

            // 500 Hz sine; loop of exactly 200 periods (17 640 frames).
            let loop_len = 17_640usize;
            let track = sine(500.0, 44_100.0, loop_len);
            let mut cursor = 0usize;
            let mut out = vec![0.0f32; 256];
            let mut collected = Vec::new();
            source.set_track_position(0);
            for _ in 0..512 {
                while source.occupied_frames() < 8_192 {
                    let end = (cursor + 4_096).min(track.len());
                    cursor += source.push(&track[cursor..end]);
                    if cursor >= track.len() {
                        cursor = 0;
                        source.set_track_position(0);
                    }
                }
                processor.process(&mut out);
                collected.extend_from_slice(&out);
            }

            let scan = &collected[16_384..];
            let bound = 2.0 * std::f32::consts::PI * 500.0 * rate.max(1.0) as f32 / 44_100.0 * 1.5;
            let mut worst = 0.0f32;
            for w in scan.windows(2) {
                worst = worst.max((w[1] - w[0]).abs());
            }
            assert!(
                worst <= bound,
                "rate {rate}: loop seam click {worst:.5} > {bound:.5}"
            );
        }
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
