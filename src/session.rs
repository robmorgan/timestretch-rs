//! One synchronous, resumable execution adapter for every host.
use crate::session_types::*;
use crate::support::resample::{MultiSincResampler, SincInterpTable};
use crate::support::ring_buffer::RingBuffer;
use std::collections::VecDeque;
use std::sync::Arc;
use timestretch_core::memory::HeapBytes;
use timestretch_core::{NativeProcessor, RatioAdjustment};

const SOURCE_CHUNK: usize = 2048;
const NATIVE_INPUT_CAP: usize = 65536;

struct Converter {
    bank: MultiSincResampler,
    flat: Vec<f32>,
    output: Vec<Vec<f32>>,
    step: f64,
}
impl Converter {
    fn owned_buffer_bytes(&self) -> usize {
        self.flat.heap_bytes() + self.output.heap_bytes() + self.bank.owned_buffer_bytes()
    }

    fn new(channels: usize, capacity: usize, step: f64) -> Self {
        Self {
            bank: MultiSincResampler::new(
                Arc::clone(&SincInterpTable::new_stream_default()),
                channels,
            ),
            flat: vec![0.0; channels * capacity],
            output: (0..channels)
                .map(|_| Vec::with_capacity(capacity * 6 + 256))
                .collect(),
            step,
        }
    }
    fn run(&mut self, frames: usize) -> SessionResult<()> {
        self.bank
            .process_flat_capped(
                &self.flat[..frames * self.output.len()],
                frames,
                self.step,
                &mut self.output,
                usize::MAX,
            )
            .map_err(|_| SessionError::Capacity)
    }
    fn reset(&mut self) {
        self.bank.reset();
        for c in &mut self.output {
            c.clear();
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Span {
    start: f64,
    end: f64,
    source_start: f64,
    source_end: f64,
}

/// Allocates and validates everything before a host starts processing.
pub struct PreparedSession {
    session: StreamSession,
}
impl PreparedSession {
    pub fn new(mut config: SessionConfig) -> SessionResult<Self> {
        let native_rate = config.validate()?;
        config.backend = config.backend.resolve()?;
        let core = NativeProcessor::new_with_numerics(
            native_rate,
            config.channels,
            config.backend.numerics()?,
        )?;
        let capacity = core.output_capacity();
        let startup_native_frames = core.prefill_frames();
        let channels = config.channels;
        let max_commands = config.max_commands;
        let max_inflight = config.max_inflight_frames;
        let source_converter = (config.source_sample_rate != native_rate).then(|| {
            Converter::new(
                channels,
                SOURCE_CHUNK,
                config.source_sample_rate as f64 / native_rate as f64,
            )
        });
        let output_converter = (config.output_sample_rate != native_rate).then(|| {
            Converter::new(
                channels,
                capacity,
                native_rate as f64 / config.output_sample_rate as f64,
            )
        });
        let mut session = StreamSession {
            core,
            native_rate,
            config,
            startup_native_frames,
            source_converter,
            output_converter,
            input: (0..channels)
                .map(|_| RingBuffer::with_capacity(NATIVE_INPUT_CAP))
                .collect(),
            native_input: [vec![0.0; capacity], vec![0.0; capacity]],
            native_output: [vec![0.0; capacity], vec![0.0; capacity]],
            pending: RingBuffer::with_capacity(max_inflight * channels),
            pending_positions: RingBuffer::with_capacity(max_inflight),
            ledger: RingBuffer::with_capacity(max_inflight),
            commands: Vec::with_capacity(max_commands),
            applied_events: VecDeque::with_capacity(max_commands),
            spans: Vec::with_capacity(256),
            applied: AppliedParameters {
                requested: Parameters::default(),
                achieved_stretch: 1.0,
                achieved_pitch: 1.0,
                command_id: 0,
                requested_at: 0,
                applied_at: 0,
            },
            epoch: Epoch(1),
            transport: Transport::Playing,
            source_cursor: 0,
            feed_origin: 0,
            supplied_logical: 0,
            source_end: None,
            loop_region: None,
            input_ended: false,
            started: false,
            steps_left: 0,
            cycle_active: false,
            discard: 0,
            seek_discard: 0,
            native_written: 0,
            last_native_source: 0.0,
            rendered: 0,
            collected: 0,
            consumed: 0,
            position: 0.0,
            duration_anchor_output: 0,
            duration_anchor_source: 0.0,
            rendered_done: false,
            fault: None,
            counters: SessionCounters::default(),
        };
        session.source_end = session.config.source_frames;
        session.apply_parameters(session.config.parameters, 0, 0)?;
        Ok(Self { session })
    }
    pub fn config(&self) -> &SessionConfig {
        &self.session.config
    }
    pub fn timing(&self) -> TimingInfo {
        self.session.timing()
    }
    pub fn owned_memory(&self) -> OwnedMemory {
        self.session.owned_memory()
    }
    pub fn activate(self) -> StreamSession {
        self.session
    }
}

/// Owns DSP and transport semantics. Hosts own thread scheduling and device queues.
pub struct StreamSession {
    config: SessionConfig,
    core: NativeProcessor,
    native_rate: u32,
    startup_native_frames: usize,
    source_converter: Option<Converter>,
    output_converter: Option<Converter>,
    input: Vec<RingBuffer<f32>>,
    native_input: [Vec<f32>; 2],
    native_output: [Vec<f32>; 2],
    pending: RingBuffer<f32>,
    pending_positions: RingBuffer<f64>,
    ledger: RingBuffer<f64>,
    commands: Vec<Scheduled>,
    applied_events: VecDeque<AppliedParameters>,
    spans: Vec<Span>,
    applied: AppliedParameters,
    epoch: Epoch,
    transport: Transport,
    source_cursor: u64,
    feed_origin: u64,
    supplied_logical: u64,
    source_end: Option<u64>,
    loop_region: Option<(u64, u64)>,
    input_ended: bool,
    started: bool,
    steps_left: usize,
    cycle_active: bool,
    discard: usize,
    seek_discard: u64,
    native_written: u64,
    last_native_source: f64,
    rendered: u64,
    collected: u64,
    consumed: u64,
    position: f64,
    duration_anchor_output: u64,
    duration_anchor_source: f64,
    rendered_done: bool,
    fault: Option<SessionError>,
    counters: SessionCounters,
}
impl StreamSession {
    pub fn prepare(config: SessionConfig) -> SessionResult<PreparedSession> {
        PreparedSession::new(config)
    }
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }
    pub fn transport(&self) -> Transport {
        self.transport
    }
    pub fn set_transport(&mut self, transport: Transport) {
        self.transport = transport;
    }
    pub fn applied_parameters(&self) -> AppliedParameters {
        self.applied
    }
    pub fn next_applied(&mut self) -> Option<AppliedParameters> {
        self.applied_events.pop_front()
    }
    pub fn position(&self) -> f64 {
        self.position
    }
    pub fn fault(&self) -> Option<SessionError> {
        self.fault
    }
    pub fn counters(&self) -> SessionCounters {
        self.counters
    }
    pub fn output_ready(&self) -> usize {
        self.pending_positions.len()
    }
    pub fn commit_horizon(&self) -> u64 {
        self.rendered
    }
    pub fn finished(&self) -> bool {
        self.rendered_done && self.consumed == self.rendered
    }
    pub fn rendered_finished(&self) -> bool {
        self.rendered_done
    }
    pub fn owned_memory(&self) -> OwnedMemory {
        let adapter_bytes = self.input.heap_bytes()
            + self.native_input.heap_bytes()
            + self.native_output.heap_bytes()
            + self.pending.heap_bytes()
            + self.pending_positions.heap_bytes()
            + self.ledger.heap_bytes()
            + self.commands.capacity() * std::mem::size_of::<Scheduled>()
            + self.applied_events.capacity() * std::mem::size_of::<AppliedParameters>()
            + self.spans.capacity() * std::mem::size_of::<Span>()
            + self
                .source_converter
                .as_ref()
                .map_or(0, Converter::owned_buffer_bytes)
            + self
                .output_converter
                .as_ref()
                .map_or(0, Converter::owned_buffer_bytes);
        OwnedMemory {
            adapter_bytes,
            native_buffer_bytes: self.core.owned_buffer_bytes(),
            fft_plan_bytes: None,
        }
    }
    pub fn timing(&self) -> TimingInfo {
        let source_converter_lookahead_frames = self
            .source_converter
            .as_ref()
            .map_or(0, |c| c.bank.lookahead_frames(c.step));
        let output_converter_lookahead_native_frames = self
            .output_converter
            .as_ref()
            .map_or(0, |c| c.bank.lookahead_frames(c.step));
        let history = (2.0 * self.core.block_size() as f64 * self.config.source_sample_rate as f64
            / self.native_rate as f64)
            .ceil() as usize
            + 128;
        TimingInfo {
            native_sample_rate: self.native_rate,
            source_sample_rate: self.config.source_sample_rate,
            output_sample_rate: self.config.output_sample_rate,
            startup_source_frames: (self.startup_native_frames as f64
                * self.config.source_sample_rate as f64
                / self.native_rate as f64)
                .ceil() as usize
                + source_converter_lookahead_frames,
            source_converter_lookahead_frames,
            output_converter_lookahead_native_frames,
            source_lookahead_frames: history,
            seek_history_frames: history,
            output_alignment_delay_frames: 0,
            control_quantum_frames: (self.core.output_capacity() as f64
                * self.config.output_sample_rate as f64
                / self.native_rate as f64)
                .ceil() as usize,
            pending_output_frames: self.output_ready(),
            unacknowledged_output_frames: self.ledger.len(),
        }
    }
    /// Source coordinates belong to the session, including loop wraps and seeks.
    pub fn source_request(&self) -> Option<SourceRequest> {
        if self.fault.is_some()
            || self.rendered_done
            || self.input_ended
            || self.input[0].available() < SOURCE_CHUNK * 4 + 256
        {
            return None;
        }
        let limit = self.loop_region.map(|(_, end)| end).or(self.source_end);
        let frames = limit.map_or(SOURCE_CHUNK, |end| {
            SOURCE_CHUNK.min(end.saturating_sub(self.source_cursor) as usize)
        });
        if frames == 0 {
            return None;
        }
        Some(SourceRequest {
            epoch: self.epoch,
            start_frame: self.source_cursor,
            frames,
        })
    }
    /// Supply any nonempty prefix of the current request. No implicit sanitizing.
    pub fn supply(&mut self, request: SourceRequest, interleaved: &[f32]) -> SessionResult<usize> {
        if request.epoch != self.epoch || request.start_frame != self.source_cursor {
            return Err(SessionError::StaleSource);
        }
        let current = self.source_request().ok_or(SessionError::InvalidState)?;
        let ch = self.config.channels;
        if interleaved.len() % ch != 0 {
            return Err(SessionError::InvalidFormat);
        }
        let n = interleaved.len() / ch;
        if n == 0 || n > current.frames || n > request.frames {
            return Err(SessionError::Capacity);
        }
        if interleaved.iter().any(|x| !x.is_finite()) {
            return Err(SessionError::NonFiniteInput);
        }
        if let Some(c) = &mut self.source_converter {
            for channel in 0..ch {
                for i in 0..n {
                    c.flat[channel * n + i] = interleaved[i * ch + channel];
                }
            }
            c.run(n)?;
            for (ring, samples) in self.input.iter_mut().zip(&c.output) {
                if ring.push_slice(samples) != samples.len() {
                    return Err(SessionError::Capacity);
                }
            }
        } else {
            for frame in interleaved.chunks_exact(ch) {
                for (ring, &x) in self.input.iter_mut().zip(frame) {
                    ring.push(x);
                }
            }
        }
        self.source_cursor += n as u64;
        self.supplied_logical += n as u64;
        self.counters.supplied_frames += n as u64;
        if let Some((start, end)) = self.loop_region {
            if self.source_cursor == end {
                self.source_cursor = start;
            }
        } else if self.source_end == Some(self.source_cursor) {
            self.input_ended = true;
        }
        Ok(n)
    }
    /// Ordered EOF for a sequential source. Repeated finish is idempotent.
    pub fn finish_input(&mut self) -> SessionResult<()> {
        if self.loop_region.is_some() {
            return Err(SessionError::InvalidState);
        }
        if self.source_end.is_some_and(|end| end != self.source_cursor) {
            return Err(SessionError::InvalidState);
        }
        self.source_end = Some(self.source_cursor);
        self.config.source_frames = self.source_end;
        self.input_ended = true;
        Ok(())
    }
    pub fn schedule(
        &mut self,
        command_id: u64,
        at_output_frame: u64,
        parameters: Parameters,
    ) -> SessionResult<()> {
        parameters.validate()?;
        if self.commands.len() == self.config.max_commands {
            self.counters.rejected_commands += 1;
            return Err(SessionError::Capacity);
        }
        if at_output_frame < self.rendered {
            self.counters.late_commands += 1;
        }
        let index = self.commands.partition_point(|c| c.at <= at_output_frame);
        self.commands.insert(
            index,
            Scheduled {
                id: command_id,
                at: at_output_frame,
                parameters,
            },
        );
        Ok(())
    }
    fn apply_parameters(
        &mut self,
        parameters: Parameters,
        id: u64,
        requested_at: u64,
    ) -> SessionResult<()> {
        let adjust = match parameters.constraint {
            RatioConstraint::PreservePitch => RatioAdjustment::Stretch,
            RatioConstraint::PreserveDuration => RatioAdjustment::Pitch,
        };
        let pitch = parameters.pitch.get() as f32;
        let mut stretch = (1.0 / parameters.tempo.get()) as f32;
        // Valid f64 controls can round outside the combined domain when each
        // factor is converted to f32. Move duration inward by the minimum
        // representable step, preserving pitch and the existing native gates.
        // The scheduler's achieved factors below include this quantization.
        while stretch * pitch < 0.25 {
            stretch = f32::from_bits(stretch.to_bits() + 1);
        }
        while stretch * pitch > 32.0 {
            stretch = f32::from_bits(stretch.to_bits() - 1);
        }
        let (status, stretch, pitch) = self.core.set_ratio(stretch, pitch, adjust, true)?;
        if status != 0 {
            return Err(SessionError::InvalidState);
        }
        let source = self.duration_anchor_source
            + (self.rendered - self.duration_anchor_output) as f64
                * self.config.source_sample_rate as f64
                / self.config.output_sample_rate as f64
                / self.applied.achieved_stretch;
        self.duration_anchor_source = source;
        self.duration_anchor_output = self.rendered;
        self.applied = AppliedParameters {
            requested: parameters,
            achieved_stretch: stretch as f64,
            achieved_pitch: pitch as f64,
            command_id: id,
            requested_at,
            applied_at: self.rendered,
        };
        if !self.started {
            self.discard = self.core.initial_unused_frames();
        }
        Ok(())
    }
    fn target_frames(&self) -> Option<u64> {
        if self.loop_region.is_some() {
            return None;
        }
        self.source_end.map(|end| {
            let remaining = (end as f64 - self.duration_anchor_source).max(0.0);
            self.duration_anchor_output
                + (remaining
                    * self.applied.achieved_stretch
                    * self.config.output_sample_rate as f64
                    / self.config.source_sample_rate as f64)
                    .round() as u64
        })
    }
    fn pad_input(&mut self) -> SessionResult<()> {
        let n = SOURCE_CHUNK;
        if let Some(c) = &mut self.source_converter {
            c.flat[..n * self.config.channels].fill(0.0);
            c.run(n)?;
            for (ring, samples) in self.input.iter_mut().zip(&c.output) {
                if ring.push_slice(samples) != samples.len() {
                    return Err(SessionError::Capacity);
                }
            }
        } else {
            for ring in &mut self.input {
                for _ in 0..n {
                    if !ring.push(0.0) {
                        return Err(SessionError::Capacity);
                    }
                }
            }
        }
        Ok(())
    }
    /// Runs at most `work_quota` stages. A stage is indivisible, not a time promise.
    pub fn advance(&mut self, work_quota: usize) -> ProcessReport {
        let mut done = 0;
        let result = self.advance_inner(work_quota, &mut done);
        let status = match result {
            Ok(s) => s,
            Err(e) => {
                self.fault = Some(e);
                ProcessStatus::Fault
            }
        };
        ProcessReport {
            status,
            work_done: done,
            output_ready: self.output_ready(),
        }
    }
    fn advance_inner(&mut self, quota: usize, done: &mut usize) -> SessionResult<ProcessStatus> {
        if self.fault.is_some() {
            return Ok(ProcessStatus::Fault);
        }
        if self.finished() {
            self.transport = Transport::Stopped;
            return Ok(ProcessStatus::Finished);
        }
        if self.rendered_done {
            return Ok(ProcessStatus::Rendered);
        }
        if self.transport != Transport::Playing {
            return Ok(ProcessStatus::Paused);
        }
        for _ in 0..quota {
            if self
                .target_frames()
                .is_some_and(|target| self.rendered >= target)
            {
                self.rendered_done = true;
                if self.finished() {
                    self.transport = Transport::Stopped;
                }
                return Ok(if self.finished() {
                    ProcessStatus::Finished
                } else {
                    ProcessStatus::Rendered
                });
            }
            // Reserve the maximum complete collection before beginning a cycle.
            let max_output = (self.core.output_capacity() as f64
                * self.config.output_sample_rate as f64
                / self.native_rate as f64)
                .ceil() as usize
                + 256;
            if self.output_ready() + self.ledger.len() + max_output
                > self.config.max_inflight_frames
            {
                return Ok(ProcessStatus::NeedOutput);
            }
            if !self.cycle_active {
                while let Some(c) = self.commands.first().copied() {
                    if c.at <= self.rendered {
                        if self.applied_events.len() == self.config.max_commands {
                            return Ok(ProcessStatus::NeedAcknowledgement);
                        }
                        self.apply_parameters(c.parameters, c.id, c.at)?;
                        self.commands.remove(0);
                        self.applied_events.push_back(self.applied);
                    } else {
                        break;
                    }
                }
                let need = if self.started {
                    self.core.frames_needed(0)?
                } else {
                    self.core.prefill_frames()
                };
                if self.source_end == Some(self.source_cursor) && self.loop_region.is_none() {
                    self.input_ended = true;
                }
                if self.input[0].len() < need {
                    if !self.input_ended {
                        return Ok(ProcessStatus::NeedSource);
                    }
                    self.pad_input()?;
                    *done += 1;
                    continue;
                }
                if need > self.native_input[0].len() {
                    return Err(SessionError::Capacity);
                }
                for c in 0..self.config.channels {
                    self.input[c].pop_slice(&mut self.native_input[c][..need]);
                }
                let input = [&self.native_input[0][..need], &self.native_input[1][..need]];
                if !self.started {
                    let [l, r] = &mut self.native_output;
                    let mut output = [l.as_mut_slice(), r.as_mut_slice()];
                    let n = self.core.prefill(
                        &input[..self.config.channels],
                        Some(&mut output[..self.config.channels]),
                    )?;
                    self.started = true;
                    self.collect(n)?;
                    *done += 1;
                    continue;
                }
                self.core.queue(&input[..self.config.channels])?;
                self.steps_left = self.core.process_calls();
                self.cycle_active = true;
            }
            if self.steps_left > 0 {
                self.core.step()?;
                self.steps_left -= 1;
            }
            *done += 1;
            if self.steps_left == 0 {
                let [l, r] = &mut self.native_output;
                let mut output = [l.as_mut_slice(), r.as_mut_slice()];
                let n = self.core.read(&mut output[..self.config.channels])?;
                self.cycle_active = false;
                self.collect(n)?;
            }
        }
        Ok(if self.started {
            ProcessStatus::Progress
        } else {
            ProcessStatus::Priming
        })
    }
    fn collect(&mut self, n: usize) -> SessionResult<()> {
        let skip = self.discard.min(n);
        self.discard -= skip;
        let n = n - skip;
        if n == 0 {
            return Ok(());
        }
        let source_end = (self.core.state().source_time as f64
            * self.config.source_sample_rate as f64
            / self.native_rate as f64)
            .max(self.last_native_source);
        if self.spans.len() == self.spans.capacity() {
            self.spans.remove(0);
        }
        self.spans.push(Span {
            start: self.native_written as f64,
            end: (self.native_written + n as u64) as f64,
            source_start: self.last_native_source,
            source_end,
        });
        self.last_native_source = source_end;
        let (count, start, step) = if let Some(c) = &mut self.output_converter {
            let start = c.bank.next_output_source_pos();
            for channel in 0..self.config.channels {
                c.flat[channel * n..(channel + 1) * n]
                    .copy_from_slice(&self.native_output[channel][skip..skip + n]);
            }
            c.run(n)?;
            (c.output[0].len(), start, c.step)
        } else {
            (n, self.native_written as f64, 1.0)
        };
        self.native_written += n as u64;
        let target = self.target_frames();
        for i in 0..count {
            if self.seek_discard > 0 {
                self.seek_discard -= 1;
                continue;
            }
            if target.is_some_and(|target| self.rendered >= target) {
                self.rendered_done = true;
                break;
            }
            if self.pending_positions.available() == 0 {
                return Err(SessionError::Capacity);
            }
            for channel in 0..self.config.channels {
                let sample = match &self.output_converter {
                    Some(c) => c.output[channel][i],
                    None => self.native_output[channel][skip + i],
                };
                if !self.pending.push(sample) {
                    return Err(SessionError::Capacity);
                }
            }
            let at = start + (i + 1) as f64 * step;
            let span = self
                .spans
                .iter()
                .rev()
                .find(|s| s.start <= at)
                .copied()
                .unwrap_or_default();
            let t = if span.end > span.start {
                ((at - span.start) / (span.end - span.start)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let logical = span.source_start + t * (span.source_end - span.source_start);
            let position = self.map_source(logical);
            self.pending_positions.push(position);
            self.rendered += 1;
            self.counters.rendered_frames += 1;
        }
        Ok(())
    }
    fn map_source(&self, logical: f64) -> f64 {
        let source = self.feed_origin as f64 + logical;
        if let Some((start, end)) = self.loop_region {
            if source >= end as f64 {
                return start as f64 + (source - end as f64).rem_euclid((end - start) as f64);
            }
        }
        self.source_end.map_or(source, |end| source.min(end as f64))
    }
    /// Collect audio without advancing playback position. Acknowledge actual consumption.
    pub fn read_output(&mut self, interleaved: &mut [f32]) -> SessionResult<OutputStamp> {
        if interleaved.len() % self.config.channels != 0 {
            return Err(SessionError::InvalidFormat);
        }
        let frames = if self.transport == Transport::Playing {
            (interleaved.len() / self.config.channels)
                .min(self.output_ready())
                .min(self.ledger.available())
        } else {
            0
        };
        let stamp = OutputStamp {
            epoch: self.epoch,
            start_frame: self.collected,
            frames,
        };
        self.pending
            .pop_slice(&mut interleaved[..frames * self.config.channels]);
        for _ in 0..frames {
            self.ledger.push(
                self.pending_positions
                    .pop()
                    .ok_or(SessionError::InvalidState)?,
            );
        }
        self.collected += frames as u64;
        self.counters.collected_frames += frames as u64;
        Ok(stamp)
    }
    /// Cumulative receipts may be coalesced within an epoch. Older epochs cannot move the playhead.
    pub fn acknowledge(&mut self, receipt: OutputConsumption) -> SessionResult<bool> {
        if receipt.epoch.0 > self.epoch.0 {
            return Err(SessionError::InvalidReceipt);
        }
        if receipt.epoch != self.epoch {
            self.counters.stale_receipts += 1;
            return Ok(false);
        }
        if receipt.end_frame > self.collected {
            return Err(SessionError::InvalidReceipt);
        }
        if receipt.end_frame <= self.consumed {
            return Ok(false);
        }
        for _ in self.consumed..receipt.end_frame {
            self.position = self.ledger.pop().ok_or(SessionError::InvalidReceipt)?;
        }
        self.counters.consumed_frames += receipt.end_frame - self.consumed;
        self.consumed = receipt.end_frame;
        if self.finished() {
            self.transport = Transport::Stopped;
        }
        if self.finished() && self.loop_region.is_none() {
            if let Some(end) = self.source_end {
                self.position = end as f64;
            }
        }
        Ok(true)
    }
    pub fn seek(&mut self, target: u64) -> SessionResult<Epoch> {
        if self.config.source_access != SourceAccess::Seekable {
            return Err(SessionError::UnsupportedOperation);
        }
        if self.source_end.is_some_and(|end| target > end) {
            return Err(SessionError::InvalidParameter);
        }
        if self.loop_region.is_some_and(|(_, end)| target >= end) {
            return Err(SessionError::InvalidParameter);
        }
        let history = self.timing().seek_history_frames as u64;
        let origin = target.saturating_sub(history);
        self.restart_at(origin, target)?;
        Ok(self.epoch)
    }
    fn restart_at(&mut self, origin: u64, target: u64) -> SessionResult<()> {
        self.core.restart()?;
        if let Some(c) = &mut self.source_converter {
            c.reset();
        }
        if let Some(c) = &mut self.output_converter {
            c.reset();
        }
        for r in &mut self.input {
            r.clear();
        }
        self.pending.clear();
        self.pending_positions.clear();
        self.ledger.clear();
        self.commands.clear();
        self.applied_events.clear();
        self.spans.clear();
        self.epoch.0 = self
            .epoch
            .0
            .checked_add(1)
            .ok_or(SessionError::InvalidState)?;
        self.source_cursor = origin;
        self.feed_origin = origin;
        self.supplied_logical = 0;
        self.input_ended = false;
        self.started = false;
        self.steps_left = 0;
        self.cycle_active = false;
        self.native_written = 0;
        self.last_native_source = 0.0;
        self.rendered = 0;
        self.collected = 0;
        self.consumed = 0;
        self.position = target as f64;
        self.duration_anchor_output = 0;
        self.duration_anchor_source = target as f64;
        self.rendered_done = false;
        self.fault = None;
        self.apply_parameters(self.applied.requested, 0, 0)?;
        self.seek_discard = ((target - origin) as f64
            * self.applied.achieved_stretch
            * self.config.output_sample_rate as f64
            / self.config.source_sample_rate as f64)
            .floor() as u64;
        Ok(())
    }
    pub fn set_loop(&mut self, region: Option<(u64, u64)>) -> SessionResult<Epoch> {
        if self.config.source_access != SourceAccess::Seekable {
            return Err(SessionError::UnsupportedOperation);
        }
        if region.is_some_and(|(a, b)| a >= b || self.source_end.is_some_and(|n| b > n)) {
            return Err(SessionError::InvalidParameter);
        }
        let target = self.position as u64;
        self.loop_region = region;
        self.seek(region.map_or(target, |(a, b)| if target >= b { a } else { target }))
    }
    pub fn loop_region(&self) -> Option<(u64, u64)> {
        self.loop_region
    }
    /// Replace off the device thread. Returned storage must also be dropped off that thread.
    pub fn replace_prepared(&mut self, prepared: PreparedSession) -> SessionResult<Self> {
        let mut next = prepared.activate();
        if (
            next.config.source_sample_rate,
            next.config.output_sample_rate,
            next.config.channels,
        ) != (
            self.config.source_sample_rate,
            self.config.output_sample_rate,
            self.config.channels,
        ) || self.config.source_access != SourceAccess::Seekable
        {
            return Err(SessionError::InvalidFormat);
        }
        next.epoch = self.epoch;
        next.transport = self.transport;
        next.source_end = self.source_end;
        next.loop_region = self.loop_region;
        next.applied.requested = self.applied.requested;
        next.seek(self.position as u64)?;
        Ok(std::mem::replace(self, next))
    }
}
