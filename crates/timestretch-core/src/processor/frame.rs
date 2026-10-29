use super::{
    GrainSynthesis, Interpolation, MarkAnalysis, PhaseVocoder, PitchMarks, Resampler, Splitter,
};
use crate::{StretchError, core::ring_buffer::RingBuffer};

/// Shared ordinary-frame chain: splitter → waveform analysis / grains →
/// PV auxiliary replacement → final resampling. Hosts supply source time
/// explicitly and manage prefill, hold, and repetition scheduling themselves
/// when using this controlled-frame API. Both file and callback hosts use
/// these same owned states and bounded queues.
pub struct FrameProcessor {
    rate: u32,
    channels: usize,
    size: usize,
    capacity: usize,
    splitter: Splitter,
    analysis: MarkAnalysis,
    marks: PitchMarks,
    synthesis: GrainSynthesis,
    vocoder: PhaseVocoder,
    resampler: Resampler,
    main: Vec<RingBuffer<f32>>,
    split_a: Vec<Vec<f32>>,
    split_b: Vec<Vec<f32>>,
    main_block: Vec<Vec<f32>>,
    aux_block: Vec<Vec<f32>>,
    mixed_block: Vec<Vec<f32>>,
    consumed: u64,
}
impl FrameProcessor {
    /// Build the ordinary path. `block_size` is the PV size (2048..=8192), and
    /// capacity must be at least four blocks. Provider plans/storage allocate
    /// here; successful feed/render calls allocate nothing.
    pub fn new(
        sample_rate: u32,
        channels: usize,
        block_size: usize,
        capacity: usize,
    ) -> Result<Self, StretchError> {
        super::check_channels(channels)?;
        super::check_capacity(capacity)?;
        if !(8000..=192000).contains(&sample_rate)
            || !block_size.is_power_of_two()
            || !(2048..=8192).contains(&block_size)
            || capacity < 4 * block_size
        {
            return Err(super::invalid(
                "invalid timestretch frame-processor geometry",
            ));
        }
        let scratch = || vec![vec![0.0; block_size * 2]; 2];
        let mut vocoder = PhaseVocoder::new(block_size, channels)?;
        vocoder.set_auxiliary(true, true);
        vocoder.set_crossover_bin((1500.0 * block_size as f32 / sample_rate as f32) as usize)?;
        let base = block_size / 8;
        Ok(Self {
            rate: sample_rate,
            channels,
            size: block_size,
            capacity,
            splitter: Splitter::new(block_size / 4, sample_rate, channels, capacity)?,
            analysis: MarkAnalysis::new(base, channels, capacity)?,
            marks: PitchMarks::new(capacity / (base / 16) + 64)?,
            synthesis: GrainSynthesis::new(channels, capacity, block_size / 2)?,
            vocoder,
            resampler: Resampler::new(channels, block_size, Interpolation::Hermite)?,
            main: (0..channels)
                .map(|_| RingBuffer::with_capacity(capacity))
                .collect(),
            split_a: scratch(),
            split_b: scratch(),
            main_block: scratch(),
            aux_block: scratch(),
            mixed_block: scratch(),
            consumed: 0,
        })
    }
    /// Ordinary ratio quantization from the spectral core; returns the actual
    /// duration ratio. This API does not silently promise an unquantized ratio.
    pub fn set_stretch(&mut self, requested: f32) -> Result<f32, StretchError> {
        if !requested.is_finite() || !(0.25..=4.0).contains(&requested) {
            return Err(StretchError::InvalidRatio(
                "ordinary stretch must be 0.25..=4",
            ));
        }
        let shift = if f64::from(requested) <= 2.2 {
            3
        } else if f64::from(requested) <= 3.2 {
            4
        } else {
            5
        };
        let hi = self.size >> shift;
        let ho = (requested * hi as f32) as usize;
        let actual = ho as f32 / hi as f32;
        self.set_hops(hi, ho, actual)?;
        Ok(actual)
    }
    /// Explicit controlled hops, useful for comparison with native trace hosts.
    pub fn set_hops(
        &mut self,
        input_hop: usize,
        output_hop: usize,
        stretch: f32,
    ) -> Result<(), StretchError> {
        self.vocoder.set_hops(input_hop, output_hop, stretch)?;
        self.synthesis.set_stretch(stretch)?;
        let hz = if stretch <= 2.0 {
            1500.0
        } else {
            (1500.0 * (stretch - 1.0).powi(3)).min(self.rate as f32 * 0.5)
        };
        self.vocoder
            .set_crossover_bin((hz / self.rate as f32 * self.size as f32) as usize)?;
        Ok(())
    }
    /// Maximum source frames accepted by the next feed call (capped at one PV
    /// block). The hint is conservative near a partial splitter frame.
    pub fn available_input_frames(&self) -> usize {
        self.main[0]
            .available()
            .saturating_sub(self.size / 4)
            .min(
                self.synthesis
                    .available_input_frames()
                    .saturating_sub(self.size / 4),
            )
            .min(self.size)
    }
    /// Main-stream source frames consumed on this controlled schedule.
    pub fn consumed_source_frames(&self) -> u64 {
        self.consumed
    }
    /// Current PV state, including actual hops and persistent bypass.
    pub fn vocoder_state(&self) -> &super::VocoderState {
        self.vocoder.state()
    }
    /// Feed arbitrary planar source fragments. More analysis lookahead can be
    /// queued before rendering. Caller must honor the capacity hint.
    pub fn push(&mut self, input: &[&[f32]]) -> Result<usize, StretchError> {
        let frames = super::input_frames(input, self.channels)?;
        if frames > self.available_input_frames() {
            return Err(StretchError::BufferOverflow {
                buffer: "timestretch source queue",
                requested: frames,
                available: self.available_input_frames(),
            });
        }
        if self.marks.records().len() >= 257 && self.synthesis.state().last_mark_index >= 129 {
            self.marks.retain_last(self.marks.records().len() - 128)?;
            self.synthesis.discard_mark_prefix(128)?;
        }
        let expected = self.splitter.output_frames(frames)?;
        if expected > self.main[0].available() || expected > self.synthesis.available_input_frames()
        {
            return Err(super::invalid("insufficient timestretch queue capacity"));
        }
        let (al, ar) = self.split_a.split_at_mut(1);
        let (bl, br) = self.split_b.split_at_mut(1);
        let mut av = [al[0].as_mut_slice(), ar[0].as_mut_slice()];
        let mut bv = [bl[0].as_mut_slice(), br[0].as_mut_slice()];
        let written =
            self.splitter
                .process(input, &mut av[..self.channels], &mut bv[..self.channels])?;
        let band = [&self.split_b[0][..written], &self.split_b[1][..written]];
        self.analysis
            .process(&band[..self.channels], &mut self.marks)?;
        self.synthesis.push(&band[..self.channels])?;
        for ch in 0..self.channels {
            self.main[ch].push_slice(&self.split_a[ch][..written]);
        }
        Ok(frames)
    }
    /// Output capacity required for the next frame and final pitch-resampling
    /// ratio. Configure hops before rendering.
    pub fn output_frames(&self, pitch: f32) -> Result<usize, StretchError> {
        self.resampler
            .output_frames(self.vocoder.state().output_hop, pitch)
    }
    /// Render one ordinary frame at a caller-supplied absolute analysis time.
    /// Returns zero when more source/mark lookahead is needed. This explicitly
    /// exposes the scheduler boundary instead of inventing a latency contract.
    pub fn process_frame(
        &mut self,
        source_time: f64,
        pitch: f32,
        output: &mut [&mut [f32]],
    ) -> Result<usize, StretchError> {
        if !source_time.is_finite() {
            return Err(super::invalid("timestretch source time must be finite"));
        }
        let hi = self.vocoder.state().input_hop;
        let ho = self.vocoder.state().output_hop;
        if hi == 0 || ho == 0 {
            return Err(super::invalid("configure frame processor before rendering"));
        }
        let needed = self.output_frames(pitch)?;
        super::output_capacity(output, self.channels, needed)?;
        if self.main[0].len() < hi {
            return Ok(0);
        }
        self.synthesis.set_play_time(source_time)?;
        if self.synthesis.frames_needed(ho, &self.marks)? != Some(0) {
            return Ok(0);
        }
        let (left, right) = self.aux_block.split_at_mut(1);
        let mut av = [&mut left[0][..ho], &mut right[0][..ho]];
        if !self
            .synthesis
            .process(&mut av[..self.channels], &self.marks)?
        {
            return Ok(0);
        }
        for ch in 0..self.channels {
            self.main[ch].peek_slice(&mut self.main_block[ch][..hi]);
        }
        let main = [&self.main_block[0][..hi], &self.main_block[1][..hi]];
        let aux = [&self.aux_block[0][..ho], &self.aux_block[1][..ho]];
        let (left, right) = self.mixed_block.split_at_mut(1);
        let mut mixed = [&mut left[0][..ho], &mut right[0][..ho]];
        self.vocoder.process(
            &main[..self.channels],
            Some(&aux[..self.channels]),
            &mut mixed[..self.channels],
        )?;
        let mixed = [&self.mixed_block[0][..ho], &self.mixed_block[1][..ho]];
        let count = self
            .resampler
            .process(&mixed[..self.channels], output, pitch)?;
        for ch in &mut self.main {
            ch.discard(hi);
        }
        self.consumed += hi as u64;
        Ok(count)
    }
    /// Reset the owned component states and source queues. Component reset
    /// methods retain their documented sample/filter storage; hosts must manage
    /// seek and prefill separately.
    pub fn reset(&mut self) {
        self.splitter.reset();
        self.analysis.reset();
        self.marks.reset();
        self.synthesis.reset();
        self.vocoder.reset();
        self.vocoder.set_auxiliary(true, true);
        self.resampler.reset();
        for queue in &mut self.main {
            queue.clear();
        }
        self.consumed = 0;
    }
    /// Total preallocated source queue capacity per channel.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
