use super::{fft::PackedFft, selector::Selector};
use crate::StretchError;
use std::f32::consts::{PI, TAU};

/// Observable scheduler and synthesis state of the phase vocoder.
#[derive(Clone, Debug, Default)]
pub struct VocoderState {
    /// Source frames consumed per analysis frame.
    pub input_hop: usize,
    /// Output frames emitted per synthesis frame.
    pub output_hop: usize,
    /// Frames analyzed since reset.
    pub frame_counter: usize,
    /// Current propagated ratio.
    pub current_stretch: f32,
    /// Ratio latched after the last channel.
    pub target_stretch: f32,
    /// Persistent original-spectrum bypass flag.
    pub original_spectrum_bypass: bool,
    /// Transient synchronization countdown in source frames.
    pub transient_counter: i32,
    /// Desired transient alignment position.
    pub transient_target: i32,
    /// Selected signed transient offset within the centered window.
    pub alignment_offset: i32,
    /// Auxiliary-window normalization state.
    pub auxiliary_window_position: i32,
    /// Analysis-area correction for transient synthesis.
    pub window_area_gain: f32,
}

/// Stateful five-stage, shared-reference stereo vocoder with auxiliary spectral
/// replacement inside synthesis. No extra output STFT or per-channel tracker.
/// The scheduler can execute individual stages; the controlled-frame API runs
/// all stages for each frame.
pub struct PhaseVocoder {
    pub(super) n: usize,
    channels: usize,
    pub(super) state: VocoderState,
    pub(super) infinite: bool,
    pub(super) sync_enabled: bool,
    pub(super) allow_aux: bool,
    pub(super) do_aux: bool,
    pub(super) active: usize,
    pub(super) crossover: usize,
    alignment_position: i32,
    alignment_limit: i32,
    pub(super) main_history: Vec<Vec<f32>>,
    pub(super) original: Vec<Vec<f32>>,
    pub(super) aux_history: Vec<Vec<f32>>,
    pub(super) output_ola: Vec<Vec<f32>>,
    pub(super) input_phase: Vec<Vec<f32>>,
    pub(super) output_phase: Vec<Vec<f32>>,
    pub(super) stored: Vec<Vec<f32>>,
    pub(super) aux_ola: Vec<Vec<f32>>,
    pub(super) magnitude: Vec<Vec<f32>>,
    pub(super) history: Vec<Vec<f32>>,
    processed: Vec<f32>,
    pub(super) current_state: Vec<i32>,
    pub(super) saved_state: Vec<i32>,
    phase_work: Vec<f32>,
    reference_phase: Vec<f32>,
    pub(super) change: Vec<f32>,
    pub(super) location: Vec<f32>,
    rotation: Vec<f32>,
    cos_rotation: Vec<f32>,
    sin_rotation: Vec<f32>,
    phase_increment: Vec<f32>,
    pub(super) time_cross: Vec<f32>,
    pub(super) energy: Vec<f32>,
    scratch: Vec<f32>,
    pub(super) centered: Vec<f32>,
    expected: Vec<f32>,
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,
    normalization_product: Vec<f32>,
    normalization_reciprocal: Vec<f32>,
    normalization_accumulator: Vec<f32>,
    pub(super) window_cosine: Vec<f32>,
    reference_work: Vec<f32>,
    aux_window: Vec<f32>,
    aux_mask: Vec<f32>,
    main_mask: Vec<f32>,
    time_ramp: Vec<f32>,
    pub(super) selector: Selector,
    fft: PackedFft,
    math: super::math::Math,
}
fn principal(value: f32) -> f32 {
    let shifted = value + PI;
    let turns = (shifted * (-0.15915494)).floor();
    turns.mul_add(TAU, shifted) + PI
}
#[inline(never)]
fn scalar_sine(value: f32) -> f32 {
    value.sin()
}
#[inline(never)]
fn scalar_cosine(value: f32) -> f32 {
    value.cos()
}

#[cfg(test)]
#[path = "pv_tests.rs"]
mod native_tests;
impl PhaseVocoder {
    /// Allocate all plans/history. Size must be a power of two in 512..=8192.
    /// Configure hops before processing. The time ramp uses a fused multiply-add
    /// of the sample index with `2 / size` and `-1`. A captured vector-provider
    /// window seed supplies the window coefficients.
    pub fn new(size: usize, channels: usize) -> Result<Self, StretchError> {
        Self::new_with_numerics(size, channels, crate::Numerics::Portable)
    }
    pub fn new_with_numerics(
        size: usize,
        channels: usize,
        numerics: crate::Numerics,
    ) -> Result<Self, StretchError> {
        if !numerics.available() {
            return Err(super::invalid("numerical backend unavailable"));
        }
        super::check_channels(channels)?;
        if !size.is_power_of_two() || !(512..=8192).contains(&size) {
            return Err(super::invalid("invalid PV geometry"));
        }
        let zero = || vec![0.0; size];
        let planar = || vec![zero(); channels];
        let mut result = Self {
            n: size,
            channels,
            state: VocoderState::default(),
            infinite: false,
            sync_enabled: true,
            allow_aux: false,
            do_aux: false,
            active: size / 2,
            crossover: size / 2,
            alignment_position: 0,
            alignment_limit: 0,
            main_history: planar(),
            original: planar(),
            aux_history: planar(),
            output_ola: planar(),
            input_phase: planar(),
            output_phase: planar(),
            stored: planar(),
            aux_ola: planar(),
            magnitude: planar(),
            history: vec![vec![0.; size / 2 + 1]; 32],
            processed: zero(),
            current_state: vec![0; size],
            saved_state: vec![0; size],
            phase_work: zero(),
            reference_phase: zero(),
            change: zero(),
            location: zero(),
            rotation: zero(),
            cos_rotation: zero(),
            sin_rotation: zero(),
            phase_increment: zero(),
            time_cross: zero(),
            energy: zero(),
            scratch: zero(),
            centered: zero(),
            expected: zero(),
            analysis_window: zero(),
            synthesis_window: vec![1.0; size],
            normalization_product: zero(),
            normalization_reciprocal: zero(),
            normalization_accumulator: zero(),
            window_cosine: super::tables::cosine(size),
            reference_work: zero(),
            aux_window: zero(),
            aux_mask: zero(),
            main_mask: zero(),
            time_ramp: (0..size)
                .map(|i| (i as f32).mul_add(2.0 / size as f32, -1.0))
                .collect(),
            selector: Selector::new(size),
            fft: PackedFft::new(size, numerics),
            math: super::math::Math::new(numerics),
        };
        result.reset();
        Ok(result)
    }
    /// Retain windows/provider tables, clear histories and processing states,
    /// and seed the normalization accumulator with FLT_MIN. Target stretch,
    /// window-area gain, and alignment offset also persist.
    pub fn reset(&mut self) {
        self.state = VocoderState {
            original_spectrum_bypass: true,
            target_stretch: self.state.target_stretch,
            window_area_gain: self.state.window_area_gain,
            alignment_offset: self.state.alignment_offset,
            ..VocoderState::default()
        };
        self.infinite = false;
        self.sync_enabled = true;
        self.allow_aux = false;
        self.do_aux = false;
        self.active = self.n / 2;
        self.crossover = self.n / 2;
        self.alignment_limit = 0;
        self.alignment_position = 0;
        for matrix in [
            &mut self.main_history,
            &mut self.aux_history,
            &mut self.output_ola,
            &mut self.input_phase,
            &mut self.output_phase,
            &mut self.stored,
            &mut self.aux_ola,
            &mut self.history,
        ] {
            for channel in matrix {
                channel.fill(0.0);
            }
        }
        self.normalization_reciprocal.fill(0.0);
        self.normalization_accumulator.fill(f32::MIN_POSITIVE);
        self.current_state.fill(0);
        self.saved_state.fill(0);
        self.selector.reset();
        self.aux_window.fill(0.0);
    }
    /// Current scalar state; arrays remain in preallocated storage.
    pub fn state(&self) -> &VocoderState {
        &self.state
    }
    /// Analysis window, exposed for independent fixture comparisons.
    pub fn analysis_window(&self) -> &[f32] {
        &self.analysis_window
    }
    /// Synthesis window, including the asymmetric compression branch.
    pub fn synthesis_window(&self) -> &[f32] {
        &self.synthesis_window
    }
    /// Auxiliary synthesis window.
    pub fn auxiliary_window(&self) -> &[f32] {
        &self.aux_window
    }
    /// Exact initialized signed temporal-localization ramp.
    pub fn time_ramp(&self) -> &[f32] {
        &self.time_ramp
    }
    /// Reference bins produced by the last frame.
    pub fn references(&self) -> &[usize] {
        &self.selector.references[..self.selector.count]
    }
    /// Dense saved synthesis states, mapped on the last channel.
    pub fn saved_states(&self) -> &[i32] {
        &self.saved_state[..self.active]
    }
    /// Configure the persistent preparation and contribution flags separately.
    pub fn set_auxiliary(&mut self, prepare: bool, allow: bool) {
        self.do_aux = prepare;
        self.allow_aux = allow;
    }
    /// Set selectable crossover bin, including an empty high band.
    pub fn set_crossover_bin(&mut self, bin: usize) -> Result<(), StretchError> {
        if bin > self.n / 2 {
            return Err(super::invalid("PV crossover exceeds Nyquist"));
        }
        self.crossover = bin;
        Ok(())
    }
    /// Configure a finite ordinary analysis/synthesis hop pair. If both hops
    /// are unchanged, the new ratio argument is ignored.
    pub fn set_hops(
        &mut self,
        input_hop: usize,
        output_hop: usize,
        ratio: f32,
    ) -> Result<(), StretchError> {
        if input_hop < self.n / 64
            || input_hop > self.n / 2
            || output_hop == 0
            || output_hop > self.n / 2
            || !ratio.is_finite()
            || !(0.1..=10.0).contains(&ratio)
        {
            return Err(super::invalid("invalid PV hops/ratio"));
        }
        self.set_windows(input_hop, output_hop, ratio);
        Ok(())
    }
    fn forward_window(&mut self, ratio: f32, force: bool) {
        let exponent = if force {
            1
        } else if ratio >= 1.5 {
            if self.state.current_stretch >= 1.5 && self.state.input_hop < self.n / 4 {
                return;
            }
            3
        } else {
            if self.state.current_stretch > 0.0
                && self.state.current_stretch < 1.5
                && self.state.input_hop < self.n / 4
            {
                return;
            }
            4
        };
        for i in 0..self.n {
            let mut base = self.window_cosine[i] * (-0.46164);
            base += 0.53836;
            self.scratch[i] = base;
            let mut value = base;
            for _ in 1..exponent {
                value *= base;
            }
            self.analysis_window[i] = value;
        }
    }
    fn set_windows(&mut self, hi: usize, ho: usize, ratio: f32) {
        if self.state.output_hop == ho && self.state.input_hop == hi {
            if self.state.target_stretch != 1.0 {
                self.state.original_spectrum_bypass = false;
            }
            return;
        }
        if self.state.input_hop != hi {
            let scale = (hi as f32 * TAU) / self.n as f32;
            for (k, v) in self.expected.iter_mut().enumerate() {
                *v = k as f32 * scale;
            }
        }
        self.forward_window(ratio, hi > self.n / 8);
        if ratio >= 0.75 {
            if self.state.current_stretch < 0.75 {
                self.synthesis_window.copy_from_slice(&self.analysis_window);
            }
        } else if !(self.state.current_stretch > 0.0 && self.state.current_stretch < 1.0) {
            for i in 0..self.n / 2 {
                let base = self.window_cosine[i].mul_add(-0.5, 0.5);
                let mut power = base * base;
                let mut tiny = power < 1e-20;
                let cube = base * power;
                tiny |= cube < 1e-20;
                power = cube;
                for _ in 4..=10 {
                    power *= base;
                    tiny |= power < 1e-20;
                }
                self.synthesis_window[i] = if tiny { 0.0 } else { power };
                self.synthesis_window[self.n - 1 - i] = cube;
            }
        }
        for i in 0..self.n {
            self.normalization_product[i] = self.analysis_window[i] * self.synthesis_window[i];
        }
        self.state.auxiliary_window_position = (self.n / 2 + ho) as i32;
        if self.allow_aux && ho < self.n / 2 {
            self.aux_window[self.n / 2..].fill(0.0);
            let scale = PI / (2 * ho - 1) as f32;
            for i in self.n / 2 - ho..self.n / 2 {
                self.aux_window[self.n - 1 - i] = (scale * (i - (self.n / 2 - ho)) as f32).sin();
            }
        }
        self.state.output_hop = ho;
        self.state.input_hop = hi;
        self.state.target_stretch = ratio;
        let mut whole = 0.0;
        let mut emitted = 0.0;
        for i in 0..self.n / 2 {
            let value = self.analysis_window[self.n / 2 + i];
            whole += value;
            emitted += if i < ho { value } else { -0.0 };
        }
        self.state.window_area_gain = (whole / emitted).min(4.0);
        let half = (hi >> 1) as i32;
        self.state.transient_target =
            ((self.state.transient_counter + half) % hi as i32 - half).max(-half);
        if self.state.frame_counter == 0 {
            self.state.current_stretch = ratio;
        }
        if ratio != 1.0 {
            self.state.original_spectrum_bypass = false;
        }
    }
    /// Execute all channel stages for one configured frame. Main input must
    /// contain input_hop samples/channel; auxiliary contains output_hop. No
    /// allocation. Returns exactly output_hop; buffers are checked up front.
    pub fn process(
        &mut self,
        main: &[&[f32]],
        aux: Option<&[&[f32]]>,
        output: &mut [&mut [f32]],
    ) -> Result<usize, StretchError> {
        let hi = self.state.input_hop;
        let ho = self.state.output_hop;
        if hi == 0 || ho == 0 {
            return Err(super::invalid("configure PV hops before processing"));
        }
        if super::input_frames(main, self.channels)? != hi {
            return Err(super::invalid("PV main input length differs from hop"));
        }
        if let Some(aux) = aux {
            if super::input_frames(aux, self.channels)? != ho {
                return Err(super::invalid(
                    "PV auxiliary length differs from output hop",
                ));
            }
        }
        super::output_capacity(output, self.channels, ho)?;
        for step in 0..self.channels * 5 {
            self.process_step(main, aux, output, step as i32);
        }
        Ok(ho)
    }
    // The Direct scheduler owns validation and calls one processing stage at a
    // time. Negative steps fill channel history without analyzing a frame.
    pub(super) fn process_step(
        &mut self,
        main: &[&[f32]],
        aux: Option<&[&[f32]]>,
        output: &mut [&mut [f32]],
        step: i32,
    ) {
        #[cfg(feature = "qa")]
        let _profile = crate::processor::qa::region(4);
        #[cfg(feature = "qa")]
        let _stage_profile = crate::processor::qa::region(9 + step.max(0) as usize % 5);
        let hi = self.state.input_hop;
        let ho = self.state.output_hop;
        let have_aux = aux.is_some() && self.do_aux;
        let ch = if step < 0 {
            !step as usize
        } else {
            step as usize / 5
        };
        let stage = if step < 0 { 0 } else { step % 5 };
        self.alignment_limit = stage;
        match stage {
            0 => {
                self.main_history[ch].copy_within(hi.., 0);
                self.main_history[ch][self.n - hi..].copy_from_slice(&main[ch][..hi]);
                if ch == 0 && self.state.transient_counter > -(self.n as i32 / 2) {
                    self.state.transient_counter -= hi as i32;
                }
                if step >= 0 {
                    self.analyze(ch);
                }
            }
            1 => {
                self.phases(ch);
                if have_aux {
                    self.analyze_aux(ch, &aux.unwrap()[ch][..ho]);
                }
            }
            2 => {
                if ch == 0 {
                    self.selector.process(
                        hi,
                        self.active,
                        self.infinite,
                        &self.phase_increment,
                        &self.magnitude[0],
                        &self.saved_state,
                    );
                    self.state.frame_counter += 1;
                }
            }
            3 => {
                self.synthesize(ch, have_aux && self.allow_aux);
                if have_aux && self.allow_aux {
                    for (x, mask) in self.scratch.iter_mut().zip(&self.aux_mask) {
                        *x *= mask;
                    }
                    for (x, mask) in self.processed.iter_mut().zip(&self.main_mask) {
                        *x *= mask;
                    }
                }
            }
            4 => {
                self.overlap(ch);
                if have_aux && self.allow_aux {
                    self.fft.inverse(&mut self.scratch);
                    for ((x, window), ola) in self
                        .scratch
                        .iter_mut()
                        .zip(&self.aux_window)
                        .zip(&mut self.aux_ola[ch])
                    {
                        *x *= window;
                        *ola += *x;
                    }
                    if !self.state.original_spectrum_bypass {
                        for (output, aux) in self.output_ola[ch][..ho]
                            .iter_mut()
                            .zip(&self.aux_ola[ch][..ho])
                        {
                            *output += aux;
                        }
                    }
                    self.aux_ola[ch].copy_within(ho.., 0);
                    self.aux_ola[ch][self.n / 2..].fill(0.0);
                    self.aux_history[ch].copy_within(self.n / 2..self.n, self.n / 2 - ho);
                    self.aux_history[ch][self.n / 2..].fill(0.0);
                }
                output[ch][..ho].copy_from_slice(&self.output_ola[ch][..ho]);
                self.output_ola[ch].copy_within(ho.., 0);
                self.output_ola[ch][self.n - ho..].fill(0.0);
                if ch == self.channels - 1
                    && self.state.auxiliary_window_position == self.n as i32 / 2
                    && ho < self.n / 2
                {
                    self.aux_window[..self.n / 2].fill(0.0);
                    for i in self.n / 2 - ho..self.n / 2 {
                        self.aux_window[i] = self.aux_window[self.n - 1 - i];
                    }
                }
            }
            _ => unreachable!(),
        }
        self.alignment_position = if step < 0 { 512 } else { self.alignment_limit };
    }
    fn analyze(&mut self, ch: usize) {
        let half = self.n / 2;
        self.scratch.fill(0.0);
        self.centered.fill(0.0);
        for (dst, (sample, window)) in self.scratch[..half].iter_mut().zip(
            self.main_history[ch][half..]
                .iter()
                .zip(&self.analysis_window[half..]),
        ) {
            *dst = sample * window;
        }
        for (dst, (sample, window)) in self.scratch[half..].iter_mut().zip(
            self.main_history[ch][..half]
                .iter()
                .zip(&self.analysis_window[..half]),
        ) {
            *dst = sample * window;
        }
        self.original[ch].copy_from_slice(&self.scratch);
        self.fft.forward(&mut self.original[ch]);
        for v in &mut self.original[ch] {
            *v *= 1.0 / self.n as f32;
        }
        if ch == 0 {
            self.centered.copy_from_slice(&self.scratch);
            for (x, ramp) in self.scratch[..half].iter_mut().zip(&self.time_ramp[half..]) {
                *x *= ramp;
            }
            for (x, ramp) in self.scratch[half..].iter_mut().zip(&self.time_ramp[..half]) {
                *x *= ramp;
            }
            self.fft.forward(&mut self.scratch);
            for ((cross, scratch), original) in self
                .time_cross
                .iter_mut()
                .zip(&self.scratch)
                .zip(&self.original[0])
            {
                *cross = scratch * (1.0 / self.n as f32);
                *cross *= original;
            }
            let (real, imag) = self.time_cross.split_at_mut(half);
            for (r, i) in real[1..].iter_mut().zip(&imag[1..]) {
                *r += i;
            }
        }
    }
    fn phases(&mut self, ch: usize) {
        let half = self.n / 2;
        if self.math.is_apple() {
            self.math.polar(
                &mut self.original[ch],
                &mut self.phase_work[..half],
                &mut self.magnitude[ch][..half],
            );
        } else {
            for k in 0..half {
                let re = self.original[ch][k];
                let im = if k == 0 {
                    0.
                } else {
                    self.original[ch][half + k]
                };
                self.phase_work[k] = im.atan2(re);
                self.magnitude[ch][k] = super::math::magnitude(re, im);
            }
        }
        self.magnitude[ch][0] = self.original[ch][0].abs() * 0.5;
        self.magnitude[ch][half] = self.original[ch][half].abs() * 0.5;
        if ch == 0 {
            for k in 0..half {
                let mut residual = -(self.input_phase[0][k] + self.expected[k]);
                residual += self.phase_work[k];
                self.phase_increment[k] = principal(residual);
            }
            self.phase_increment[0] = 0.0;
            for k in 0..half {
                self.phase_increment[k] += self.expected[k];
            }
            self.history.rotate_right(1);
            self.history[0][..half + 1].copy_from_slice(&self.magnitude[0][..half + 1]);
        }
        self.phase_work[0] = if self.original[ch][0] < 0.0 { PI } else { 0.0 };
        self.phase_work[half] = if self.original[ch][half] < 0.0 {
            PI
        } else {
            0.0
        };
        self.input_phase[ch][..half + 1].copy_from_slice(&self.phase_work[..half + 1]);
    }
    fn analyze_aux(&mut self, ch: usize, input: &[f32]) {
        if ch == 0 {
            self.aux_mask.fill(0.0);
            self.aux_mask[0] = 1.0;
            self.aux_mask[1] = 1.0;
            self.main_mask.fill(1.0);
        }
        let ho = self.state.output_hop;
        self.aux_history[ch][self.n / 2..self.n / 2 + ho].copy_from_slice(input);
        for ((x, sample), window) in self
            .scratch
            .iter_mut()
            .zip(&self.aux_history[ch])
            .zip(&self.aux_window)
        {
            *x = sample * window;
        }
        self.fft.forward(&mut self.scratch);
        for v in &mut self.scratch {
            *v *= 1.0 / self.n as f32;
        }
    }
    fn synthesize(&mut self, ch: usize, allow: bool) {
        if ch == 0 && self.selector.count != 0 && self.alignment_position < self.alignment_limit {
            self.classify();
        }
        if self.state.frame_counter == 1 {
            self.state.alignment_offset = 0;
        }
        self.processed.fill(0.0);
        let mut selected = 0;
        if self.selector.count != 0 {
            if ch == 0 {
                for r in 0..self.selector.count {
                    let k = self.selector.references[r];
                    self.reference_work[r] = if self.current_state[k] == 2
                        && self.alignment_position < self.alignment_limit
                    {
                        let expected = self.expected[k];
                        let delta =
                            ((self.input_phase[0][k] - self.output_phase[0][k]) - expected) + PI;
                        let turns = (delta * (-0.15915494)).floor();
                        let wrapped = turns.mul_add(TAU, delta) + PI;
                        self.phase_increment[k] = expected + wrapped;
                        self.output_phase[0][k] + self.phase_increment[k]
                    } else {
                        let phase = self
                            .state
                            .current_stretch
                            .mul_add(self.phase_increment[k], self.output_phase[0][k]);
                        if self.current_state[k] == 2 {
                            self.current_state[k] = 0;
                        }
                        phase
                    };
                }
                for r in 0..self.selector.count {
                    self.reference_work[r] = principal(self.reference_work[r]);
                    let k = self.selector.references[r];
                    self.reference_phase[k] = self.reference_work[r];
                    self.rotation[k] = self.reference_work[r] - self.input_phase[0][k];
                    if self.math.is_apple() {
                        self.reference_work[r] = self.rotation[k];
                    } else {
                        let (s, c) = self.rotation[k].sin_cos();
                        self.sin_rotation[k] = s;
                        self.cos_rotation[k] = c;
                    }
                }
                if self.math.is_apple() {
                    let count = self.selector.count;
                    self.math.sin_cos(
                        &self.reference_work[..count],
                        &mut self.phase_work[..count],
                        &mut self.reference_phase[..count],
                    );
                    for r in 0..count {
                        let k = self.selector.references[r];
                        self.sin_rotation[k] = self.phase_work[r];
                        self.cos_rotation[k] = self.reference_phase[r];
                    }
                }
            }
            self.output_phase[ch][..self.n / 2 + 1]
                .copy_from_slice(&self.input_phase[ch][..self.n / 2 + 1]);
            let shift = (-(self.state.alignment_offset as f32)).mul_add(
                self.state.current_stretch,
                self.state.alignment_offset as f32,
            );
            // Validate channel/geometry once, exposing disjoint slices to the optimizer.
            let n = self.n;
            let output_phase = &mut self.output_phase[ch][..n];
            let original = &self.original[ch][..n];
            let stored = &mut self.stored[ch][..n];
            let magnitude = &self.magnitude[ch][..n];
            let processed = &mut self.processed[..n];
            let rotation = &self.rotation[..n];
            let current_state = &self.current_state[..n];
            let cos_rotation = &self.cos_rotation[..n];
            let sin_rotation = &self.sin_rotation[..n];
            let saved_state = &self.saved_state[..n];
            let location = &self.location[..n];
            let aux_mask = &mut self.aux_mask[..n];
            let main_mask = &mut self.main_mask[..n];
            for k in 1..self.active {
                let reference = self.selector.map[k];
                let state = current_state[reference];
                let im = self.n / 2 + k;
                output_phase[k] += rotation[reference];
                if state == 0 || state == 3 || (state == 2 && shift == 0.0) {
                    let c = cos_rotation[reference];
                    let s = sin_rotation[reference];
                    processed[k] = original[k].mul_add(c, (-original[im]) * s);
                    processed[im] = original[k].mul_add(s, original[im] * c);
                } else {
                    if state != 1 {
                        output_phase[k] =
                            ((shift * TAU) / self.n as f32).mul_add(k as f32, output_phase[k]);
                    }
                    let magnitude = if state == 1 { stored[k] } else { magnitude[k] };
                    let phase = output_phase[k];
                    processed[k] = magnitude * scalar_cosine(phase);
                    processed[im] = magnitude * scalar_sine(phase);
                }
                if state == 0 || state == 3 {
                    if saved_state[k] != 2 {
                        stored[k] = magnitude[k];
                    }
                    if ch == 0
                        && allow
                        && k >= self.crossover
                        && saved_state[k] != 2
                        && location[reference] < 0.3
                        && current_state[k] != 3
                    {
                        selected += 1;
                        aux_mask[k] = 1.0;
                        aux_mask[im] = 1.0;
                        main_mask[k] = 0.0;
                        main_mask[im] = 0.0;
                    }
                } else if state != 1 {
                    let index = (self.state.alignment_offset + self.n as i32 / 2) as usize;
                    let gain = (self.normalization_product[self.n / 2]
                        / self.normalization_product[index])
                        .min(1.5);
                    processed[k] *= self.state.window_area_gain * gain;
                    processed[im] *= self.state.window_area_gain * gain;
                }
            }
            let half = self.n / 2;
            self.processed[0] = (self.magnitude[ch][0] + self.magnitude[ch][0])
                * scalar_cosine(self.output_phase[ch][0]);
            self.processed[half] = (self.magnitude[ch][half] + self.magnitude[ch][half])
                * scalar_cosine(self.output_phase[ch][half]);
        }
        let high = self.active.saturating_sub(self.crossover);
        if ch == 0 && allow && high > 0 && (high - selected) as f32 / (high as f32) < 0.25 {
            for k in self.crossover..self.active {
                self.aux_mask[k] = 1.0;
                self.aux_mask[self.n / 2 + k] = 1.0;
                self.main_mask[k] = 0.0;
                self.main_mask[self.n / 2 + k] = 0.0;
            }
        }
        if ch == self.channels - 1 {
            for k in 1..self.active {
                self.saved_state[k] = self.current_state[self.selector.map[k]];
            }
            self.state.current_stretch = self.state.target_stretch;
        }
    }
    fn overlap(&mut self, ch: usize) {
        let half = self.n / 2;
        let ho = self.state.output_hop;
        if self.state.original_spectrum_bypass {
            self.processed.copy_from_slice(&self.original[ch]);
        }
        self.fft.inverse(&mut self.processed);
        for ((sample, window), output) in self.processed[half..]
            .iter_mut()
            .zip(&self.synthesis_window[..half])
            .zip(&mut self.output_ola[ch][..half])
        {
            *sample *= window;
            *output += *sample;
        }
        for ((sample, window), output) in self.processed[..half]
            .iter_mut()
            .zip(&self.synthesis_window[half..])
            .zip(&mut self.output_ola[ch][half..])
        {
            *sample *= window;
            *output += *sample;
        }
        if ch == 0 && self.state.auxiliary_window_position > -(half as i32) {
            for (sum, product) in self
                .normalization_accumulator
                .iter_mut()
                .zip(&self.normalization_product)
            {
                *sum += product;
            }
            for i in 0..ho {
                self.normalization_accumulator[i] = self.normalization_accumulator[i].max(1e-5);
                if !self.math.is_apple() {
                    self.normalization_reciprocal[i] = 1.0 / self.normalization_accumulator[i];
                }
            }
            if self.math.is_apple() {
                self.math.reciprocal(
                    &self.normalization_accumulator[..ho],
                    &mut self.normalization_reciprocal[..ho],
                );
            }
            self.normalization_accumulator.copy_within(ho.., 0);
            self.normalization_accumulator[self.n - ho..].fill(0.0);
            self.state.auxiliary_window_position -= ho as i32;
        }
        for (output, gain) in self.output_ola[ch][..ho]
            .iter_mut()
            .zip(&self.normalization_reciprocal[..ho])
        {
            *output *= gain;
        }
    }
}

use crate::memory::HeapBytes;
impl HeapBytes for PhaseVocoder {
    fn heap_bytes(&self) -> usize {
        self.main_history.heap_bytes()
            + self.original.heap_bytes()
            + self.aux_history.heap_bytes()
            + self.output_ola.heap_bytes()
            + self.input_phase.heap_bytes()
            + self.output_phase.heap_bytes()
            + self.stored.heap_bytes()
            + self.aux_ola.heap_bytes()
            + self.magnitude.heap_bytes()
            + self.history.heap_bytes()
            + self.processed.heap_bytes()
            + self.current_state.heap_bytes()
            + self.saved_state.heap_bytes()
            + self.phase_work.heap_bytes()
            + self.reference_phase.heap_bytes()
            + self.change.heap_bytes()
            + self.location.heap_bytes()
            + self.rotation.heap_bytes()
            + self.cos_rotation.heap_bytes()
            + self.sin_rotation.heap_bytes()
            + self.phase_increment.heap_bytes()
            + self.time_cross.heap_bytes()
            + self.energy.heap_bytes()
            + self.scratch.heap_bytes()
            + self.centered.heap_bytes()
            + self.expected.heap_bytes()
            + self.analysis_window.heap_bytes()
            + self.synthesis_window.heap_bytes()
            + self.normalization_product.heap_bytes()
            + self.normalization_reciprocal.heap_bytes()
            + self.normalization_accumulator.heap_bytes()
            + self.window_cosine.heap_bytes()
            + self.reference_work.heap_bytes()
            + self.aux_window.heap_bytes()
            + self.aux_mask.heap_bytes()
            + self.main_mask.heap_bytes()
            + self.time_ramp.heap_bytes()
            + self.selector.heap_bytes()
            + self.fft.heap_bytes()
    }
}
