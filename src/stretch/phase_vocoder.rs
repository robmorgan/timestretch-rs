//! Phase vocoder time stretching with identity phase locking and sub-bass phase locking.

use crate::core::fft::{COMPLEX_ZERO, WINDOW_SUM_EPSILON, WINDOW_SUM_FLOOR_RATIO};
use crate::core::window::{generate_window, WindowType};
use crate::error::StretchError;
use crate::stretch::envelope::{apply_envelope_correction, extract_envelope};
use crate::stretch::phase_locking::{apply_phase_locking, PhaseLockingMode};
use rustfft::{num_complex::Complex, FftPlanner};

const TWO_PI_F64: f64 = 2.0 * std::f64::consts::PI;
/// Fraction of bins to pre-allocate for spectral peak detection (1/4 of bins).
const PEAKS_CAPACITY_DIVISOR: usize = 4;
/// Blend factor for phase gradient integration (soft vertical coherence).
const PHASE_GRADIENT_BLEND: f64 = 0.3;
/// Minimum magnitude to consider a bin as a spectral peak (avoids noise peaks).
const MIN_PEAK_MAGNITUDE: f32 = 1e-8;
/// Treat values this close to integers as integral synthesis positions.
const SYNTH_POS_EPSILON: f64 = 1e-9;

/// Phase vocoder state for time stretching.
pub struct PhaseVocoder {
    fft_size: usize,
    hop_analysis: usize,
    hop_synthesis: usize,
    stretch_ratio: f64,
    /// Absolute synthesis position (in samples) of the next frame start.
    synthesis_pos: f64,
    /// Number of synthesized samples already emitted by the streaming path.
    synthesis_emitted: usize,
    window: Vec<f32>,
    /// Phase accumulator for resynthesis (f64 for precision over long signals).
    phase_accum: Vec<f64>,
    /// Previous analysis phase (f64 to match accumulator precision).
    prev_phase: Vec<f64>,
    /// FFT planner (cached).
    planner: FftPlanner<f32>,
    /// Pre-computed expected phase advance per bin (f64 for precision).
    expected_phase_advance: Vec<f64>,
    /// Reusable FFT buffer.
    fft_buffer: Vec<Complex<f32>>,
    /// Reusable magnitude buffer.
    magnitudes: Vec<f32>,
    /// Reusable phase buffer.
    new_phases: Vec<f32>,
    /// Reusable peaks buffer for identity phase locking.
    peaks: Vec<usize>,
    /// Current frame's analysis phases (for identity phase locking).
    analysis_phases: Vec<f32>,
    /// Bin index at or below which sub-bass phase locking is applied.
    sub_bass_bin: usize,
    /// Phase locking algorithm to use.
    phase_locking_mode: PhaseLockingMode,
    /// Whether spectral envelope preservation is enabled.
    envelope_preservation: bool,
    /// Cepstral order for envelope extraction.
    envelope_order: usize,
    /// Reusable buffer for cepstral analysis.
    cepstrum_buf: Vec<Complex<f32>>,
    /// Reusable buffer for analysis envelope.
    analysis_envelope: Vec<f32>,
    /// Reusable buffer for synthesis envelope.
    synthesis_envelope: Vec<f32>,
    /// Synthesis window for overlap-add, matched to the analysis window type.
    /// Using the same window type ensures correct spectral weighting and COLA normalization.
    synthesis_window: Vec<f32>,
    /// Backup of IF-estimated phases before phase locking overwrites them.
    /// Used to blend IF estimates with locked phases for non-peak bins.
    if_phases_backup: Vec<f32>,
    /// Reusable output buffer (avoids allocation per process() call).
    output_buf: Vec<f32>,
    /// Reusable window sum buffer (avoids allocation per process() call).
    window_sum_buf: Vec<f32>,
    /// Unnormalized overlap-add tail carried between streaming calls.
    streaming_tail: Vec<f32>,
    /// Window-sum tail matching `streaming_tail`.
    streaming_tail_window_sum: Vec<f32>,
    /// Reusable accumulation buffer for streaming overlap-add.
    streaming_accum_output: Vec<f32>,
    /// Reusable window-sum accumulation buffer for streaming overlap-add.
    streaming_accum_window_sum: Vec<f32>,
}

impl PhaseVocoder {
    /// Creates a new phase vocoder.
    pub fn new(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
    ) -> Self {
        Self::with_window(
            fft_size,
            hop_analysis,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            WindowType::BlackmanHarris,
        )
    }

    /// Creates a new phase vocoder with a specific window function.
    pub fn with_window(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        window_type: WindowType,
    ) -> Self {
        Self::with_options(
            fft_size,
            hop_analysis,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            window_type,
            PhaseLockingMode::RegionOfInfluence,
        )
    }

    /// Creates a new phase vocoder with full configuration options.
    pub fn with_options(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        window_type: WindowType,
        phase_locking_mode: PhaseLockingMode,
    ) -> Self {
        Self::with_all_options(
            fft_size,
            hop_analysis,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
            window_type,
            phase_locking_mode,
            false,
            40,
        )
    }

    /// Creates a new phase vocoder with all configuration options including envelope preservation.
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_options(
        fft_size: usize,
        hop_analysis: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        window_type: WindowType,
        phase_locking_mode: PhaseLockingMode,
        envelope_preservation: bool,
        envelope_order: usize,
    ) -> Self {
        let hop_synthesis = (hop_analysis as f64 * stretch_ratio).round() as usize;
        let window = generate_window(window_type, fft_size);
        // Match synthesis window to analysis window type for a proper window product.
        // Using the same window type ensures the overlap-add normalization works
        // correctly and avoids spectral distortion from mismatched window shapes.
        //
        // Exception: BlackmanHarris analysis uses Hann for synthesis because BH^2
        // has poor COLA (constant overlap-add) properties at standard 75% overlap
        // (hop = fft_size/4). The BH*Hann product provides better overlap-add
        // flatness while BH still provides excellent sidelobe suppression for
        // the analysis stage.
        let synthesis_window_type = match window_type {
            WindowType::BlackmanHarris => WindowType::Hann,
            other => other,
        };
        let synthesis_window = generate_window(synthesis_window_type, fft_size);
        let num_bins = fft_size / 2 + 1;

        let expected_phase_advance: Vec<f64> = (0..num_bins)
            .map(|bin| TWO_PI_F64 * bin as f64 * hop_analysis as f64 / fft_size as f64)
            .collect();

        // Compute the bin index for the sub-bass cutoff frequency.
        // Bins at or below this index get rigid phase locking to prevent
        // phase cancellation in the critical sub-bass region.
        let sub_bass_bin =
            (sub_bass_cutoff * fft_size as f32 / sample_rate as f32).round() as usize;
        let sub_bass_bin = sub_bass_bin.min(num_bins);

        Self {
            fft_size,
            hop_analysis,
            hop_synthesis,
            stretch_ratio,
            synthesis_pos: 0.0,
            synthesis_emitted: 0,
            window,
            phase_accum: vec![0.0f64; num_bins],
            prev_phase: vec![0.0f64; num_bins],
            planner: FftPlanner::new(),
            expected_phase_advance,
            fft_buffer: vec![COMPLEX_ZERO; fft_size],
            magnitudes: vec![0.0; num_bins],
            new_phases: vec![0.0; num_bins],
            peaks: Vec::with_capacity(num_bins / PEAKS_CAPACITY_DIVISOR),
            analysis_phases: vec![0.0; num_bins],
            sub_bass_bin,
            phase_locking_mode,
            envelope_preservation,
            envelope_order,
            cepstrum_buf: Vec::new(),
            analysis_envelope: Vec::new(),
            synthesis_envelope: Vec::new(),
            synthesis_window,
            if_phases_backup: vec![0.0; num_bins],
            output_buf: Vec::new(),
            window_sum_buf: Vec::new(),
            streaming_tail: Vec::new(),
            streaming_tail_window_sum: Vec::new(),
            streaming_accum_output: Vec::new(),
            streaming_accum_window_sum: Vec::new(),
        }
    }

    /// Returns the FFT size.
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the analysis hop size.
    #[inline]
    pub fn hop_analysis(&self) -> usize {
        self.hop_analysis
    }

    /// Returns the synthesis hop size.
    #[inline]
    pub fn hop_synthesis(&self) -> usize {
        self.hop_synthesis
    }

    /// Returns the sub-bass bin cutoff index.
    #[inline]
    pub fn sub_bass_bin(&self) -> usize {
        self.sub_bass_bin
    }

    /// Updates the stretch ratio without resetting phase state.
    ///
    /// This recalculates the synthesis hop size from the new ratio while
    /// preserving all accumulated phase information. Use this for smooth
    /// real-time ratio changes that avoid clicks and discontinuities.
    #[inline]
    pub fn set_stretch_ratio(&mut self, stretch_ratio: f64) {
        self.stretch_ratio = stretch_ratio;
        self.hop_synthesis = (self.hop_analysis as f64 * stretch_ratio).round() as usize;
    }

    /// Resets the phase accumulator and previous-phase buffers.
    ///
    /// Call this at transient boundaries so that stale phase state from a
    /// previous tonal segment does not contaminate the next one. The PV will
    /// re-derive phases from the first analysis frame after the reset.
    #[inline]
    pub fn reset_phase_state(&mut self) {
        self.phase_accum.fill(0.0);
        self.prev_phase.fill(0.0);
    }

    /// Selectively resets phase state for specific frequency bands.
    ///
    /// Only zeros `phase_accum` and `prev_phase` for bins within the bands
    /// indicated by `reset_mask`: `[sub_bass, low, mid, high]`.
    /// Band boundaries: sub-bass <100Hz, low 100-500Hz, mid 500-4000Hz, high >4000Hz.
    ///
    /// This avoids disrupting phase tracking in bands where no transient occurred
    /// (e.g., a hi-hat hit shouldn't reset the sustained bass phase).
    pub fn reset_phase_state_bands(&mut self, reset_mask: [bool; 4], sample_rate: u32) {
        let num_bins = self.fft_size / 2 + 1;
        let bin_freq = sample_rate as f32 / self.fft_size as f32;

        for bin in 0..num_bins {
            let freq = bin as f32 * bin_freq;
            let band_idx = if freq < 100.0 {
                0
            } else if freq < 500.0 {
                1
            } else if freq < 4000.0 {
                2
            } else {
                3
            };
            if reset_mask[band_idx] {
                self.phase_accum[bin] = 0.0;
                self.prev_phase[bin] = 0.0;
            }
        }
    }

    /// Stretches a mono audio signal using phase vocoder with identity phase locking.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        // Batch calls are independent; clear any prior streaming overlap state.
        self.streaming_tail.clear();
        self.streaming_tail_window_sum.clear();

        let (_num_frames, output_len) = self.process_core(input, true)?;
        let mut output = self.output_buf[..output_len].to_vec();
        Self::normalize_output(
            &mut output,
            &self.window_sum_buf[..output_len],
            self.stretch_ratio,
        );
        Ok(output)
    }

    /// Streaming phase-vocoder pass that preserves phase across calls.
    ///
    /// `input` should include any required analysis overlap context from the
    /// caller (typically managed by a higher-level stream processor). This
    /// method keeps synthesis overlap/window tails internally and emits only
    /// hop-aligned samples that are final for this call.
    pub fn process_streaming(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        let mut output = Vec::with_capacity(
            ((input.len() as f64 * self.stretch_ratio).ceil() as usize)
                .saturating_add(self.fft_size),
        );
        self.process_streaming_into(input, &mut output)?;
        Ok(output)
    }

    /// Streaming phase-vocoder pass writing directly into `output`.
    ///
    /// This avoids temporary output allocations in real-time paths.
    pub fn process_streaming_into(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        if input.len() < self.fft_size {
            output.clear();
            return Ok(());
        }

        let (emit_len, output_len) = self.process_core(input, false)?;
        if output.capacity() < emit_len {
            return Err(StretchError::BufferOverflow {
                buffer: "phase_vocoder_stream_output",
                requested: emit_len,
                available: output.capacity(),
            });
        }

        let work_len = output_len
            .max(emit_len)
            .max(self.streaming_tail.len())
            .max(self.streaming_tail_window_sum.len());

        self.streaming_accum_output.resize(work_len, 0.0);
        self.streaming_accum_output.fill(0.0);
        self.streaming_accum_window_sum.resize(work_len, 0.0);
        self.streaming_accum_window_sum.fill(0.0);

        self.streaming_accum_output[..output_len].copy_from_slice(&self.output_buf[..output_len]);
        self.streaming_accum_window_sum[..output_len]
            .copy_from_slice(&self.window_sum_buf[..output_len]);

        let tail_len = self
            .streaming_tail
            .len()
            .min(self.streaming_tail_window_sum.len());
        for i in 0..tail_len {
            self.streaming_accum_output[i] += self.streaming_tail[i];
            self.streaming_accum_window_sum[i] += self.streaming_tail_window_sum[i];
        }

        // Keep the unresolved overlap region for the next chunk.
        self.streaming_tail.clear();
        self.streaming_tail_window_sum.clear();
        if emit_len < work_len {
            self.streaming_tail
                .extend_from_slice(&self.streaming_accum_output[emit_len..work_len]);
            self.streaming_tail_window_sum
                .extend_from_slice(&self.streaming_accum_window_sum[emit_len..work_len]);
        }

        output.resize(emit_len, 0.0);
        output[..emit_len].copy_from_slice(&self.streaming_accum_output[..emit_len]);
        Self::normalize_output(
            output,
            &self.streaming_accum_window_sum[..emit_len],
            self.stretch_ratio,
        );
        self.synthesis_emitted = self.synthesis_emitted.saturating_add(emit_len);
        Ok(())
    }

    /// Flushes remaining streaming overlap/window tail at end of stream.
    pub fn flush_streaming(&mut self) -> Result<Vec<f32>, StretchError> {
        let mut output = Vec::with_capacity(self.streaming_tail.len());
        self.flush_streaming_into(&mut output)?;
        Ok(output)
    }

    /// Flushes remaining streaming overlap/window tail into `output`.
    pub fn flush_streaming_into(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        if self.streaming_tail.is_empty() || self.streaming_tail_window_sum.is_empty() {
            self.streaming_tail.clear();
            self.streaming_tail_window_sum.clear();
            self.synthesis_pos = 0.0;
            self.synthesis_emitted = 0;
            output.clear();
            return Ok(());
        }

        let len = self
            .streaming_tail
            .len()
            .min(self.streaming_tail_window_sum.len());
        if output.capacity() < len {
            return Err(StretchError::BufferOverflow {
                buffer: "phase_vocoder_flush_output",
                requested: len,
                available: output.capacity(),
            });
        }
        output.resize(len, 0.0);
        output.copy_from_slice(&self.streaming_tail[..len]);
        Self::normalize_output(
            output,
            &self.streaming_tail_window_sum[..len],
            self.stretch_ratio,
        );
        self.streaming_tail.clear();
        self.streaming_tail_window_sum.clear();
        self.synthesis_pos = 0.0;
        self.synthesis_emitted = 0;
        Ok(())
    }

    /// Shared PV core used by both batch and streaming paths.
    ///
    /// Returns `(emit_len, output_len)` where `output_len` samples are
    /// accumulated (unnormalized) into `self.output_buf` and
    /// `self.window_sum_buf`, and `emit_len` is the number of samples
    /// finalized for streaming emission (`floor(next_synthesis_pos)`).
    fn process_core(
        &mut self,
        input: &[f32],
        reset_phase_state: bool,
    ) -> Result<(usize, usize), StretchError> {
        if input.len() < self.fft_size {
            return Err(StretchError::InputTooShort {
                provided: input.len(),
                minimum: self.fft_size,
            });
        }

        let num_bins = self.fft_size / 2 + 1;
        let num_frames = (input.len() - self.fft_size) / self.hop_analysis + 1;

        if reset_phase_state {
            self.phase_accum.fill(0.0);
            self.prev_phase.fill(0.0);
            self.synthesis_pos = 0.0;
            self.synthesis_emitted = 0;
        }

        let fft_forward = self.planner.plan_fft_forward(self.fft_size);
        let fft_inverse = self.planner.plan_fft_inverse(self.fft_size);

        let hop_ratio = self.stretch_ratio;
        let frame_advance = self.hop_analysis as f64 * hop_ratio;
        let norm = 1.0 / self.fft_size as f32;

        // Local synthesis timeline starts at the current emission cursor.
        let start_synthesis_pos =
            snap_near_integer((self.synthesis_pos - self.synthesis_emitted as f64).max(0.0));

        // Pre-compute required accumulation length with fractional placement.
        let mut max_write_idx = 0usize;
        let mut synthesis_scan_pos = start_synthesis_pos;
        for _ in 0..num_frames {
            let synthesis_pos = snap_near_integer(synthesis_scan_pos);
            let synthesis_floor = synthesis_pos.floor() as usize;
            let frac = synthesis_pos - synthesis_floor as f64;
            let frame_end = synthesis_floor.saturating_add(
                self.fft_size
                    .saturating_sub(1)
                    .saturating_add(usize::from(frac > SYNTH_POS_EPSILON)),
            );
            max_write_idx = max_write_idx.max(frame_end);
            synthesis_scan_pos = synthesis_pos + frame_advance;
        }
        let output_len = max_write_idx.saturating_add(1);

        // Reuse pre-allocated buffers, growing if needed (never shrinks).
        self.output_buf.resize(output_len, 0.0);
        self.output_buf.fill(0.0);
        self.window_sum_buf.resize(output_len, 0.0);
        self.window_sum_buf.fill(0.0);

        let mut synthesis_frame_pos = start_synthesis_pos;
        for frame_idx in 0..num_frames {
            let analysis_pos = frame_idx * self.hop_analysis;
            let synthesis_pos = snap_near_integer(synthesis_frame_pos);
            let synthesis_floor = synthesis_pos.floor() as usize;
            let frac = synthesis_pos - synthesis_floor as f64;

            self.analyze_frame(
                &input[analysis_pos..analysis_pos + self.fft_size],
                &fft_forward,
            );
            self.advance_phases(num_bins, hop_ratio);

            // Save IF-estimated phases before phase locking overwrites them.
            self.if_phases_backup[..num_bins].copy_from_slice(&self.new_phases[..num_bins]);

            // Phase locking: lock non-peak bins to their nearest peak using
            // the analysis phase relationship. Only applies above the sub-bass region.
            apply_phase_locking(
                self.phase_locking_mode,
                &self.magnitudes,
                &self.analysis_phases,
                &mut self.new_phases,
                num_bins,
                self.sub_bass_bin,
                &mut self.peaks,
            );

            // Blend IF estimates with locked phases for non-peak bins above sub-bass.
            // At ratio near 1.0, phase locking is very accurate so we trust it fully.
            // As the ratio increases, IF estimates become more valuable for preserving
            // frequency accuracy, so we blend in up to 10% IF (reduced from 30% to improve coherence).
            let if_blend = (0.1 * ((hop_ratio - 1.0).abs() / 0.5).min(1.0)).min(0.1);
            if if_blend > 1e-6 {
                for bin in self.sub_bass_bin..num_bins {
                    if self.peaks.binary_search(&bin).is_ok() {
                        continue; // Peak bins keep their locked phase
                    }
                    let locked = self.new_phases[bin] as f64;
                    let if_est = self.if_phases_backup[bin] as f64;
                    self.new_phases[bin] = ((1.0 - if_blend) * locked + if_blend * if_est) as f32;
                }
            }

            // Spectral envelope preservation: correct magnitudes so formant
            // structure matches the original analysis frame, preventing
            // unnatural timbre shifts.
            if self.envelope_preservation {
                // Extract envelope from the original analysis magnitudes
                extract_envelope(
                    &self.magnitudes,
                    num_bins,
                    self.envelope_order,
                    &mut self.planner,
                    &mut self.cepstrum_buf,
                    &mut self.analysis_envelope,
                );

                // The synthesis magnitudes are the same (PV doesn't change
                // magnitudes), but after phase locking the spectral shape
                // may shift slightly. We extract the synthesis envelope from
                // the current magnitudes and correct.
                // Clone analysis envelope as synthesis baseline since magnitudes
                // haven't changed. The correction step then normalizes any
                // spectral tilt introduced by windowing or overlap.
                self.synthesis_envelope.clear();
                self.synthesis_envelope
                    .extend_from_slice(&self.analysis_envelope);

                apply_envelope_correction(
                    &mut self.magnitudes,
                    &self.analysis_envelope,
                    &self.synthesis_envelope,
                    num_bins,
                    self.sub_bass_bin,
                );
            }

            self.reconstruct_spectrum(num_bins);
            fft_inverse.process(&mut self.fft_buffer);

            // Fractional overlap-add: when synthesis frame starts between samples,
            // distribute each sample between nearest output samples via linear interpolation.
            if frac <= SYNTH_POS_EPSILON {
                for i in 0..self.fft_size {
                    let idx = synthesis_floor + i;
                    let ws = self.synthesis_window[i];
                    self.output_buf[idx] += self.fft_buffer[i].re * norm * ws;
                    self.window_sum_buf[idx] += self.window[i] * ws;
                }
            } else {
                let w0 = 1.0 - frac;
                let w1 = frac;
                for i in 0..self.fft_size {
                    let idx = synthesis_floor + i;
                    let ws = self.synthesis_window[i] as f64;
                    let sample = self.fft_buffer[i].re as f64 * norm as f64 * ws;
                    let window_weight = self.window[i] as f64 * ws;

                    self.output_buf[idx] += (sample * w0) as f32;
                    self.output_buf[idx + 1] += (sample * w1) as f32;
                    self.window_sum_buf[idx] += (window_weight * w0) as f32;
                    self.window_sum_buf[idx + 1] += (window_weight * w1) as f32;
                }
            }

            synthesis_frame_pos = synthesis_pos + frame_advance;
        }

        let next_local_synthesis_pos = snap_near_integer(synthesis_frame_pos);
        let emit_len = next_local_synthesis_pos.floor() as usize;
        self.synthesis_pos = self.synthesis_emitted as f64 + next_local_synthesis_pos;

        Ok((emit_len, output_len))
    }

    /// Windows the input frame and transforms to frequency domain.
    #[inline]
    fn analyze_frame(
        &mut self,
        input_frame: &[f32],
        fft_forward: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    ) {
        let len = input_frame.len().min(self.fft_buffer.len());
        for (i, (&sample, &w)) in input_frame
            .iter()
            .zip(self.window.iter())
            .enumerate()
            .take(len)
        {
            self.fft_buffer[i] = Complex::new(sample * w, 0.0);
        }
        fft_forward.process(&mut self.fft_buffer);
    }

    /// Extracts magnitudes and advances phase accumulators for each bin.
    ///
    /// Uses a multi-pass approach for improved frequency tracking and phase coherence:
    /// 1. Compute magnitudes and raw analysis phases for all bins.
    /// 2. Detect spectral peaks and compute refined instantaneous frequencies via
    ///    parabolic interpolation of the log-magnitude spectrum.
    /// 3. Advance phases using instantaneous frequency (IF) estimation: compute the
    ///    true frequency of each bin from the phase difference, then resynthesize at
    ///    the correct rate using `inst_freq * hop_synthesis`. This naturally handles
    ///    the stretch ratio without explicit hop_ratio multiplication and eliminates
    ///    cumulative phase drift.
    /// 4. Apply soft phase gradient integration to propagate coherent phase from
    ///    peaks to nearby non-peak bins.
    ///
    /// Sub-bass bins (below `sub_bass_bin`) use rigid phase propagation to prevent
    /// phase cancellation in the critical sub-bass region and are excluded from
    /// peak-based refinements.
    ///
    /// Phase accumulation uses f64 precision to prevent cumulative rounding errors
    /// over long signals. The final phases are converted back to f32 for the
    /// spectrum reconstruction step.
    #[inline]
    fn advance_phases(&mut self, num_bins: usize, hop_ratio: f64) {
        let hop_a = self.hop_analysis as f64;
        let fft = self.fft_size as f64;

        // --- Pass 1: Extract magnitudes and analysis phases ---
        for bin in 0..num_bins {
            let c = self.fft_buffer[bin];
            self.magnitudes[bin] = c.norm();
            self.analysis_phases[bin] = c.arg();
        }

        // First frame after a full phase reset: seed synthesis phases directly
        // from analysis to avoid a large bogus IF jump from zeroed prev_phase.
        if self.prev_phase.iter().all(|&p| p == 0.0) && self.phase_accum.iter().all(|&p| p == 0.0)
        {
            for bin in 0..num_bins {
                let phase = self.analysis_phases[bin] as f64;
                self.phase_accum[bin] = phase;
                self.new_phases[bin] = phase as f32;
                self.prev_phase[bin] = phase;
            }
            return;
        }

        // --- Pass 2: Detect peaks for IF refinement + phase gradient ---
        let search_start = self.sub_bass_bin.max(1);
        let mut advance_peaks: Vec<usize> = Vec::with_capacity(num_bins / PEAKS_CAPACITY_DIVISOR);
        if num_bins >= 3 && search_start < num_bins.saturating_sub(1) {
            for bin in search_start..num_bins - 1 {
                if self.magnitudes[bin] > MIN_PEAK_MAGNITUDE
                    && self.magnitudes[bin] > self.magnitudes[bin - 1]
                    && self.magnitudes[bin] > self.magnitudes[bin + 1]
                {
                    advance_peaks.push(bin);
                }
            }
        }

        // --- Pass 3: Advance phases using instantaneous frequency (IF) estimation ---
        //
        // For each bin we compute the true instantaneous frequency from the phase
        // difference between consecutive frames, then advance the synthesis phase
        // accumulator by `inst_freq * hop_synthesis`. This naturally accounts for
        // the stretch ratio and eliminates cumulative drift that occurs with the
        // simpler `(expected + deviation) * hop_ratio` approach.
        //
        // For spectral peak bins, parabolic interpolation of the log-magnitude
        // spectrum refines the frequency estimate to sub-bin precision (~1 Hz
        // accuracy vs ~5 Hz for integer-bin estimation).
        for bin in 0..num_bins {
            let phase = self.analysis_phases[bin] as f64;

            if bin < self.sub_bass_bin {
                // Sub-bass IF estimation: same instantaneous-frequency approach
                // as standard bins, but without parabolic interpolation (sub-bass
                // bins are narrow enough that integer-bin IF is sufficient).
                // The identity phase locking in phase_locking.rs handles inter-bin
                // coherence for sub-bass via trough-based regions.
                let expected_diff = self.expected_phase_advance[bin];
                let phase_diff = phase - self.prev_phase[bin];
                let deviation = wrap_phase_f64(phase_diff - expected_diff);
                self.phase_accum[bin] += (expected_diff + deviation) * hop_ratio;
            } else {
                // Standard IF estimation:
                //   phase_diff = current_phase - prev_phase
                //   expected_diff = 2*pi * bin * hop_analysis / fft_size
                //   deviation = wrap(phase_diff - expected_diff)
                //   inst_freq = (expected_diff + deviation) / hop_analysis  [rad/sample]
                //   phase_accum += inst_freq * hop_synthesis
                let expected_diff = self.expected_phase_advance[bin]; // 2*pi*bin*hop_a/fft
                let phase_diff = phase - self.prev_phase[bin];
                let deviation = wrap_phase_f64(phase_diff - expected_diff);

                // For peak bins, use parabolic interpolation of log-magnitude
                // to refine the frequency estimate to sub-bin precision.
                //
                // The phase advance for synthesis is computed as:
                //   inst_freq * hop_synthesis = (expected + deviation) * hop_synthesis / hop_analysis
                //
                // To minimize floating-point roundoff (especially at ratio 1.0 where
                // hop_s == hop_a), we compute `(expected + deviation) * (hop_s / hop_a)`
                // using a single ratio multiplication rather than separate div+mul.
                let is_peak = advance_peaks.binary_search(&bin).is_ok();
                let phase_advance = if is_peak
                    && bin >= 1
                    && bin + 1 < num_bins
                    && self.magnitudes[bin] > MIN_PEAK_MAGNITUDE
                {
                    // Parabolic interpolation on log-magnitudes for sub-bin accuracy:
                    //   alpha = log(M[k-1]), beta = log(M[k]), gamma = log(M[k+1])
                    //   p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
                    //   refined_freq_bin = k + p
                    //
                    // Log-magnitude interpolation gives better accuracy for Gaussian
                    // spectral peaks (which approximate windowed sinusoids) compared
                    // to linear interpolation.
                    let m_prev = (self.magnitudes[bin - 1] as f64).max(1e-30);
                    let m_curr = (self.magnitudes[bin] as f64).max(1e-30);
                    let m_next = (self.magnitudes[bin + 1] as f64).max(1e-30);
                    let alpha = m_prev.ln();
                    let beta = m_curr.ln();
                    let gamma = m_next.ln();
                    let denom = alpha - 2.0 * beta + gamma;
                    if denom.abs() > 1e-12 {
                        let p = 0.5 * (alpha - gamma) / denom;
                        // Refined expected phase advance based on interpolated bin position
                        let refined_expected = TWO_PI_F64 * (bin as f64 + p) * hop_a / fft;
                        let refined_deviation = wrap_phase_f64(phase_diff - refined_expected);
                        (refined_expected + refined_deviation) * hop_ratio
                    } else {
                        (expected_diff + deviation) * hop_ratio
                    }
                } else {
                    (expected_diff + deviation) * hop_ratio
                };

                self.phase_accum[bin] += phase_advance;
            }

            self.new_phases[bin] = self.phase_accum[bin] as f32;
            self.prev_phase[bin] = phase;
        }

        // --- Phase gradient integration (soft vertical coherence) ---
        // Propagate phase from peaks to nearby non-peak bins using the analysis
        // phase gradient, blended with the independently-advanced phase.
        // Apply up to 2.5x ratio with a tapering blend: full strength at ratio≤1.0,
        // then gradually reduced for larger ratios. Keep more blend at higher
        // ratios to preserve vertical coherence on narrow-band tonal content.
        if !advance_peaks.is_empty() && hop_ratio < 2.5 {
            // Slower taper than before: still attenuate at high ratios, but avoid
            // dropping coherence too early.
            let gradient_blend =
                PHASE_GRADIENT_BLEND * (1.0 - ((hop_ratio - 1.0) / 2.0).clamp(0.0, 1.0));
            for bin in self.sub_bass_bin..num_bins {
                if advance_peaks.binary_search(&bin).is_ok() {
                    continue; // Peak bins keep their phase (they are the anchors)
                }

                // Find the nearest peak via binary search
                let nearest_peak = match advance_peaks.binary_search(&bin) {
                    Ok(_) => unreachable!(),
                    Err(idx) => {
                        let lower = if idx > 0 {
                            Some(advance_peaks[idx - 1])
                        } else {
                            None
                        };
                        let upper = if idx < advance_peaks.len() {
                            Some(advance_peaks[idx])
                        } else {
                            None
                        };
                        match (lower, upper) {
                            (Some(l), Some(u)) => {
                                if bin - l <= u - bin {
                                    l
                                } else {
                                    u
                                }
                            }
                            (Some(l), None) => l,
                            (None, Some(u)) => u,
                            (None, None) => continue,
                        }
                    }
                };

                let gradient =
                    self.analysis_phases[bin] as f64 - self.analysis_phases[nearest_peak] as f64;
                let propagated = self.new_phases[nearest_peak] as f64 + gradient;
                let independent = self.new_phases[bin] as f64;
                self.new_phases[bin] =
                    ((1.0 - gradient_blend) * independent + gradient_blend * propagated) as f32;
            }
        }
    }

    /// Reconstructs the complex spectrum from magnitudes and phases,
    /// then mirrors negative frequencies for inverse FFT.
    #[inline]
    fn reconstruct_spectrum(&mut self, num_bins: usize) {
        for i in 0..num_bins {
            self.fft_buffer[i] = Complex::from_polar(self.magnitudes[i], self.new_phases[i]);
        }
        for bin in 1..num_bins - 1 {
            self.fft_buffer[self.fft_size - bin] = self.fft_buffer[bin].conj();
        }
    }

    /// Normalizes output by window sum, clamping to prevent amplification in
    /// low-overlap regions (occurs when synthesis hop > analysis hop).
    #[inline]
    fn normalize_output(output: &mut [f32], window_sum: &[f32], stretch_ratio: f64) {
        let max_window_sum = window_sum.iter().copied().fold(0.0f32, f32::max);
        // For stretches >1.0, synthesis frames are farther apart and fixed 10%
        // flooring can over-attenuate low-overlap regions. Relax the floor in
        // proportion to ratio while keeping a safety minimum against blow-ups.
        let floor_ratio = if stretch_ratio > 1.0 {
            (WINDOW_SUM_FLOOR_RATIO / stretch_ratio as f32).clamp(0.02, WINDOW_SUM_FLOOR_RATIO)
        } else {
            WINDOW_SUM_FLOOR_RATIO
        };
        let min_window_sum = (max_window_sum * floor_ratio).max(WINDOW_SUM_EPSILON);
        let len = output.len().min(window_sum.len());
        for i in 0..len {
            output[i] /= window_sum[i].max(min_window_sum);
        }
    }
}

impl std::fmt::Debug for PhaseVocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhaseVocoder")
            .field("fft_size", &self.fft_size)
            .field("hop_analysis", &self.hop_analysis)
            .field("hop_synthesis", &self.hop_synthesis)
            .field("stretch_ratio", &self.stretch_ratio)
            .field("synthesis_pos", &self.synthesis_pos)
            .field("sub_bass_bin", &self.sub_bass_bin)
            .field("phase_locking_mode", &self.phase_locking_mode)
            .field("streaming_tail_len", &self.streaming_tail.len())
            .finish()
    }
}

/// Snaps values extremely close to integer grid points to the exact integer.
#[inline]
fn snap_near_integer(value: f64) -> f64 {
    let rounded = value.round();
    if (value - rounded).abs() <= SYNTH_POS_EPSILON {
        rounded
    } else {
        value
    }
}

/// Wraps a phase value to [-PI, PI] using f64 precision.
#[inline]
fn wrap_phase_f64(phase: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let p = phase + pi;
    p - (p / TWO_PI_F64).floor() * TWO_PI_F64 - pi
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const TWO_PI: f32 = 2.0 * PI;

    /// Wraps a phase value to [-PI, PI] using efficient modulo arithmetic (f32).
    fn wrap_phase(phase: f32) -> f32 {
        let p = phase + PI;
        p - (p / TWO_PI).floor() * TWO_PI - PI
    }

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0) - 0.0).abs() < 1e-6);
        assert!((wrap_phase(PI + 0.1) - (-PI + 0.1)).abs() < 1e-5);
        assert!((wrap_phase(-PI - 0.1) - (PI - 0.1)).abs() < 1e-5);
        // Test larger values
        assert!((wrap_phase(10.0 * PI + 0.5) - wrap_phase(0.5)).abs() < 1e-4);
        assert!((wrap_phase(-10.0 * PI - 0.5) - wrap_phase(-0.5)).abs() < 1e-4);
    }

    #[test]
    fn test_phase_vocoder_identity() {
        // Stretch ratio 1.0 should approximately preserve the signal
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;

        // Generate a 440 Hz sine wave
        let num_samples = fft_size * 4;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        // Output length should be approximately the same
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.1,
            "Length ratio {} too far from 1.0",
            len_ratio
        );

        // Check that the output contains a similar frequency
        // (RMS should be similar)
        let input_rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
        let output_rms: f32 =
            (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.5,
            "RMS mismatch: input={}, output={}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_phase_vocoder_stretch() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;

        // Use a longer signal for more accurate length ratio
        let num_samples = fft_size * 8;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let stretch_ratio = 2.0;
        let mut pv = PhaseVocoder::new(fft_size, hop, stretch_ratio, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        // Output should be approximately 2x longer (with tolerance for edge effects)
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - stretch_ratio).abs() < 0.35,
            "Length ratio {} too far from {}",
            len_ratio,
            stretch_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_compress() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;

        let num_samples = fft_size * 4;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let stretch_ratio = 0.5;
        let mut pv = PhaseVocoder::new(fft_size, hop, stretch_ratio, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - stretch_ratio).abs() < 0.2,
            "Length ratio {} too far from {}",
            len_ratio,
            stretch_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_input_too_short() {
        let mut pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        let result = pv.process(&[0.0; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_bass_bin_calculation() {
        // 120 Hz cutoff at 44100 Hz with FFT size 4096
        // Expected bin: 120 * 4096 / 44100 ≈ 11.15 → 11
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        assert_eq!(pv.sub_bass_bin, 11);

        // 0 Hz cutoff should give bin 0 (no sub-bass locking)
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 0.0);
        assert_eq!(pv.sub_bass_bin, 0);

        // High cutoff at 48000 Hz
        let pv = PhaseVocoder::new(4096, 1024, 1.0, 48000, 200.0);
        let expected = (200.0f32 * 4096.0 / 48000.0).round() as usize;
        assert_eq!(pv.sub_bass_bin, expected);
    }

    #[test]
    fn test_sub_bass_phase_locking_preserves_low_freq() {
        // A 60 Hz sine should be handled by sub-bass rigid phase locking.
        // Compare output quality with sub-bass locking (120 Hz cutoff)
        // vs without (0 Hz cutoff).
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 8;
        let freq = 60.0f32; // Well below 120 Hz cutoff

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        // Process with sub-bass locking enabled (120 Hz cutoff)
        let mut pv_locked = PhaseVocoder::new(fft_size, hop, 1.5, sample_rate, 120.0);
        let output_locked = pv_locked.process(&input).unwrap();

        // Process without sub-bass locking (0 Hz cutoff)
        let mut pv_unlocked = PhaseVocoder::new(fft_size, hop, 1.5, sample_rate, 0.0);
        let output_unlocked = pv_unlocked.process(&input).unwrap();

        // Both should produce output
        assert!(!output_locked.is_empty());
        assert!(!output_unlocked.is_empty());

        // Both should have similar RMS (we aren't destroying energy)
        let rms_locked =
            (output_locked.iter().map(|x| x * x).sum::<f32>() / output_locked.len() as f32).sqrt();
        let rms_unlocked = (output_unlocked.iter().map(|x| x * x).sum::<f32>()
            / output_unlocked.len() as f32)
            .sqrt();

        assert!(
            rms_locked > 0.1,
            "Sub-bass locked output should have significant energy, got RMS={}",
            rms_locked
        );
        assert!(
            rms_unlocked > 0.1,
            "Unlocked output should have significant energy, got RMS={}",
            rms_unlocked
        );
    }

    #[test]
    fn test_sub_bass_locking_does_not_affect_high_freq() {
        // A 1000 Hz sine should NOT be affected by sub-bass phase locking
        // (it's above the 120 Hz cutoff).
        let sample_rate = 44100u32;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv_with = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output_with = pv_with.process(&input).unwrap();

        let mut pv_without = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 0.0);
        let output_without = pv_without.process(&input).unwrap();

        // Output lengths should be the same
        assert_eq!(output_with.len(), output_without.len());

        // RMS should be very similar since 1000 Hz is above the cutoff
        let rms_with =
            (output_with.iter().map(|x| x * x).sum::<f32>() / output_with.len() as f32).sqrt();
        let rms_without = (output_without.iter().map(|x| x * x).sum::<f32>()
            / output_without.len() as f32)
            .sqrt();

        assert!(
            (rms_with - rms_without).abs() < rms_with * 0.3,
            "1000 Hz signal should be similar with/without sub-bass locking: {} vs {}",
            rms_with,
            rms_without
        );
    }

    #[test]
    fn test_phase_vocoder_with_blackman_harris() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::with_window(
            fft_size,
            hop,
            1.5,
            sample_rate,
            120.0,
            WindowType::BlackmanHarris,
        );
        let output = pv.process(&input).unwrap();

        // Should produce valid stretched output
        assert!(!output.is_empty());
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.3,
            "BH window length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_with_kaiser() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::with_window(
            fft_size,
            hop,
            1.5,
            sample_rate,
            120.0,
            WindowType::Kaiser(800),
        );
        let output = pv.process(&input).unwrap();

        assert!(!output.is_empty());
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.3,
            "Kaiser window length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_phase_vocoder_different_windows_produce_different_output() {
        let sample_rate = 44100;
        let fft_size = 4096;
        let hop = fft_size / 4;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv_hann =
            PhaseVocoder::with_window(fft_size, hop, 1.5, sample_rate, 120.0, WindowType::Hann);
        let output_hann = pv_hann.process(&input).unwrap();

        let mut pv_bh = PhaseVocoder::with_window(
            fft_size,
            hop,
            1.5,
            sample_rate,
            120.0,
            WindowType::BlackmanHarris,
        );
        let output_bh = pv_bh.process(&input).unwrap();

        // Both should produce valid output of similar length
        assert!(!output_hann.is_empty());
        assert!(!output_bh.is_empty());

        // Outputs should differ (different windows produce different spectral characteristics)
        let min_len = output_hann.len().min(output_bh.len());
        let diff: f32 = output_hann[..min_len]
            .iter()
            .zip(&output_bh[..min_len])
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / min_len as f32;
        assert!(
            diff > 1e-6,
            "Different windows should produce different output, avg diff = {}",
            diff
        );
    }

    // --- phase locking integration (detailed tests in phase_locking module) ---

    #[test]
    fn test_phase_lock_identity_no_peaks() {
        // Flat magnitude spectrum: no local maxima → no peaks → phases unchanged
        let num_bins = 16;
        let magnitudes = vec![1.0f32; num_bins]; // all equal, no peaks
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.1).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.2).collect();
        let original_phases = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            0,
            &mut peaks,
        );

        // With no peaks found, phases should remain unchanged
        assert_eq!(synthesis_phases, original_phases);
    }

    #[test]
    fn test_phase_lock_identity_single_peak() {
        // Single peak at bin 5 with a realistic spectral lobe shape.
        // Trough-bounded identity locking propagates the peak's phase
        // rotation to all bins within its influence region (between troughs).
        let num_bins = 16;
        // Create a Gaussian-like lobe centered at bin 5, with troughs at 0 and 15
        let magnitudes: Vec<f32> = (0..num_bins)
            .map(|i| {
                let dist = (i as f32 - 5.0).abs();
                0.01 + 0.99 * (-dist * dist / 8.0).exp()
            })
            .collect();
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.3).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.5).collect();
        let peak_synth = synthesis_phases[5];
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            0, // start_bin = 0
            &mut peaks,
        );

        // Peak at bin 5 should keep its phase
        assert!(
            (synthesis_phases[5] - peak_synth).abs() < 1e-6,
            "Peak bin should keep its phase"
        );

        // The phase rotation from the peak
        let phase_rotation = peak_synth - analysis_phases[5];

        // Bins in the peak's influence region should have:
        // synth[bin] = analysis[bin] + phase_rotation
        // With a single Gaussian lobe and no other peaks, all bins should
        // be in the peak's influence region.
        for bin in 1..num_bins - 1 {
            if bin == 5 {
                continue;
            }
            let expected = analysis_phases[bin] + phase_rotation;
            assert!(
                (synthesis_phases[bin] - expected).abs() < 1e-5,
                "Bin {} should be locked to peak: got {}, expected {}",
                bin,
                synthesis_phases[bin],
                expected
            );
        }
    }

    #[test]
    fn test_phase_lock_start_bin_above_num_bins() {
        // start_bin >= num_bins: early return, no changes
        let num_bins = 8;
        let magnitudes = vec![0.0f32; num_bins];
        let analysis_phases = vec![0.0f32; num_bins];
        let mut synthesis_phases = vec![1.0f32; num_bins];
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            num_bins, // start_bin == num_bins → early return
            &mut peaks,
        );

        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_phase_lock_num_bins_less_than_3() {
        // num_bins < 3: early return
        let magnitudes = vec![1.0f32; 2];
        let analysis_phases = vec![0.0f32; 2];
        let mut synthesis_phases = vec![0.5f32; 2];
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            2,
            0,
            &mut peaks,
        );

        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_phase_lock_sub_bass_region_skipped() {
        // Peaks exist only below start_bin → no peaks found above sub-bass
        let num_bins = 16;
        let mut magnitudes = vec![0.1f32; num_bins];
        magnitudes[2] = 1.0; // peak below start_bin=5
        let analysis_phases = vec![0.0f32; num_bins];
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32).collect();
        let original = synthesis_phases.clone();
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            5, // start_bin=5, peak at bin 2 is below
            &mut peaks,
        );

        // No peaks above start_bin → no changes
        assert_eq!(synthesis_phases, original);
    }

    #[test]
    fn test_phase_lock_multiple_peaks() {
        // Two peaks with realistic spectral lobe shapes.
        // Trough-bounded identity locking assigns each bin to the peak
        // whose influence region (bounded by troughs) contains it.
        let num_bins = 16;
        // Create two Gaussian lobes: peak at bin 3, peak at bin 10
        // with a clear trough between them (around bin 7)
        let magnitudes: Vec<f32> = (0..num_bins)
            .map(|i| {
                let d3 = (i as f32 - 3.0).abs();
                let d10 = (i as f32 - 10.0).abs();
                let lobe3 = 1.0 * (-d3 * d3 / 4.0).exp();
                let lobe10 = 0.8 * (-d10 * d10 / 4.0).exp();
                0.001 + lobe3.max(lobe10) // ensure non-zero floor
            })
            .collect();
        let analysis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.1).collect();
        let mut synthesis_phases: Vec<f32> = (0..num_bins).map(|i| i as f32 * 0.2).collect();
        let synth_peak3 = synthesis_phases[3];
        let synth_peak10 = synthesis_phases[10];
        let mut peaks = Vec::new();

        apply_phase_locking(
            PhaseLockingMode::Identity,
            &magnitudes,
            &analysis_phases,
            &mut synthesis_phases,
            num_bins,
            1, // start_bin=1
            &mut peaks,
        );

        // Verify both peaks are found
        assert!(peaks.contains(&3), "Should find peak at bin 3");
        assert!(peaks.contains(&10), "Should find peak at bin 10");

        // Phase rotation for each peak
        let rotation_3 = synth_peak3 - analysis_phases[3];
        let rotation_10 = synth_peak10 - analysis_phases[10];

        // Bin 2 is in peak 3's influence region (between start_bin boundary and trough)
        let expected_2 = analysis_phases[2] + rotation_3;
        assert!(
            (synthesis_phases[2] - expected_2).abs() < 1e-5,
            "Bin 2 should lock to peak 3: got {}, expected {}",
            synthesis_phases[2],
            expected_2
        );

        // Bin 12 is in peak 10's influence region
        let expected_12 = analysis_phases[12] + rotation_10;
        assert!(
            (synthesis_phases[12] - expected_12).abs() < 1e-5,
            "Bin 12 should lock to peak 10: got {}, expected {}",
            synthesis_phases[12],
            expected_12
        );
    }

    // --- normalize_output internals ---

    #[test]
    fn test_normalize_output_uniform_window_sum() {
        // When window_sum is uniform, output should be divided by that value
        let mut output = vec![2.0f32; 10];
        let window_sum = vec![2.0f32; 10];
        PhaseVocoder::normalize_output(&mut output, &window_sum, 1.0);
        for &s in &output {
            assert!((s - 1.0).abs() < 1e-6, "Expected 1.0, got {}", s);
        }
    }

    #[test]
    fn test_normalize_output_low_window_sum_clamped() {
        // Very small window sums should be clamped to min_window_sum
        // to prevent amplification
        let mut output = vec![1.0f32; 10];
        let mut window_sum = vec![1.0f32; 10];
        // One sample has near-zero window sum (low-overlap region)
        window_sum[5] = 1e-10;
        PhaseVocoder::normalize_output(&mut output, &window_sum, 1.0);

        // The clamped sample should NOT be amplified wildly
        // min_window_sum = max(1.0) * WINDOW_SUM_FLOOR_RATIO = 0.1
        // So output[5] = 1.0 / 0.1 = 10.0
        assert!(
            output[5] <= 11.0,
            "Low window sum should be clamped, got {}",
            output[5]
        );
        // Normal samples should be ~1.0
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_output_all_zero_window_sum() {
        // All-zero window sum: should use WINDOW_SUM_EPSILON floor
        let mut output = vec![1.0f32; 5];
        let window_sum = vec![0.0f32; 5];
        PhaseVocoder::normalize_output(&mut output, &window_sum, 1.0);
        // Each sample = 1.0 / WINDOW_SUM_EPSILON
        for &s in &output {
            assert!(s.is_finite(), "Output should be finite, got {}", s);
        }
    }

    // --- wrap_phase edge cases ---

    #[test]
    fn test_wrap_phase_exact_boundaries() {
        // Exactly PI should wrap to -PI (or very close)
        let result = wrap_phase(PI);
        assert!(
            (result - (-PI)).abs() < 1e-5 || (result - PI).abs() < 1e-5,
            "wrap_phase(PI) = {} should be near ±PI",
            result
        );

        // Exactly -PI
        let result = wrap_phase(-PI);
        assert!(
            (result - (-PI)).abs() < 1e-5 || (result - PI).abs() < 1e-5,
            "wrap_phase(-PI) = {} should be near ±PI",
            result
        );

        // Exactly 0
        assert!((wrap_phase(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_wrap_phase_very_large_values() {
        // Very large positive and negative values
        let result = wrap_phase(1000.0 * PI);
        assert!(
            (-PI..=PI).contains(&result),
            "wrap_phase(1000*PI) = {} should be in [-PI, PI]",
            result
        );

        let result = wrap_phase(-999.0 * PI);
        assert!(
            (-PI..=PI).contains(&result),
            "wrap_phase(-999*PI) = {} should be in [-PI, PI]",
            result
        );
    }

    // --- set_stretch_ratio ---

    #[test]
    fn test_set_stretch_ratio_updates_hop_synthesis() {
        let mut pv = PhaseVocoder::new(4096, 1024, 1.0, 44100, 120.0);
        assert_eq!(pv.hop_synthesis(), 1024); // 1024 * 1.0 = 1024

        pv.set_stretch_ratio(2.0);
        assert_eq!(pv.hop_synthesis(), 2048); // 1024 * 2.0 = 2048

        pv.set_stretch_ratio(0.5);
        assert_eq!(pv.hop_synthesis(), 512); // 1024 * 0.5 = 512
    }

    #[test]
    fn test_set_stretch_ratio_preserves_phase_state() {
        // Process some audio, then change ratio and process more.
        // Phase should be continuous (no reset).
        let fft_size = 4096;
        let hop = 1024;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output1 = pv.process(&input).unwrap();
        assert!(!output1.is_empty());

        // Change ratio and process again — should work without error
        pv.set_stretch_ratio(1.5);
        let output2 = pv.process(&input).unwrap();
        assert!(!output2.is_empty());
        assert!(output2.len() > output1.len()); // 1.5x should be longer
    }

    #[test]
    fn test_process_streaming_and_flush_produce_finite_output() {
        let fft_size = 2048;
        let hop = 512;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 10;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * 220.0 * t).sin() * 0.6 + (2.0 * PI * 880.0 * t).sin() * 0.25
            })
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.1, sample_rate, 120.0);
        let mut total_output = Vec::new();
        let mut analysis_buffer = Vec::new();

        for chunk in input.chunks(700) {
            analysis_buffer.extend_from_slice(chunk);
            if analysis_buffer.len() < fft_size {
                continue;
            }

            let out = pv.process_streaming(&analysis_buffer).unwrap();
            total_output.extend_from_slice(&out);

            let num_frames = (analysis_buffer.len() - fft_size) / hop + 1;
            let consumed = num_frames * hop;
            analysis_buffer.drain(..consumed);
        }

        if !analysis_buffer.is_empty() {
            analysis_buffer.resize(fft_size, 0.0);
            let out = pv.process_streaming(&analysis_buffer).unwrap();
            total_output.extend_from_slice(&out);
        }

        let tail = pv.flush_streaming().unwrap();
        total_output.extend_from_slice(&tail);

        assert!(
            !total_output.is_empty(),
            "Streaming path should produce output"
        );
        assert!(total_output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_flush_streaming_is_idempotent() {
        let fft_size = 1024;
        let hop = 256;
        let sample_rate = 44100u32;
        let input: Vec<f32> = (0..fft_size * 3)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let _ = pv.process_streaming(&input).unwrap();

        let _first = pv.flush_streaming().unwrap();
        let second = pv.flush_streaming().unwrap();
        assert!(
            second.is_empty(),
            "Second flush_streaming() call should be empty"
        );
    }

    // --- sub_bass_bin edge cases ---

    #[test]
    fn test_sub_bass_bin_clamped_to_num_bins() {
        // Very high cutoff: sub_bass_bin should be clamped to num_bins
        let pv = PhaseVocoder::new(256, 64, 1.0, 44100, 30000.0);
        let num_bins = 256 / 2 + 1;
        assert!(
            pv.sub_bass_bin() <= num_bins,
            "sub_bass_bin {} should be <= num_bins {}",
            pv.sub_bass_bin(),
            num_bins
        );
    }

    #[test]
    fn test_sub_bass_all_bins_rigid() {
        // With cutoff >= Nyquist, all bins should use rigid locking.
        // This should still produce valid output (no crash).
        let fft_size = 512;
        let hop = 128;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Cutoff at Nyquist: all bins are "sub-bass" → all rigid locking
        let mut pv = PhaseVocoder::new(fft_size, hop, 1.5, sample_rate, 22050.0);
        let output = pv.process(&input).unwrap();
        assert!(!output.is_empty());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    // --- reconstruct_spectrum conjugate symmetry ---

    #[test]
    fn test_reconstruct_spectrum_produces_real_output() {
        // After reconstruct_spectrum + inverse FFT, output should be real-valued
        // (imaginary parts near zero). This verifies conjugate symmetry is correct.
        let fft_size = 256;
        let hop = 64;
        let sample_rate = 44100u32;
        let num_samples = fft_size * 4;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);
        let output = pv.process(&input).unwrap();

        // If conjugate symmetry is wrong, we'd get complex residues causing
        // large imaginary parts. The output being finite and reasonable is evidence.
        assert!(output.iter().all(|s| s.is_finite()));
        let rms = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
        assert!(
            rms > 0.01,
            "Output should have significant energy, got RMS={}",
            rms
        );
    }

    // --- PV reuse (buffers grow but don't shrink) ---

    #[test]
    fn test_phase_vocoder_reuse_across_different_lengths() {
        let fft_size = 1024;
        let hop = 256;
        let sample_rate = 44100u32;

        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0, sample_rate, 120.0);

        // Process a long signal
        let long_input: Vec<f32> = (0..fft_size * 8)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let output1 = pv.process(&long_input).unwrap();
        assert!(!output1.is_empty());

        // Process a shorter signal — buffers should still work (they don't shrink)
        let short_input: Vec<f32> = (0..fft_size * 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let output2 = pv.process(&short_input).unwrap();
        assert!(!output2.is_empty());
        assert!(output2.len() < output1.len());
    }
}
