//! Multi-resolution phase vocoder using a three-band filterbank.
//!
//! Splits audio into sub-bass (0-200 Hz), mid (200-4000 Hz), and high (4000+ Hz)
//! frequency bands using Linkwitz-Riley crossover filters, then processes each
//! band with a phase vocoder using an FFT size optimized for that frequency range:
//!
//! - **Sub-bass**: Large FFT (default 16384) for precise frequency resolution
//! - **Mid**: Medium FFT (default 4096) for balanced time-frequency trade-off
//! - **High**: Small FFT (default 1024) for sharp temporal resolution
//!
//! The stretched bands are summed to produce the final output.

use crate::core::crossover::ThreeBandSplitter;
use crate::error::StretchError;
use crate::stretch::phase_vocoder::{PerFrameFlux, PhaseVocoder};

/// Default crossover frequency between sub-bass and mid bands (Hz).
const DEFAULT_LOW_CROSSOVER: f64 = 200.0;
/// Default crossover frequency between mid and high bands (Hz).
const DEFAULT_HIGH_CROSSOVER: f64 = 4000.0;

/// Multiplier for sub-bass FFT size relative to mid FFT size.
const SUB_BASS_FFT_MULTIPLIER: usize = 4;
/// Divisor for high-band FFT size relative to mid FFT size.
const HIGH_FFT_DIVISOR: usize = 4;

/// Minimum FFT size for any band (must be a power of 2).
const MIN_FFT_SIZE: usize = 256;

/// Multi-resolution phase vocoder using three frequency bands
/// with different FFT sizes for optimal time-frequency resolution.
///
/// Each band is processed independently by its own [`PhaseVocoder`] instance
/// with an FFT size tuned for the frequency content in that range. The results
/// are summed to produce the final stretched audio.
///
/// # Example
///
/// ```
/// use timestretch::stretch::multi_resolution::MultiResolutionStretcher;
///
/// let mut stretcher = MultiResolutionStretcher::new(
///     4096,       // mid FFT size
///     1.5,        // stretch ratio
///     44100,      // sample rate
///     120.0,      // sub-bass cutoff
/// );
///
/// let input = vec![0.0f32; 44100]; // 1 second of silence
/// let output = stretcher.process(&input).unwrap();
/// assert!(output.len() > input.len()); // ~1.5x longer
/// ```
pub struct MultiResolutionStretcher {
    /// Three-band crossover splitter.
    splitter: ThreeBandSplitter,
    /// Phase vocoder for sub-bass band (large FFT).
    sub_bass_pv: PhaseVocoder,
    /// Phase vocoder for mid band (medium FFT).
    mid_pv: PhaseVocoder,
    /// Phase vocoder for high band (small FFT).
    high_pv: PhaseVocoder,
    /// FFT size for sub-bass band.
    sub_bass_fft_size: usize,
    /// FFT size for mid band.
    mid_fft_size: usize,
    /// FFT size for high band.
    high_fft_size: usize,
    /// Stretch ratio.
    stretch_ratio: f64,
    /// Pre-allocated buffer for sub-bass band input.
    sub_bass_buf: Vec<f32>,
    /// Pre-allocated buffer for mid band input.
    mid_buf: Vec<f32>,
    /// Pre-allocated buffer for high band input.
    high_buf: Vec<f32>,
    /// Band-domain mirrors of the caller's retained streaming window,
    /// one per band, kept in lockstep with the caller's rolling input.
    stream_band_bufs: [Vec<f32>; 3],
    /// Frames of the caller's window already split through the crossover.
    ///
    /// The stateful IIR splitter must see each input sample exactly once, so
    /// only `input[stream_retained_frames..]` is split on each streaming call.
    stream_retained_frames: usize,
    /// Per-band emitted-but-not-yet-summed output queues.
    ///
    /// Band PVs release resolved samples at different rates (the sub-bass
    /// PV holds the largest unresolved overlap tail), so emissions queue
    /// here until every band has produced a sample for the same synthesis
    /// position; only the aligned minimum is summed into the caller output.
    stream_pending: [Vec<f32>; 3],
    /// Scratch buffer for a single band PV's per-call emission.
    stream_band_out: Vec<f32>,
    /// Combined per-frame flux from the most recent streaming call.
    stream_last_flux: Option<PerFrameFlux>,
}

impl MultiResolutionStretcher {
    /// Creates a new multi-resolution stretcher.
    ///
    /// The `mid_fft_size` parameter sets the FFT size for the mid band (200-4000 Hz).
    /// Sub-bass uses `mid_fft_size * 4` and high uses `mid_fft_size / 4`.
    ///
    /// # Arguments
    ///
    /// * `mid_fft_size` - FFT size for the mid band (e.g., 4096)
    /// * `stretch_ratio` - Time-stretch ratio (>1.0 = slower, <1.0 = faster)
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `sub_bass_cutoff` - Sub-bass phase lock cutoff for each PV instance
    pub fn new(
        mid_fft_size: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
    ) -> Self {
        let sub_bass_fft = (mid_fft_size * SUB_BASS_FFT_MULTIPLIER).max(MIN_FFT_SIZE);
        let high_fft = (mid_fft_size / HIGH_FFT_DIVISOR).max(MIN_FFT_SIZE);

        let sub_bass_hop = sub_bass_fft / 4;
        let mid_hop = mid_fft_size / 4;
        let high_hop = high_fft / 4;

        let sub_bass_pv = PhaseVocoder::new(
            sub_bass_fft,
            sub_bass_hop,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
        );
        let mid_pv = PhaseVocoder::new(
            mid_fft_size,
            mid_hop,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
        );
        let high_pv = PhaseVocoder::new(
            high_fft,
            high_hop,
            stretch_ratio,
            sample_rate,
            sub_bass_cutoff,
        );

        Self {
            splitter: ThreeBandSplitter::new(
                DEFAULT_LOW_CROSSOVER,
                DEFAULT_HIGH_CROSSOVER,
                sample_rate,
            ),
            sub_bass_pv,
            mid_pv,
            high_pv,
            sub_bass_fft_size: sub_bass_fft,
            mid_fft_size,
            high_fft_size: high_fft,
            stretch_ratio,
            sub_bass_buf: Vec::new(),
            mid_buf: Vec::new(),
            high_buf: Vec::new(),
            stream_band_bufs: [Vec::new(), Vec::new(), Vec::new()],
            stream_retained_frames: 0,
            stream_pending: [Vec::new(), Vec::new(), Vec::new()],
            stream_band_out: Vec::new(),
            stream_last_flux: None,
        }
    }

    /// Creates a multi-resolution stretcher with a cap on the sub-bass FFT.
    ///
    /// Identical to [`Self::new`] except that the derived sub-bass FFT size
    /// (`mid_fft_size * 4`) is clamped to `sub_bass_fft_cap`. Used by the
    /// streaming path, where a 16384-sample sub-bass window may exceed the
    /// latency budget of the selected stream profile.
    pub fn with_sub_bass_fft_cap(
        mid_fft_size: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        sub_bass_fft_cap: usize,
    ) -> Self {
        let mut s = Self::new(mid_fft_size, stretch_ratio, sample_rate, sub_bass_cutoff);
        // The streaming consumption cadence is keyed to the sub-bass hop, so
        // the sub-bass FFT must stay a power-of-two multiple of the mid FFT
        // (>= mid) for all band hops to stay mutually aligned.
        let capped = s
            .sub_bass_fft_size
            .min(sub_bass_fft_cap.max(MIN_FFT_SIZE))
            .max(mid_fft_size);
        if capped != s.sub_bass_fft_size {
            s.sub_bass_fft_size = capped;
            s.sub_bass_pv = PhaseVocoder::new(
                capped,
                capped / 4,
                stretch_ratio,
                sample_rate,
                sub_bass_cutoff,
            );
        }
        s
    }

    /// Creates a new multi-resolution stretcher with custom crossover frequencies.
    ///
    /// # Arguments
    ///
    /// * `mid_fft_size` - FFT size for the mid band
    /// * `stretch_ratio` - Time-stretch ratio
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `sub_bass_cutoff` - Sub-bass phase lock cutoff for each PV instance
    /// * `low_crossover` - Crossover frequency between sub-bass and mid (Hz)
    /// * `high_crossover` - Crossover frequency between mid and high (Hz)
    #[allow(clippy::too_many_arguments)]
    pub fn with_crossover_freqs(
        mid_fft_size: usize,
        stretch_ratio: f64,
        sample_rate: u32,
        sub_bass_cutoff: f32,
        low_crossover: f64,
        high_crossover: f64,
    ) -> Self {
        let mut s = Self::new(mid_fft_size, stretch_ratio, sample_rate, sub_bass_cutoff);
        s.splitter = ThreeBandSplitter::new(low_crossover, high_crossover, sample_rate);
        s
    }

    /// Updates the stretch ratio for all three bands.
    pub fn set_stretch_ratio(&mut self, ratio: f64) {
        self.stretch_ratio = ratio;
        self.sub_bass_pv.set_stretch_ratio(ratio);
        self.mid_pv.set_stretch_ratio(ratio);
        self.high_pv.set_stretch_ratio(ratio);
    }

    /// Enables or disables adaptive phase-lock mode switching on all band vocoders.
    pub fn set_adaptive_phase_locking(&mut self, enabled: bool) {
        self.sub_bass_pv.set_adaptive_phase_locking(enabled);
        self.mid_pv.set_adaptive_phase_locking(enabled);
        self.high_pv.set_adaptive_phase_locking(enabled);
    }

    /// Declares smooth small-step ratio updates on all band vocoders (see
    /// [`PhaseVocoder::set_smooth_ratio_updates`]).
    pub fn set_smooth_ratio_updates(&mut self, smooth: bool) {
        self.sub_bass_pv.set_smooth_ratio_updates(smooth);
        self.mid_pv.set_smooth_ratio_updates(smooth);
        self.high_pv.set_smooth_ratio_updates(smooth);
    }

    /// Sets envelope correction strength on all band vocoders.
    pub fn set_envelope_strength(&mut self, strength: f32) {
        self.sub_bass_pv.set_envelope_strength(strength);
        self.mid_pv.set_envelope_strength(strength);
        self.high_pv.set_envelope_strength(strength);
    }

    /// Enables/disables adaptive envelope-order selection on all band vocoders.
    pub fn set_adaptive_envelope_order(&mut self, enabled: bool) {
        self.sub_bass_pv.set_adaptive_envelope_order(enabled);
        self.mid_pv.set_adaptive_envelope_order(enabled);
        self.high_pv.set_adaptive_envelope_order(enabled);
    }

    /// Resets the phase state of all three phase vocoders.
    pub fn reset_phase_state(&mut self) {
        self.sub_bass_pv.reset_phase_state();
        self.mid_pv.reset_phase_state();
        self.high_pv.reset_phase_state();
        self.splitter.reset();
    }

    /// Resets phase state for specific frequency bands across all PVs.
    pub fn reset_phase_state_bands(&mut self, reset_mask: [bool; 4], sample_rate: u32) {
        self.sub_bass_pv
            .reset_phase_state_bands(reset_mask, sample_rate);
        self.mid_pv.reset_phase_state_bands(reset_mask, sample_rate);
        self.high_pv
            .reset_phase_state_bands(reset_mask, sample_rate);
    }

    /// Returns the sub-bass FFT size.
    #[inline]
    pub fn sub_bass_fft_size(&self) -> usize {
        self.sub_bass_fft_size
    }

    /// Returns the mid-band FFT size.
    #[inline]
    pub fn mid_fft_size(&self) -> usize {
        self.mid_fft_size
    }

    /// Returns the high-band FFT size.
    #[inline]
    pub fn high_fft_size(&self) -> usize {
        self.high_fft_size
    }

    /// Stretches a mono audio signal using multi-resolution processing.
    ///
    /// Splits the input into three frequency bands, stretches each with
    /// an optimally-sized phase vocoder, and sums the results.
    ///
    /// Bands with input shorter than their FFT size fall back to linear
    /// resampling to avoid errors.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, StretchError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        let len = input.len();

        // Resize band buffers if needed (grow only, never shrink in hot path)
        if self.sub_bass_buf.len() < len {
            self.sub_bass_buf.resize(len, 0.0);
            self.mid_buf.resize(len, 0.0);
            self.high_buf.resize(len, 0.0);
        }

        // Split input into three bands
        self.splitter.process(
            input,
            &mut self.sub_bass_buf[..len],
            &mut self.mid_buf[..len],
            &mut self.high_buf[..len],
        );

        let out_len_fallback = (len as f64 * self.stretch_ratio).round().max(1.0) as usize;

        // Process each band with its own PV (or fall back to linear resample)
        let sub_bass_out = if len >= self.sub_bass_fft_size {
            self.sub_bass_pv.process(&self.sub_bass_buf[..len])?
        } else {
            crate::core::resample::resample_linear(&self.sub_bass_buf[..len], out_len_fallback)
        };

        let mid_out = if len >= self.mid_fft_size {
            self.mid_pv.process(&self.mid_buf[..len])?
        } else {
            crate::core::resample::resample_linear(&self.mid_buf[..len], out_len_fallback)
        };

        let high_out = if len >= self.high_fft_size {
            self.high_pv.process(&self.high_buf[..len])?
        } else {
            crate::core::resample::resample_linear(&self.high_buf[..len], out_len_fallback)
        };

        // Sum the three bands, zero-padding shorter outputs.
        // Zero-padding preserves phase coherence between bands — resampling
        // to a common length would shift phases and cause destructive
        // interference. The shorter bands are naturally near-silent at their
        // tails due to PV edge effects, so zero-padding is safe.
        let max_len = sub_bass_out.len().max(mid_out.len()).max(high_out.len());

        let mut output = vec![0.0f32; max_len];
        for (i, s) in sub_bass_out.iter().enumerate() {
            output[i] += s;
        }
        for (i, s) in mid_out.iter().enumerate() {
            output[i] += s;
        }
        for (i, s) in high_out.iter().enumerate() {
            output[i] += s;
        }

        Ok(output)
    }

    /// Per-band hop of the streaming consumption driver (the sub-bass PV).
    #[inline]
    fn sub_bass_hop(&self) -> usize {
        self.sub_bass_fft_size / 4
    }

    /// Frames a streaming call over a `window_frames`-long window consumes.
    ///
    /// Mirrors the phase-vocoder streaming contract: the caller passes its
    /// full retained rolling window to [`Self::process_streaming_into`] and
    /// discards exactly this many frames from the front afterwards. The
    /// cadence is keyed to the sub-bass PV (largest FFT); its hop is a
    /// multiple of the mid and high hops, so one consumption figure keeps
    /// all three band vocoders frame-aligned across calls.
    #[inline]
    pub fn streaming_frames_consumed(&self, window_frames: usize) -> usize {
        let fft = self.sub_bass_fft_size;
        let hop = self.sub_bass_hop();
        if hop == 0 || window_frames < fft {
            return 0;
        }
        ((window_frames - fft) / hop + 1) * hop
    }

    /// Pre-allocates all streaming buffers for a maximum window size.
    ///
    /// Call once at build time with the caller's rolling-window capacity so
    /// [`Self::process_streaming_into`] performs no allocations afterwards.
    /// `max_ratio` bounds the output-side buffers the same way the caller's
    /// output buffers are bounded.
    pub fn reserve_streaming_capacity(&mut self, max_window_frames: usize, max_ratio: f64) {
        let out_bound = Self::stream_out_bound(max_window_frames, max_ratio)
            .saturating_add(self.sub_bass_fft_size.saturating_mul(2));
        for buf in &mut self.stream_band_bufs {
            reserve_to(buf, max_window_frames);
        }
        for buf in &mut self.stream_pending {
            reserve_to(buf, out_bound);
        }
        reserve_to(&mut self.stream_band_out, out_bound);
    }

    /// Output-side capacity bound for a render consuming `frames` frames.
    #[inline]
    fn stream_out_bound(frames: usize, ratio: f64) -> usize {
        let ratio_mult = (ratio.max(1.0).ceil() as usize).saturating_add(1);
        frames.saturating_mul(ratio_mult)
    }

    /// Streaming multi-resolution pass writing directly into `output`.
    ///
    /// Contract parity with [`PhaseVocoder::process_streaming_into`]:
    ///
    /// - `input` is the caller's full retained rolling window (analysis
    ///   overlap included). After the call the caller discards
    ///   [`Self::streaming_frames_consumed`]`(input.len())` frames from the
    ///   front and appends new audio before the next call.
    /// - `output` is overwritten with the samples that are final for this
    ///   call; an insufficient `output` capacity is a
    ///   [`StretchError::BufferOverflow`], never a reallocation.
    /// - Windows shorter than the sub-bass FFT emit nothing (empty output,
    ///   `Ok`), matching the PV's short-input behavior.
    ///
    /// Internally the Linkwitz-Riley crossover runs incrementally (stateful
    /// IIR biquads persist across calls; only the new tail of the window is
    /// split), each band feeds its own phase vocoder in streaming mode, and
    /// band outputs are queued and summed once every band has resolved the
    /// same synthesis position. Allocation-free after
    /// [`Self::reserve_streaming_capacity`] (or after first-call warmup).
    pub fn process_streaming_into(
        &mut self,
        input: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<(), StretchError> {
        output.clear();

        // Split only the samples this call appended to the caller's window;
        // the retained prefix was split (and buffered per band) previously.
        if input.len() < self.stream_retained_frames {
            return Err(StretchError::InvalidState(
                "multi-resolution streaming window shrank without reset",
            ));
        }
        let new_samples = &input[self.stream_retained_frames..];
        if !new_samples.is_empty() {
            for band_buf in &mut self.stream_band_bufs {
                let old_len = band_buf.len();
                band_buf.resize(old_len + new_samples.len(), 0.0);
            }
            let [sub, mid, high] = &mut self.stream_band_bufs;
            let start = sub.len() - new_samples.len();
            self.splitter.process(
                new_samples,
                &mut sub[start..],
                &mut mid[start..],
                &mut high[start..],
            );
        }
        self.stream_retained_frames = input.len();

        let window = input.len();
        let consumed = self.streaming_frames_consumed(window);
        if consumed == 0 {
            return self.emit_aligned_pending(output);
        }

        // The band PVs error on insufficient output capacity rather than
        // reallocating, so keep the shared band scratch ahead of this call's
        // worst-case emission (grow-only; a no-op in the steady state).
        let out_bound = Self::stream_out_bound(consumed, self.stretch_ratio)
            .saturating_add(self.sub_bass_fft_size.saturating_mul(2));
        reserve_to(&mut self.stream_band_out, out_bound);

        // Feed each band PV a prefix of its band window sized so that every
        // band consumes exactly `consumed` frames this call: a PV over a
        // window of `fft + consumed - hop` frames analyzes `consumed / hop`
        // hop-aligned frames, keeping all three PVs on one shared timeline.
        let sub_len = window;
        let mid_len = self.mid_fft_size + consumed - self.mid_fft_size / 4;
        let high_len = self.high_fft_size + consumed - self.high_fft_size / 4;

        let mut flux = PerFrameFlux::default();
        let mut flux_seen = false;
        for (band, band_len) in [(0usize, sub_len), (1, mid_len), (2, high_len)] {
            let pv = match band {
                0 => &mut self.sub_bass_pv,
                1 => &mut self.mid_pv,
                _ => &mut self.high_pv,
            };
            let band_input = &self.stream_band_bufs[band][..band_len];
            self.stream_band_out.clear();
            pv.process_streaming_into(band_input, &mut self.stream_band_out)?;
            self.stream_pending[band].extend_from_slice(&self.stream_band_out);
            if let Some(f) = pv.last_frame_flux() {
                flux_seen = true;
                flux.sub_bass += f.sub_bass;
                flux.low += f.low;
                flux.mid += f.mid;
                flux.high += f.high;
                flux.transient_bin_count = flux
                    .transient_bin_count
                    .saturating_add(f.transient_bin_count);
                flux.total_bins_rising = flux.total_bins_rising.saturating_add(f.total_bins_rising);
            }
        }
        self.stream_last_flux = if flux_seen { Some(flux) } else { None };

        // Drop the consumed prefix from each band mirror.
        for band_buf in &mut self.stream_band_bufs {
            band_buf.copy_within(consumed.., 0);
            band_buf.truncate(band_buf.len() - consumed);
        }
        self.stream_retained_frames -= consumed;

        self.emit_aligned_pending(output)
    }

    /// Sums the aligned front of the per-band pending queues into `output`.
    fn emit_aligned_pending(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        let emit = self.stream_pending.iter().map(Vec::len).min().unwrap_or(0);
        if emit == 0 {
            return Ok(());
        }
        if output.capacity() < emit {
            return Err(StretchError::BufferOverflow {
                buffer: "multi_resolution_stream_output",
                requested: emit,
                available: output.capacity(),
            });
        }
        output.resize(emit, 0.0);
        let [sub, mid, high] = &self.stream_pending;
        for i in 0..emit {
            output[i] = sub[i] + mid[i] + high[i];
        }
        for pending in &mut self.stream_pending {
            pending.copy_within(emit.., 0);
            pending.truncate(pending.len() - emit);
        }
        Ok(())
    }

    /// Flushes each band vocoder's remaining overlap tail into `output`.
    ///
    /// End-of-stream counterpart of [`Self::process_streaming_into`]:
    /// remaining band tails are summed with zero-padding to the longest
    /// band (mirroring the offline `process` summation) and all streaming
    /// state is reset for a fresh stream.
    pub fn flush_streaming_into(&mut self, output: &mut Vec<f32>) -> Result<(), StretchError> {
        for (band, pv) in [
            (0usize, &mut self.sub_bass_pv),
            (1, &mut self.mid_pv),
            (2, &mut self.high_pv),
        ] {
            self.stream_band_out.clear();
            pv.flush_streaming_into(&mut self.stream_band_out)?;
            self.stream_pending[band].extend_from_slice(&self.stream_band_out);
        }

        let total = self.stream_pending.iter().map(Vec::len).max().unwrap_or(0);
        output.clear();
        if output.capacity() < total {
            return Err(StretchError::BufferOverflow {
                buffer: "multi_resolution_flush_output",
                requested: total,
                available: output.capacity(),
            });
        }
        output.resize(total, 0.0);
        for pending in &self.stream_pending {
            for (out, &s) in output.iter_mut().zip(pending.iter()) {
                *out += s;
            }
        }

        self.reset_streaming_state();
        Ok(())
    }

    /// Clears all streaming state — band mirrors, pending queues, crossover
    /// IIR state, and each band PV's per-stream state — without deallocating.
    pub fn reset_streaming_state(&mut self) {
        for buf in &mut self.stream_band_bufs {
            buf.clear();
        }
        for buf in &mut self.stream_pending {
            buf.clear();
        }
        self.stream_retained_frames = 0;
        self.stream_last_flux = None;
        self.splitter.reset();
        self.sub_bass_pv.reset_streaming_state();
        self.mid_pv.reset_streaming_state();
        self.high_pv.reset_streaming_state();
    }

    /// Combined per-frame spectral flux from the most recent streaming call.
    ///
    /// Field-wise sum of the three band vocoders' [`PerFrameFlux`] reports.
    /// Each band PV computes flux against absolute bin frequencies over its
    /// own band-limited spectrum, so the sum lands in the scheduler's
    /// existing `[sub_bass, low, mid, high]` layout without a new mapping:
    ///
    /// - filterbank sub-bass band (< 200 Hz) contributes to the `sub_bass`
    ///   (< 100 Hz) and `low` (100–500 Hz) fields,
    /// - filterbank mid band (200–4000 Hz) contributes to `low` and `mid`,
    /// - filterbank high band (> 4000 Hz) contributes to `high`.
    ///
    /// `transient_bin_count` / `total_bins_rising` are saturating sums over
    /// all three spectra; normalize against [`Self::total_spectral_bins`].
    pub fn last_frame_flux(&self) -> Option<PerFrameFlux> {
        self.stream_last_flux
    }

    /// Total spectral bin count across the three band vocoders.
    ///
    /// The denominator for bin-fraction heuristics over the combined
    /// [`Self::last_frame_flux`] report.
    pub fn total_spectral_bins(&self) -> usize {
        (self.sub_bass_fft_size / 2 + 1)
            + (self.mid_fft_size / 2 + 1)
            + (self.high_fft_size / 2 + 1)
    }

    /// Streaming buffering gate in samples (mono frames).
    ///
    /// The maximum over the band PVs' buffering gates — `fft * 3/2` per the
    /// stream-processor convention, dominated by the sub-bass FFT. The
    /// Linkwitz-Riley crossovers are IIR and add no buffering; their group
    /// delay near the crossover points is a few milliseconds of phase lag,
    /// not gated latency.
    pub fn latency_samples(&self) -> usize {
        self.sub_bass_fft_size * 3 / 2
    }
}

/// Grows `buf`'s capacity to at least `capacity` (never shrinks).
#[inline]
fn reserve_to(buf: &mut Vec<f32>, capacity: usize) {
    if buf.capacity() < capacity {
        buf.reserve(capacity - buf.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that multi-resolution stretcher produces output of approximately correct length.
    #[test]
    fn test_multi_res_output_length() {
        let sample_rate = 44100;
        let stretch_ratio = 1.5;
        let mut stretcher = MultiResolutionStretcher::new(4096, stretch_ratio, sample_rate, 120.0);

        // Generate 2 seconds of 440 Hz sine (long enough for all FFT sizes)
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Output should be approximately stretch_ratio * input_len
        let expected = (len as f64 * stretch_ratio) as usize;
        let tolerance = expected / 5; // 20% tolerance for PV edge effects
        assert!(
            output.len().abs_diff(expected) < tolerance,
            "Output length {} too far from expected {} (tolerance {})",
            output.len(),
            expected,
            tolerance
        );
    }

    /// Test that stretch ratio 1.0 with multi-resolution produces near-identity output.
    #[test]
    fn test_multi_res_identity_stretch() {
        let sample_rate = 44100;
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.0, sample_rate, 120.0);

        // Generate 2 seconds of 440 Hz sine
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Output length should be very close to input length for ratio 1.0
        let length_diff = output.len().abs_diff(input.len());
        assert!(
            length_diff < len / 10,
            "Identity stretch length diff too large: {length_diff} (input={}, output={})",
            input.len(),
            output.len()
        );

        // Check that the output has similar energy to the input
        let input_energy: f64 = input.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let output_energy: f64 = output
            .iter()
            .take(input.len())
            .map(|s| (*s as f64) * (*s as f64))
            .sum();

        let energy_ratio = output_energy / input_energy;
        assert!(
            (0.3..3.0).contains(&energy_ratio),
            "Energy ratio {energy_ratio:.3} too far from 1.0 for identity stretch"
        );
    }

    /// Test that a 100 Hz sine stretched 1.5x preserves frequency.
    #[test]
    fn test_multi_res_preserves_low_freq() {
        let sample_rate = 44100;
        let freq = 100.0f32;
        let stretch_ratio = 1.5;
        let mut stretcher = MultiResolutionStretcher::new(4096, stretch_ratio, sample_rate, 120.0);

        // 2 seconds of 100 Hz sine
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Measure dominant frequency via zero-crossing rate
        // Skip edges (start/end) where PV artifacts are worst
        let skip = sample_rate as usize / 2;
        let analysis_len = output.len().saturating_sub(skip * 2);
        if analysis_len < sample_rate as usize {
            // Not enough output to analyze
            return;
        }

        let analysis = &output[skip..skip + analysis_len];
        let zero_crossings = analysis
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        // Zero crossings per second = 2 * frequency
        let measured_freq =
            zero_crossings as f64 / 2.0 / (analysis_len as f64 / sample_rate as f64);

        // Frequency should be preserved (within 15% tolerance for PV processing)
        let freq_error = (measured_freq - freq as f64).abs() / freq as f64;
        assert!(
            freq_error < 0.15,
            "100 Hz sine frequency not preserved: measured {measured_freq:.1} Hz, error {:.1}%",
            freq_error * 100.0
        );
    }

    /// Test that a high-frequency sine stretched 1.5x preserves frequency.
    #[test]
    fn test_multi_res_preserves_high_freq() {
        let sample_rate = 44100;
        let freq = 8000.0f32;
        let stretch_ratio = 1.5;
        let mut stretcher = MultiResolutionStretcher::new(4096, stretch_ratio, sample_rate, 120.0);

        // 2 seconds of 8000 Hz sine
        let len = sample_rate as usize * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Measure dominant frequency via zero-crossing rate
        let skip = sample_rate as usize / 2;
        let analysis_len = output.len().saturating_sub(skip * 2);
        if analysis_len < sample_rate as usize {
            return;
        }

        let analysis = &output[skip..skip + analysis_len];
        let zero_crossings = analysis
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        let measured_freq =
            zero_crossings as f64 / 2.0 / (analysis_len as f64 / sample_rate as f64);

        let freq_error = (measured_freq - freq as f64).abs() / freq as f64;
        assert!(
            freq_error < 0.15,
            "8 kHz sine frequency not preserved: measured {measured_freq:.1} Hz, error {:.1}%",
            freq_error * 100.0
        );
    }

    /// Test that set_stretch_ratio updates all bands.
    #[test]
    fn test_multi_res_set_ratio() {
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.0, 44100, 120.0);
        stretcher.set_stretch_ratio(2.0);

        // Generate enough input
        let len = 44100 * 2;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = stretcher.process(&input).unwrap();

        // Should be approximately 2x the input length
        let expected = len * 2;
        let tolerance = expected / 5;
        assert!(
            output.len().abs_diff(expected) < tolerance,
            "After set_stretch_ratio(2.0): output {} vs expected {} (tol {})",
            output.len(),
            expected,
            tolerance
        );
    }

    /// Test that empty input produces empty output.
    #[test]
    fn test_multi_res_empty_input() {
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.5, 44100, 120.0);
        let output = stretcher.process(&[]).unwrap();
        assert!(output.is_empty());
    }

    /// Test that short input (below FFT sizes) still produces output via fallback.
    #[test]
    fn test_multi_res_short_input_fallback() {
        let mut stretcher = MultiResolutionStretcher::new(4096, 1.5, 44100, 120.0);

        // Input shorter than any FFT size -- all bands fall back to linear resample
        let input = vec![0.5f32; 100];
        let output = stretcher.process(&input).unwrap();
        assert!(
            !output.is_empty(),
            "Short input should still produce output via fallback"
        );
    }

    /// Test FFT size getters.
    #[test]
    fn test_multi_res_fft_sizes() {
        let stretcher = MultiResolutionStretcher::new(4096, 1.0, 44100, 120.0);
        assert_eq!(stretcher.sub_bass_fft_size(), 16384);
        assert_eq!(stretcher.mid_fft_size(), 4096);
        assert_eq!(stretcher.high_fft_size(), 1024);
    }

    /// Test that FFT sizes scale proportionally.
    #[test]
    fn test_multi_res_fft_size_scaling() {
        let stretcher = MultiResolutionStretcher::new(2048, 1.0, 44100, 120.0);
        assert_eq!(stretcher.sub_bass_fft_size(), 8192);
        assert_eq!(stretcher.mid_fft_size(), 2048);
        assert_eq!(stretcher.high_fft_size(), 512);
    }

    /// Test that minimum FFT size is enforced.
    #[test]
    fn test_multi_res_min_fft_size() {
        // With mid_fft_size = 512, high would be 128 but MIN_FFT_SIZE = 256
        let stretcher = MultiResolutionStretcher::new(512, 1.0, 44100, 120.0);
        assert_eq!(stretcher.high_fft_size(), 256);
    }

    /// EDM-flavored test signal: sub sine, mid lead, hats.
    fn streaming_test_signal(len: usize, sample_rate: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.4 * (2.0 * std::f32::consts::PI * 55.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 9000.0 * t).sin()
            })
            .collect()
    }

    /// Drives the streaming API with the rolling-window caller contract.
    fn stream_in_chunks(
        stretcher: &mut MultiResolutionStretcher,
        input: &[f32],
        chunk_size: usize,
    ) -> Vec<f32> {
        let mut window: Vec<f32> = Vec::new();
        let mut out = Vec::new();
        let mut call_out = Vec::with_capacity(input.len() * 4 + 262_144);
        for chunk in input.chunks(chunk_size) {
            window.extend_from_slice(chunk);
            let consumed = stretcher.streaming_frames_consumed(window.len());
            stretcher
                .process_streaming_into(&window, &mut call_out)
                .expect("streaming call");
            out.extend_from_slice(&call_out);
            window.drain(..consumed);
        }
        call_out.clear();
        stretcher
            .flush_streaming_into(&mut call_out)
            .expect("streaming flush");
        out.extend_from_slice(&call_out);
        out
    }

    /// Chunked and whole-buffer streaming invocations must match: the same
    /// hop-aligned analysis frames are processed either way, so the emitted
    /// stream may differ only by per-block normalization noise.
    #[test]
    fn test_multi_res_streaming_chunked_vs_whole_parity() {
        let sample_rate = 44100;
        let input = streaming_test_signal(sample_rate as usize * 3, sample_rate);

        let mut chunked_stretcher = MultiResolutionStretcher::new(1024, 1.25, sample_rate, 120.0);
        let chunked = stream_in_chunks(&mut chunked_stretcher, &input, 1024);

        let mut whole_stretcher = MultiResolutionStretcher::new(1024, 1.25, sample_rate, 120.0);
        let whole = stream_in_chunks(&mut whole_stretcher, &input, input.len());

        assert!(!chunked.is_empty(), "chunked run produced no output");
        let common = chunked.len().min(whole.len());
        assert!(
            chunked.len().abs_diff(whole.len()) <= common / 10,
            "chunked ({}) and whole ({}) lengths diverged",
            chunked.len(),
            whole.len()
        );
        // Skip the stream head: the PV normalizes each emitted block against
        // that block's own window-sum peak, so while the overlap-add window
        // sum is still ramping from zero the same samples land in blocks of
        // different spans (one giant block vs many small ones) and get
        // different normalization floors. Identical analysis frames make the
        // steady-state body match almost exactly.
        let skip = (whole_stretcher.latency_samples() as f64 * 1.25 * 2.0) as usize;
        assert!(common > skip * 2, "not enough output to compare");
        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        for i in skip..common {
            let d = (chunked[i] - whole[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = i;
            }
        }
        assert!(
            max_diff < 1e-3,
            "chunked vs whole max sample diff {max_diff} at {max_idx}/{common} exceeds tolerance"
        );
    }

    /// Streaming consumption cadence: multiples of the sub-bass hop, zero
    /// below the sub-bass FFT gate.
    #[test]
    fn test_multi_res_streaming_consumption_math() {
        let stretcher = MultiResolutionStretcher::new(1024, 1.0, 44100, 120.0);
        let sub_fft = stretcher.sub_bass_fft_size();
        let sub_hop = sub_fft / 4;
        assert_eq!(stretcher.streaming_frames_consumed(sub_fft - 1), 0);
        assert_eq!(stretcher.streaming_frames_consumed(sub_fft), sub_hop);
        assert_eq!(
            stretcher.streaming_frames_consumed(sub_fft + sub_hop * 2),
            sub_hop * 3
        );
    }

    /// The combined flux report appears once streaming processing runs.
    #[test]
    fn test_multi_res_streaming_flux_report() {
        let sample_rate = 44100;
        let mut stretcher = MultiResolutionStretcher::new(1024, 1.1, sample_rate, 120.0);
        assert!(stretcher.last_frame_flux().is_none());

        let input = streaming_test_signal(stretcher.sub_bass_fft_size() * 2, sample_rate);
        let mut out = Vec::with_capacity(input.len() * 4);
        stretcher
            .process_streaming_into(&input, &mut out)
            .expect("streaming call");
        let flux = stretcher
            .last_frame_flux()
            .expect("flux after a processed streaming call");
        assert!(
            flux.sub_bass >= 0.0 && flux.low >= 0.0 && flux.mid >= 0.0 && flux.high >= 0.0,
            "flux fields must be non-negative"
        );
        assert!(stretcher.total_spectral_bins() > 0);
    }

    /// The sub-bass FFT cap clamps the derived size but never below mid.
    #[test]
    fn test_multi_res_sub_bass_fft_cap() {
        let capped = MultiResolutionStretcher::with_sub_bass_fft_cap(2048, 1.0, 44100, 120.0, 8192);
        assert_eq!(capped.sub_bass_fft_size(), 8192);
        let clamped =
            MultiResolutionStretcher::with_sub_bass_fft_cap(4096, 1.0, 44100, 120.0, 8192);
        assert_eq!(clamped.sub_bass_fft_size(), 8192);
        let floor = MultiResolutionStretcher::with_sub_bass_fft_cap(4096, 1.0, 44100, 120.0, 256);
        assert_eq!(floor.sub_bass_fft_size(), 4096);
        assert_eq!(capped.latency_samples(), 8192 * 3 / 2);
    }
}
