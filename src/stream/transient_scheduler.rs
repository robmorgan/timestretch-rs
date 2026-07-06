//! Persistent transient-event scheduling for deterministic stream processing.

use crate::core::fft::COMPLEX_ZERO;
use crate::core::window::{generate_window, WindowType};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Sub-bass/low-band split (Hz) used for reset-mask routing.
const BAND_SUB_END_HZ: f64 = 100.0;
/// Low/mid split (Hz) used for reset-mask routing.
const BAND_LOW_END_HZ: f64 = 500.0;
/// Mid/high split (Hz) used for reset-mask routing.
const BAND_MID_END_HZ: f64 = 4000.0;
/// EMA coefficient for adaptive spectral-flux statistics.
const FLUX_EMA_ALPHA: f64 = 0.2;
/// Sigma multiplier for adaptive spectral-flux threshold.
const FLUX_THRESHOLD_SIGMA: f64 = 2.5;
/// Required jump versus previous frame flux to classify a transient.
const FLUX_SPIKE_RATIO: f64 = 1.6;
/// Absolute guard to suppress near-silence false triggers.
const FLUX_ABS_MIN: f64 = 1e-4;
/// Extra emphasis on high-band flux.
const FLUX_HIGH_WEIGHT: f64 = 1.25;
/// Extra trigger threshold per overlap window when modulation already holds
/// low bands phase-locked.
const FLUX_MODULATION_THRESHOLD_SCALE_STEP: f64 = 0.08;
/// Extra spike-ratio requirement per overlap window during modulation holds.
const FLUX_MODULATION_SPIKE_RATIO_STEP: f64 = 0.05;
/// Minimum upper-band share required before low-band-suppressed modulation
/// converts a detected event into a mid/high phase reset.
const FLUX_MODULATION_MIN_UPPER_SHARE: f64 = 0.55;
/// Minimum upper-band energy relative to the adaptive threshold before
/// modulation-hold events may reset only the upper bands.
const FLUX_MODULATION_MIN_UPPER_THRESHOLD_SHARE: f64 = 0.35;
/// Required upper-band dominance over held low bands before modulation-hold
/// mode turns a seam-side event into a fresh upper-band reset.
const FLUX_MODULATION_MIN_UPPER_DOMINANCE: f64 = 1.25;
/// Number of flux frames to observe before trigger checks.
const FLUX_WARMUP_FRAMES: usize = 3;
/// Maximum analysis frames scanned per scheduler pass.
const FLUX_MAX_SCAN_FRAMES: usize = 8;
/// Minimum cooldown frames after an event to avoid duplicate resets.
const FLUX_RESET_COOLDOWN_FRAMES: usize = 2;
/// Base overlap windows used by tests when verifying modulation cooldown scaling.
#[cfg(test)]
const MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS: usize = 2;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TransientSchedulerStats {
    pub(crate) events_detected_total: u64,
    pub(crate) reset_band_counts_total: [u64; 4],
}

/// Stateful spectral-flux transient scheduler.
///
/// The scheduler consumes stereo interleaved analysis snapshots (L/R),
/// computes a stereo-coherent transient score from per-channel magnitudes, and
/// emits a per-band phase-reset
/// mask (`[sub_bass, low, mid, high]`) when a transient event is detected.
pub(crate) struct TransientEventScheduler {
    fft_size: usize,
    hop_size: usize,
    sample_rate: u32,
    max_frames: usize,
    num_bins: usize,
    sub_end_bin: usize,
    low_end_bin: usize,
    mid_end_bin: usize,
    fft_forward: Arc<dyn rustfft::Fft<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    fft_buffer: Vec<Complex<f32>>,
    prev_magnitudes: Vec<f32>,
    window: Vec<f32>,
    left_buffer: Vec<f32>,
    right_buffer: Vec<f32>,
    left_magnitudes: Vec<f32>,
    mean_flux: f64,
    var_flux: f64,
    prev_flux: f64,
    warmup_frames: usize,
    cooldown_frames: usize,
    last_processed_frame_start: Option<usize>,
    stats: TransientSchedulerStats,
}

impl TransientEventScheduler {
    pub(crate) fn new(
        fft_size: usize,
        hop_size: usize,
        sample_rate: u32,
        max_frames: usize,
    ) -> Self {
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let num_bins = fft_size / 2 + 1;
        let bin_hz = sample_rate as f64 / fft_size as f64;
        let sub_end_bin =
            ((BAND_SUB_END_HZ / bin_hz).floor() as usize).min(num_bins.saturating_sub(1));
        let low_end_bin =
            ((BAND_LOW_END_HZ / bin_hz).floor() as usize).min(num_bins.saturating_sub(1));
        let mid_end_bin =
            ((BAND_MID_END_HZ / bin_hz).floor() as usize).min(num_bins.saturating_sub(1));

        Self {
            fft_size,
            hop_size,
            sample_rate,
            max_frames,
            num_bins,
            sub_end_bin,
            low_end_bin,
            mid_end_bin,
            fft_scratch: vec![COMPLEX_ZERO; fft_forward.get_inplace_scratch_len()],
            fft_forward,
            fft_buffer: vec![COMPLEX_ZERO; fft_size],
            prev_magnitudes: vec![0.0; num_bins],
            window: generate_window(WindowType::Hann, fft_size),
            left_buffer: Vec::with_capacity(max_frames),
            right_buffer: Vec::with_capacity(max_frames),
            left_magnitudes: vec![0.0; num_bins],
            mean_flux: 0.0,
            var_flux: 0.0,
            prev_flux: 0.0,
            warmup_frames: FLUX_WARMUP_FRAMES,
            cooldown_frames: 0,
            last_processed_frame_start: None,
            stats: TransientSchedulerStats::default(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.prev_magnitudes.fill(0.0);
        self.mean_flux = 0.0;
        self.var_flux = 0.0;
        self.prev_flux = 0.0;
        self.warmup_frames = FLUX_WARMUP_FRAMES;
        self.cooldown_frames = 0;
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.last_processed_frame_start = None;
        self.stats = TransientSchedulerStats::default();
    }

    #[inline]
    fn reset_cooldown_frames(&self, modulation_overlap_windows: usize) -> usize {
        if self.hop_size == 0 {
            return FLUX_RESET_COOLDOWN_FRAMES;
        }

        // Hold the scheduler through the full overlapping-window footprint of
        // a detected event so the same click is not reclassified as a fresh
        // transient on the next few callback snapshots.
        let overlap_frames = self.fft_size.div_ceil(self.hop_size).saturating_sub(1);
        let mut cooldown_frames = FLUX_RESET_COOLDOWN_FRAMES.max(overlap_frames);

        if modulation_overlap_windows > 0 {
            // Cross-unity/near-unity modulation already keeps low bands phase
            // locked. Extend the accepted-event hold so the same broadband hit
            // does not retrigger a second upper-band-only reset while the
            // modulation seam is still draining through overlapped callbacks.
            cooldown_frames = cooldown_frames
                .saturating_add(overlap_frames.saturating_mul(modulation_overlap_windows));
        }

        cooldown_frames
    }

    #[inline]
    fn rejected_modulation_hold_cooldown_frames(&self, modulation_overlap_windows: usize) -> usize {
        if self.hop_size == 0 {
            return 1;
        }

        // Rejected low-dominant events should still hold through the currently
        // overlapping callback footprint plus almost one more overlap window so
        // the same physical onset is not re-evaluated as a fresh upper-band
        // reset right as the overlap tail drains through adjacent callbacks.
        let overlap_frames = self
            .fft_size
            .div_ceil(self.hop_size)
            .saturating_sub(1)
            .max(1);
        overlap_frames
            .saturating_mul(modulation_overlap_windows.max(1))
            .saturating_add(overlap_frames.saturating_sub(1))
    }

    #[inline]
    fn trigger_requirements(
        &self,
        suppress_low_bands: bool,
        modulation_overlap_windows: usize,
    ) -> (f64, f64) {
        if !suppress_low_bands || modulation_overlap_windows == 0 {
            return (1.0, FLUX_SPIKE_RATIO);
        }

        let overlap_windows = modulation_overlap_windows as f64;
        let threshold_scale = 1.0 + FLUX_MODULATION_THRESHOLD_SCALE_STEP * overlap_windows;
        let spike_ratio =
            FLUX_SPIKE_RATIO * (1.0 + FLUX_MODULATION_SPIKE_RATIO_STEP * overlap_windows);

        (threshold_scale, spike_ratio)
    }

    #[inline]
    fn should_accept_upper_band_reset_during_modulation_hold(
        &self,
        sub_flux: f64,
        low_flux: f64,
        mid_flux: f64,
        high_flux: f64,
        threshold: f64,
    ) -> bool {
        let upper_flux = mid_flux + high_flux;
        let total_flux = sub_flux + low_flux + upper_flux;
        if total_flux <= 0.0 {
            return false;
        }

        let upper_share = upper_flux / total_flux;
        let upper_strength = upper_flux / threshold.max(1e-12);
        let held_low_flux = sub_flux + low_flux;
        let upper_dominance = upper_flux / held_low_flux.max(1e-12);
        upper_share >= FLUX_MODULATION_MIN_UPPER_SHARE
            && upper_strength >= FLUX_MODULATION_MIN_UPPER_THRESHOLD_SHARE
            && upper_dominance >= FLUX_MODULATION_MIN_UPPER_DOMINANCE
    }

    /// Detects a transient event from stereo interleaved input and returns a
    /// per-band reset mask when detected.
    ///
    /// The input may be larger than the configured scheduler capacity; in that
    /// case, only the most-recent `max_frames` region is analyzed.
    pub(crate) fn detect_stereo_reset_mask(
        &mut self,
        interleaved_stereo: &[f32],
        frame_origin: usize,
        suppress_low_bands: bool,
        modulation_overlap_windows: usize,
    ) -> Option<[bool; 4]> {
        if self.hop_size == 0 || interleaved_stereo.len() < self.fft_size.saturating_mul(2) {
            return None;
        }

        let mut frames = interleaved_stereo.len() / 2;
        if frames < self.fft_size.saturating_add(self.hop_size) {
            return None;
        }

        let mut start_sample = 0usize;
        let mut absolute_frame_origin = frame_origin;
        if frames > self.max_frames {
            let drop_frames = frames - self.max_frames;
            start_sample = drop_frames.saturating_mul(2);
            frames = self.max_frames;
            absolute_frame_origin = absolute_frame_origin.saturating_add(drop_frames);
        }
        let stereo = &interleaved_stereo[start_sample..start_sample + frames.saturating_mul(2)];

        if self.left_buffer.capacity() < frames || self.right_buffer.capacity() < frames {
            return None;
        }
        self.left_buffer.clear();
        self.right_buffer.clear();
        for frame in stereo.chunks_exact(2) {
            self.left_buffer.push(frame[0]);
            self.right_buffer.push(frame[1]);
        }

        self.scan_frames(
            frames,
            absolute_frame_origin,
            true,
            suppress_low_bands,
            modulation_overlap_windows,
        )
    }

    /// Detects a transient event from a mono input snapshot and returns a
    /// per-band reset mask when detected.
    ///
    /// Mirrors [`Self::detect_stereo_reset_mask`]; all flux statistics,
    /// cooldowns, and mask routing are shared, only the per-frame magnitude
    /// source differs (single channel instead of an L/R average).
    pub(crate) fn detect_mono_reset_mask(
        &mut self,
        mono: &[f32],
        frame_origin: usize,
        suppress_low_bands: bool,
        modulation_overlap_windows: usize,
    ) -> Option<[bool; 4]> {
        if self.hop_size == 0 || mono.len() < self.fft_size {
            return None;
        }

        let mut frames = mono.len();
        if frames < self.fft_size.saturating_add(self.hop_size) {
            return None;
        }

        let mut start_sample = 0usize;
        let mut absolute_frame_origin = frame_origin;
        if frames > self.max_frames {
            let drop_frames = frames - self.max_frames;
            start_sample = drop_frames;
            frames = self.max_frames;
            absolute_frame_origin = absolute_frame_origin.saturating_add(drop_frames);
        }

        if self.left_buffer.capacity() < frames {
            return None;
        }
        self.left_buffer.clear();
        self.left_buffer
            .extend_from_slice(&mono[start_sample..start_sample + frames]);

        self.scan_frames(
            frames,
            absolute_frame_origin,
            false,
            suppress_low_bands,
            modulation_overlap_windows,
        )
    }

    /// Shared per-frame flux scan over the prepared channel buffers.
    ///
    /// Reads `left_buffer` (and `right_buffer` when `stereo`) and runs the
    /// incremental spectral-flux transient detection, returning the combined
    /// reset mask for the first accepted event, if any.
    fn scan_frames(
        &mut self,
        frames: usize,
        absolute_frame_origin: usize,
        stereo: bool,
        suppress_low_bands: bool,
        modulation_overlap_windows: usize,
    ) -> Option<[bool; 4]> {
        let num_frames = (frames - self.fft_size) / self.hop_size + 1;
        if num_frames < 2 {
            return None;
        }

        // We only need recent analysis frames to schedule phase resets in the
        // deterministic stream path.
        let start_frame = num_frames.saturating_sub(FLUX_MAX_SCAN_FRAMES);
        let mut reset_mask = [false; 4];

        for frame_idx in start_frame..num_frames {
            let start = frame_idx * self.hop_size;
            let absolute_frame_start = absolute_frame_origin.saturating_add(start);
            if let Some(last_start) = self.last_processed_frame_start {
                // Require one full-hop of absolute forward progress before a
                // new scheduler pass can rescan the same transient region.
                // This prevents sub-hop-overlapped callbacks from burning the
                // cooldown on nearly identical FFT windows.
                if absolute_frame_start < last_start.saturating_add(self.hop_size) {
                    continue;
                }
            }

            let left_frame = &self.left_buffer[start..start + self.fft_size];

            for (dst, (&sample, &window)) in self
                .fft_buffer
                .iter_mut()
                .zip(left_frame.iter().zip(self.window.iter()))
            {
                *dst = Complex::new(sample * window, 0.0);
            }
            self.fft_forward
                .process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);

            if stereo {
                for bin in 1..self.num_bins {
                    self.left_magnitudes[bin] = self.fft_buffer[bin].norm();
                }

                let right_frame = &self.right_buffer[start..start + self.fft_size];
                for (dst, (&sample, &window)) in self
                    .fft_buffer
                    .iter_mut()
                    .zip(right_frame.iter().zip(self.window.iter()))
                {
                    *dst = Complex::new(sample * window, 0.0);
                }
                self.fft_forward
                    .process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);
            }

            let mut sub_flux = 0.0f64;
            let mut low_flux = 0.0f64;
            let mut mid_flux = 0.0f64;
            let mut high_flux = 0.0f64;

            for bin in 1..self.num_bins {
                let mag = if stereo {
                    // Average per-channel magnitudes. This avoids mid-channel
                    // cancellation for anti-phase/wide stereo transients.
                    (self.left_magnitudes[bin] + self.fft_buffer[bin].norm()) * 0.5
                } else {
                    self.fft_buffer[bin].norm()
                };
                let diff = (mag - self.prev_magnitudes[bin]).max(0.0) as f64;
                if bin <= self.sub_end_bin {
                    sub_flux += diff;
                } else if bin <= self.low_end_bin {
                    low_flux += diff;
                } else if bin <= self.mid_end_bin {
                    mid_flux += diff;
                } else {
                    high_flux += diff;
                }
                self.prev_magnitudes[bin] = mag;
            }

            let flux = sub_flux * 0.8 + low_flux + mid_flux + high_flux * FLUX_HIGH_WEIGHT;
            if self.warmup_frames > 0 {
                self.update_flux_stats(flux);
                self.prev_flux = flux;
                self.warmup_frames = self.warmup_frames.saturating_sub(1);
                continue;
            }

            let sigma = self.var_flux.max(0.0).sqrt();
            let (threshold_scale, spike_ratio) =
                self.trigger_requirements(suppress_low_bands, modulation_overlap_windows);
            let threshold = (self.mean_flux + FLUX_THRESHOLD_SIGMA * sigma) * threshold_scale;
            let is_transient = flux > threshold
                && flux > self.prev_flux.max(FLUX_ABS_MIN) * spike_ratio
                && flux > FLUX_ABS_MIN;

            let triggered_event = is_transient && self.cooldown_frames == 0;
            if triggered_event {
                let mut event_mask =
                    self.select_reset_mask(sub_flux, low_flux, mid_flux, high_flux, threshold);
                if suppress_low_bands {
                    if !self.should_accept_upper_band_reset_during_modulation_hold(
                        sub_flux, low_flux, mid_flux, high_flux, threshold,
                    ) {
                        self.cooldown_frames = self
                            .rejected_modulation_hold_cooldown_frames(modulation_overlap_windows);
                        self.update_flux_stats(flux);
                        self.prev_flux = flux;
                        self.last_processed_frame_start = Some(absolute_frame_start);
                        continue;
                    }
                    event_mask[0] = false;
                    event_mask[1] = false;
                }
                for i in 0..4 {
                    reset_mask[i] |= event_mask[i];
                    if event_mask[i] {
                        self.stats.reset_band_counts_total[i] =
                            self.stats.reset_band_counts_total[i].saturating_add(1);
                    }
                }
                self.stats.events_detected_total =
                    self.stats.events_detected_total.saturating_add(1);
                self.cooldown_frames = self.reset_cooldown_frames(modulation_overlap_windows);
            }

            self.update_flux_stats(flux);
            self.prev_flux = flux;
            if !triggered_event {
                self.cooldown_frames = self.cooldown_frames.saturating_sub(1);
            }
            self.last_processed_frame_start = Some(absolute_frame_start);
            if triggered_event {
                // The caller can apply only one reset mask per scheduler pass.
                // Stop after the first accepted event so later onsets remain
                // individually schedulable on the next overlapping callback
                // instead of being merged into one broader reset.
                break;
            }
        }

        if reset_mask.iter().any(|&v| v) {
            Some(reset_mask)
        } else {
            None
        }
    }

    #[inline]
    fn update_flux_stats(&mut self, flux: f64) {
        let delta = flux - self.mean_flux;
        self.mean_flux += FLUX_EMA_ALPHA * delta;
        self.var_flux += FLUX_EMA_ALPHA * (delta * delta - self.var_flux);
    }

    /// Builds a per-band phase-reset mask from detected band fluxes.
    ///
    /// Mask layout: `[sub_bass, low, mid, high]`.
    fn select_reset_mask(
        &self,
        sub_flux: f64,
        low_flux: f64,
        mid_flux: f64,
        high_flux: f64,
        threshold: f64,
    ) -> [bool; 4] {
        let peak = low_flux.max(mid_flux).max(high_flux).max(1e-12);
        let mut mask = [false; 4];

        // Always protect upper content on detected events for crisp attacks.
        // Mid+high resets are deterministic to avoid missing percussive edges
        // when one upper band under-reports due to windowed energy split.
        mask[2] = true;
        mask[3] = true;

        // Kick-assist low-band reset: engage only when the event carries
        // enough actual low-end transient energy, not just an upper-band hit
        // with a modest low shelf.
        let low_energy_spike = sub_flux + low_flux > threshold * 0.75;
        let low_dominant =
            low_flux > peak * 0.30 && low_flux > threshold * 0.12 && low_energy_spike;
        let low_broadband_support =
            low_flux > peak * 0.22 && (mid_flux > peak * 0.25 || high_flux > peak * 0.25);
        let low_balance_guard = low_flux > (mid_flux + high_flux) * 0.18;
        if low_dominant || (low_broadband_support && low_energy_spike && low_balance_guard) {
            mask[1] = true;
        }

        // Keep sub resets conservative to avoid destabilizing sustained bass.
        if sub_flux > low_flux * 0.8 && sub_flux + low_flux > threshold * 1.05 {
            mask[0] = true;
        }

        mask
    }

    #[allow(dead_code)]
    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub(crate) fn stats(&self) -> TransientSchedulerStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TransientEventScheduler, FLUX_MODULATION_MIN_UPPER_DOMINANCE,
        FLUX_MODULATION_MIN_UPPER_SHARE, FLUX_MODULATION_MIN_UPPER_THRESHOLD_SHARE,
        FLUX_MODULATION_SPIKE_RATIO_STEP, FLUX_MODULATION_THRESHOLD_SCALE_STEP,
        FLUX_RESET_COOLDOWN_FRAMES, FLUX_SPIKE_RATIO,
        MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS,
    };
    use std::f32::consts::PI;

    #[test]
    fn scheduler_detects_click_transient() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut stereo = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            // Place the click near the tail so the scheduler's recent-frame
            // scan window observes it.
            let click = if (3400..3420).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        assert!(mask.is_some(), "expected transient reset mask");
        let mask = mask.unwrap();
        assert!(
            mask[2] || mask[3],
            "expected at least mid/high reset for click transient, got {:?}",
            mask
        );
    }

    #[test]
    fn scheduler_detects_antiphase_click_transient() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut stereo = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let base_l = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let base_r = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let click = if (3400..3420).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base_l + click;
            stereo[i * 2 + 1] = base_r - click; // anti-phase transient
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        assert!(
            mask.is_some(),
            "expected reset mask for anti-phase transient content"
        );
    }

    #[test]
    fn scheduler_reset_clears_state() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let stereo = vec![0.0f32; frames * 2];

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let _ = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        scheduler.reset();
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        assert!(
            mask.is_none(),
            "silent input should not produce reset mask after reset"
        );
    }

    #[test]
    fn scheduler_skips_duplicate_frames_for_same_origin() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut stereo = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let click = if (3400..3420).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let first = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        let second = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        assert!(first.is_some(), "first pass should observe transient");
        assert!(
            second.is_none(),
            "second pass with same origin should not reprocess duplicate frames"
        );
    }

    #[test]
    fn scheduler_detects_mono_click_transient() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut mono = vec![0.0f32; frames];
        for (i, sample) in mono.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let click = if (3400..3420).contains(&i) { 2.0 } else { 0.0 };
            *sample = base + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let mask = scheduler.detect_mono_reset_mask(&mono, 0, false, 0);
        assert!(mask.is_some(), "expected mono transient reset mask");
        let mask = mask.unwrap();
        assert!(
            mask[2] || mask[3],
            "expected at least mid/high reset for mono click transient, got {:?}",
            mask
        );
    }

    #[test]
    fn scheduler_mono_skips_duplicate_frames_for_same_origin() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut mono = vec![0.0f32; frames];
        for (i, sample) in mono.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let click = if (3400..3420).contains(&i) { 2.0 } else { 0.0 };
            *sample = base + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let first = scheduler.detect_mono_reset_mask(&mono, 0, false, 0);
        let second = scheduler.detect_mono_reset_mask(&mono, 0, false, 0);
        assert!(first.is_some(), "first mono pass should observe transient");
        assert!(
            second.is_none(),
            "second mono pass with same origin should not reprocess duplicate frames"
        );
    }

    #[test]
    fn scheduler_mono_ignores_steady_tone() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mono: Vec<f32> = (0..frames)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / sr as f32).sin() * 0.3)
            .collect();

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let mask = scheduler.detect_mono_reset_mask(&mono, 0, false, 0);
        assert!(
            mask.is_none(),
            "steady mono tone should not trigger phase resets, got {:?}",
            mask
        );
    }

    #[test]
    fn scheduler_select_reset_mask_always_sets_mid_and_high() {
        let mut scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);
        scheduler.warmup_frames = 0;
        let mask = scheduler.select_reset_mask(0.0, 0.2, 0.1, 0.3, 1.0);
        assert!(mask[3], "high band should always reset on detected events");
        assert!(mask[2], "mid band should always reset on detected events");
    }

    #[test]
    fn scheduler_select_reset_mask_enables_low_for_broadband_percussion() {
        let mut scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);
        scheduler.warmup_frames = 0;
        let mask = scheduler.select_reset_mask(0.18, 0.32, 0.90, 1.00, 0.60);
        assert!(
            mask[1],
            "low band should reset when broadband hits include meaningful low energy"
        );
        assert!(
            !mask[0],
            "sub band should remain conservative for moderate low-end events"
        );
    }

    #[test]
    fn scheduler_select_reset_mask_skips_low_for_bright_transient_with_weak_low_shelf() {
        let mut scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);
        scheduler.warmup_frames = 0;
        let mask = scheduler.select_reset_mask(0.02, 0.31, 0.75, 0.95, 0.60);
        assert!(
            !mask[1],
            "low band should stay locked when upper-band transients only carry a weak low shelf"
        );
        assert!(mask[2], "mid band should still reset for bright transients");
        assert!(
            mask[3],
            "high band should still reset for bright transients"
        );
    }

    #[test]
    fn scheduler_extends_cooldown_when_modulation_holds_low_bands() {
        let fft = 1024usize;
        let hop = 256usize;
        let overlap_frames = fft.div_ceil(hop).saturating_sub(1);
        let scheduler = TransientEventScheduler::new(fft, hop, 44_100, 4096);

        let base = scheduler.reset_cooldown_frames(0);
        let modulation =
            scheduler.reset_cooldown_frames(MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS);

        assert_eq!(
            base,
            FLUX_RESET_COOLDOWN_FRAMES.max(overlap_frames),
            "base cooldown should continue to cover one overlap footprint"
        );
        assert_eq!(
            modulation,
            base + overlap_frames * MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS,
            "modulation low-band holds should add one extra overlap window of duplicate-reset protection"
        );
    }

    #[test]
    fn scheduler_cooldown_scales_with_requested_modulation_overlap_windows() {
        let fft = 1024usize;
        let hop = 256usize;
        let overlap_frames = fft.div_ceil(hop).saturating_sub(1);
        let scheduler = TransientEventScheduler::new(fft, hop, 44_100, 4096);

        let base = scheduler.reset_cooldown_frames(0);
        let strong_modulation =
            scheduler.reset_cooldown_frames(MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS + 2);

        assert_eq!(
            strong_modulation,
            base + overlap_frames * (MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS + 2),
            "larger modulation bursts should hold accepted resets through extra overlap windows"
        );
    }

    #[test]
    fn scheduler_rejected_modulation_hold_cooldown_nearly_matches_full_reset_hold() {
        let fft = 1024usize;
        let hop = 256usize;
        let overlap_frames = fft.div_ceil(hop).saturating_sub(1);
        let scheduler = TransientEventScheduler::new(fft, hop, 44_100, 4096);

        let rejected = scheduler.rejected_modulation_hold_cooldown_frames(
            MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS,
        );
        let accepted =
            scheduler.reset_cooldown_frames(MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS);

        assert_eq!(
            rejected,
            overlap_frames * MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS
                + overlap_frames.saturating_sub(1),
            "rejected modulation-hold events should hold through the overlap tail plus one more near-full window"
        );
        assert!(
            rejected < accepted,
            "rejected low-dominant events should still unblock slightly sooner than a real accepted reset"
        );
    }

    #[test]
    fn scheduler_trigger_requirements_stay_neutral_without_modulation_hold() {
        let scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);

        assert_eq!(
            scheduler.trigger_requirements(false, 0),
            (1.0, FLUX_SPIKE_RATIO)
        );
        assert_eq!(
            scheduler.trigger_requirements(true, 0),
            (1.0, FLUX_SPIKE_RATIO)
        );
    }

    #[test]
    fn scheduler_trigger_requirements_tighten_with_modulation_overlap_windows() {
        let scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);

        let (threshold_scale, spike_ratio) =
            scheduler.trigger_requirements(true, MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS);

        assert_eq!(
            threshold_scale,
            1.0
                + FLUX_MODULATION_THRESHOLD_SCALE_STEP
                    * MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS as f64,
            "low-band-suppressed modulation should require a proportionally stronger flux threshold"
        );
        assert_eq!(
            spike_ratio,
            FLUX_SPIKE_RATIO
                * (1.0
                    + FLUX_MODULATION_SPIKE_RATIO_STEP
                        * MODULATION_RESET_COOLDOWN_BASE_OVERLAP_WINDOWS as f64),
            "low-band-suppressed modulation should require a proportionally larger frame-to-frame spike"
        );
    }

    #[test]
    fn scheduler_modulation_hold_rejects_low_dominant_event_without_clear_upper_attack() {
        let scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);
        let threshold = 1.0;
        let upper_flux = 0.34;
        let total_flux = upper_flux / (FLUX_MODULATION_MIN_UPPER_SHARE - 0.01);
        let low_flux = total_flux - upper_flux;

        assert!(
            !scheduler.should_accept_upper_band_reset_during_modulation_hold(
                0.0, low_flux, upper_flux * 0.35, upper_flux * 0.65, threshold
            ),
            "modulation-hold mode should reject events whose energy stays mostly in the held low bands"
        );
    }

    #[test]
    fn scheduler_modulation_hold_accepts_clear_upper_band_attack() {
        let scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);
        let threshold = 1.0;
        let upper_flux = FLUX_MODULATION_MIN_UPPER_THRESHOLD_SHARE + 0.08;
        let low_flux = upper_flux * (1.0 / FLUX_MODULATION_MIN_UPPER_SHARE - 1.0) * 0.6;

        assert!(
            scheduler.should_accept_upper_band_reset_during_modulation_hold(
                0.0, low_flux, upper_flux * 0.45, upper_flux * 0.55, threshold
            ),
            "modulation-hold mode should still allow distinct upper-band attacks to reseed the vocoder"
        );
    }

    #[test]
    fn scheduler_modulation_hold_rejects_borderline_upper_share_when_low_bands_still_compete() {
        let scheduler = TransientEventScheduler::new(1024, 256, 44_100, 4096);
        let threshold = 1.0;
        let held_low_flux = 0.32;
        let upper_flux = held_low_flux * (FLUX_MODULATION_MIN_UPPER_DOMINANCE - 0.02);
        let total_flux = held_low_flux + upper_flux;

        assert!(
            upper_flux / total_flux > FLUX_MODULATION_MIN_UPPER_SHARE,
            "test fixture should still clear the existing upper-share gate"
        );
        assert!(
            upper_flux / threshold > FLUX_MODULATION_MIN_UPPER_THRESHOLD_SHARE,
            "test fixture should still clear the existing upper-strength gate"
        );
        assert!(
            !scheduler.should_accept_upper_band_reset_during_modulation_hold(
                0.0,
                held_low_flux,
                upper_flux * 0.45,
                upper_flux * 0.55,
                threshold,
            ),
            "modulation-hold mode should keep rejecting seam-side events until upper bands clearly dominate the held low bands"
        );
    }

    #[test]
    fn scheduler_stats_accumulate_and_reset() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut stereo = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let click = if (3400..3420).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let _ = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        let stats = scheduler.stats();
        assert!(
            stats.events_detected_total > 0,
            "expected at least one detected transient event"
        );
        assert!(
            stats.reset_band_counts_total[2] > 0 && stats.reset_band_counts_total[3] > 0,
            "expected upper-band reset counts to accumulate"
        );

        scheduler.reset();
        let reset_stats = scheduler.stats();
        assert_eq!(reset_stats.events_detected_total, 0);
        assert_eq!(reset_stats.reset_band_counts_total, [0, 0, 0, 0]);
    }

    #[test]
    fn scheduler_tail_transient_preserves_full_cooldown_window() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut stereo = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            // Align the click so only the final scanned analysis frame sees it.
            let click = if (3840..3860).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        assert!(mask.is_some(), "expected tail transient reset mask");
        assert_eq!(
            scheduler.cooldown_frames,
            FLUX_RESET_COOLDOWN_FRAMES.max(fft.div_ceil(hop).saturating_sub(1)),
            "detected events should keep the full configured cooldown for subsequent analysis frames"
        );
    }

    #[test]
    fn scheduler_sub_hop_overlap_does_not_advance_cursor_or_burn_cooldown() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let callback_frames = 4096usize;
        let half_hop = hop / 2;
        let total_frames = callback_frames + half_hop;
        let mut stereo = vec![0.0f32; total_frames * 2];
        for i in 0..total_frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            // Align the click so only the final scanned analysis frame in the
            // first callback sees it.
            let click = if (3840..3860).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, callback_frames);
        let first = scheduler.detect_stereo_reset_mask(&stereo[..callback_frames * 2], 0, false, 0);
        assert!(
            first.is_some(),
            "expected initial tail transient reset mask"
        );
        assert_eq!(
            scheduler.last_processed_frame_start,
            Some(3072),
            "tail click should advance the processed-frame cursor to the final schedulable frame"
        );
        let expected_cooldown = scheduler.reset_cooldown_frames(0);
        assert_eq!(
            scheduler.cooldown_frames, expected_cooldown,
            "detected event should arm the full cooldown"
        );

        let shifted_start = half_hop * 2;
        let shifted_end = shifted_start + callback_frames * 2;
        let second = scheduler.detect_stereo_reset_mask(
            &stereo[shifted_start..shifted_end],
            half_hop,
            false,
            0,
        );
        assert!(
            second.is_none(),
            "sub-hop-overlapped callback should not schedule a duplicate transient reset"
        );
        assert_eq!(
            scheduler.last_processed_frame_start,
            Some(3072),
            "sub-hop-overlapped callback should not advance the processed-frame cursor"
        );
        assert_eq!(
            scheduler.cooldown_frames, expected_cooldown,
            "sub-hop-overlapped callback should not consume cooldown on nearly identical windows"
        );
        assert_eq!(
            scheduler.stats().events_detected_total,
            1,
            "sub-hop-overlapped callback should not count as a fresh transient event"
        );
    }

    #[test]
    fn scheduler_tail_transient_does_not_retrigger_across_repeated_half_hop_callbacks() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let callback_frames = 4096usize;
        let half_hop = hop / 2;
        let total_frames = callback_frames + hop * 4;
        let mut stereo = vec![0.0f32; total_frames * 2];
        for i in 0..total_frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            // Keep a single late transient inside the overlapping callback
            // windows long enough to exercise repeated half-hop re-entry.
            let click = if (3840..3860).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, callback_frames);
        let mut triggered_origins = Vec::new();
        for origin in (0..=8).map(|step| step * half_hop) {
            let start = origin * 2;
            let end = start + callback_frames * 2;
            let mask = scheduler.detect_stereo_reset_mask(&stereo[start..end], origin, false, 0);
            if mask.is_some() {
                triggered_origins.push(origin);
            }
        }

        assert_eq!(
            triggered_origins,
            vec![0],
            "repeated half-hop callbacks should not retrigger one tail transient after the initial pass"
        );
        assert_eq!(
            scheduler.stats().events_detected_total,
            1,
            "one tail transient should count as a single reset event across repeated half-hop callbacks"
        );
    }

    #[test]
    fn scheduler_tail_transient_does_not_retrigger_across_mixed_sub_hop_callbacks() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let callback_frames = 4096usize;
        let quarter_hop = hop / 4;
        let total_frames = callback_frames + hop * 4;
        let mut stereo = vec![0.0f32; total_frames * 2];
        for i in 0..total_frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            // Keep one late transient inside the overlapping callback windows
            // while the callback origin advances by mixed sub-hop strides.
            let click = if (3840..3860).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + click;
            stereo[i * 2 + 1] = base * 0.9 + click;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, callback_frames);
        let expected_last_start = Some(3072usize);
        let expected_cooldown = scheduler.reset_cooldown_frames(0);
        let mut triggered_origins = Vec::new();

        for origin in [
            0usize,
            quarter_hop,
            hop / 2,
            hop - quarter_hop,
            hop,
            hop + quarter_hop,
            hop + hop / 2,
            hop * 2,
        ] {
            let start = origin * 2;
            let end = start + callback_frames * 2;
            let mask = scheduler.detect_stereo_reset_mask(&stereo[start..end], origin, false, 0);
            if mask.is_some() {
                triggered_origins.push(origin);
            }

            if origin > 0 && origin < hop {
                assert!(
                    mask.is_none(),
                    "mixed sub-hop callback at origin {origin} should not reschedule the same tail transient"
                );
                assert_eq!(
                    scheduler.last_processed_frame_start,
                    expected_last_start,
                    "mixed sub-hop callback at origin {origin} should not advance the processed-frame cursor"
                );
                assert_eq!(
                    scheduler.cooldown_frames, expected_cooldown,
                    "mixed sub-hop callback at origin {origin} should not burn the transient cooldown"
                );
            }
        }

        assert_eq!(
            triggered_origins,
            vec![0],
            "mixed sub-hop callbacks should not retrigger one tail transient after the initial pass"
        );
        assert_eq!(
            scheduler.stats().events_detected_total,
            1,
            "one tail transient should count as a single reset event across mixed sub-hop callbacks"
        );
    }

    #[test]
    fn scheduler_mixed_sub_hop_callbacks_schedule_next_distinct_transient_once() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let callback_frames = 4096usize;
        let quarter_hop = hop / 4;
        let total_frames = callback_frames + hop * 5;
        let mut stereo = vec![0.0f32; total_frames * 2];
        for i in 0..total_frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            // The first click is late enough to trigger near the end of the
            // initial callback without being visible in the prior analysis
            // frame. The second begins just after the last pre-cooldown
            // window so it should fire on the first post-cooldown mixed
            // callback that clears the full-hop gate.
            let click_a = if (3328..3348).contains(&i) { 2.0 } else { 0.0 };
            let click_b = if (4416..4496).contains(&i) { 4.0 } else { 0.0 };
            let sample = base + click_a + click_b;
            stereo[i * 2] = sample;
            stereo[i * 2 + 1] = sample * 0.9;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, callback_frames);
        let mut triggered_origins = Vec::new();
        for origin in [
            0usize,
            quarter_hop,
            hop / 2,
            hop - quarter_hop,
            hop,
            hop + quarter_hop,
            hop + hop / 2,
            hop + hop - quarter_hop,
            hop * 2,
            hop * 2 + quarter_hop,
            hop * 2 + hop / 2,
            hop * 2 + hop - quarter_hop,
            hop * 3,
            hop * 3 + quarter_hop,
            hop * 3 + hop / 2,
            hop * 3 + hop - quarter_hop,
        ] {
            let start = origin * 2;
            let end = start + callback_frames * 2;
            let mask = scheduler.detect_stereo_reset_mask(&stereo[start..end], origin, false, 0);
            if mask.is_some() {
                triggered_origins.push(origin);
            }
        }

        assert_eq!(
            triggered_origins,
            vec![0, hop * 2 + quarter_hop],
            "mixed sub-hop overlap should keep suppressing the first transient while still scheduling the next distinct transient exactly once"
        );
        assert_eq!(
            scheduler.stats().events_detected_total,
            2,
            "two distinct transients should produce exactly two reset events across mixed sub-hop callbacks"
        );
    }

    #[test]
    fn scheduler_broad_tail_transient_does_not_retrigger_across_overlapping_callbacks() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let callback_frames = 4096usize;
        let total_frames = callback_frames + hop * 4;
        let mut stereo = vec![0.0f32; total_frames * 2];
        for i in 0..total_frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let burst = if (3600..4000).contains(&i) { 2.0 } else { 0.0 };
            stereo[i * 2] = base + burst;
            stereo[i * 2 + 1] = base * 0.9 + burst;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, callback_frames);
        for origin in [0usize, hop, hop * 2, hop * 3] {
            let start = origin * 2;
            let end = start + callback_frames * 2;
            let _ = scheduler.detect_stereo_reset_mask(&stereo[start..end], origin, false, 0);
        }

        let stats = scheduler.stats();
        assert_eq!(
            stats.events_detected_total, 1,
            "one broad tail transient should not retrigger across overlapping callbacks"
        );
    }

    #[test]
    fn scheduler_pass_emits_only_first_schedulable_event() {
        let sr = 44_100u32;
        let fft = 1024usize;
        let hop = 256usize;
        let frames = 4096usize;
        let mut stereo = vec![0.0f32; frames * 2];
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let base = (2.0 * PI * 220.0 * t).sin() * 0.2;
            let click_a = if (2816..2836).contains(&i) { 2.0 } else { 0.0 };
            let click_b = if (3840..3860).contains(&i) { 2.0 } else { 0.0 };
            let sample = base + click_a + click_b;
            stereo[i * 2] = sample;
            stereo[i * 2 + 1] = sample * 0.9;
        }

        let mut scheduler = TransientEventScheduler::new(fft, hop, sr, frames);
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0, false, 0);
        assert!(mask.is_some(), "expected first-event reset mask");

        let stats = scheduler.stats();
        assert_eq!(
            stats.events_detected_total, 1,
            "one scheduler pass should emit only the earliest schedulable transient event"
        );
    }
}
