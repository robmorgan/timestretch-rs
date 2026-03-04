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
/// Number of flux frames to observe before trigger checks.
const FLUX_WARMUP_FRAMES: usize = 3;
/// Maximum analysis frames scanned per scheduler pass.
const FLUX_MAX_SCAN_FRAMES: usize = 8;
/// Cooldown frames after an event to avoid duplicate resets.
const FLUX_RESET_COOLDOWN_FRAMES: usize = 2;

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

    /// Detects a transient event from stereo interleaved input and returns a
    /// per-band reset mask when detected.
    ///
    /// The input may be larger than the configured scheduler capacity; in that
    /// case, only the most-recent `max_frames` region is analyzed.
    pub(crate) fn detect_stereo_reset_mask(
        &mut self,
        interleaved_stereo: &[f32],
        frame_origin: usize,
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
                if absolute_frame_start <= last_start {
                    continue;
                }
            }

            let left_frame = &self.left_buffer[start..start + self.fft_size];
            let right_frame = &self.right_buffer[start..start + self.fft_size];

            for (dst, (&sample, &window)) in self
                .fft_buffer
                .iter_mut()
                .zip(left_frame.iter().zip(self.window.iter()))
            {
                *dst = Complex::new(sample * window, 0.0);
            }
            self.fft_forward
                .process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);

            for bin in 1..self.num_bins {
                self.left_magnitudes[bin] = self.fft_buffer[bin].norm();
            }

            for (dst, (&sample, &window)) in self
                .fft_buffer
                .iter_mut()
                .zip(right_frame.iter().zip(self.window.iter()))
            {
                *dst = Complex::new(sample * window, 0.0);
            }
            self.fft_forward
                .process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);

            let mut sub_flux = 0.0f64;
            let mut low_flux = 0.0f64;
            let mut mid_flux = 0.0f64;
            let mut high_flux = 0.0f64;

            for bin in 1..self.num_bins {
                // Average per-channel magnitudes. This avoids mid-channel
                // cancellation for anti-phase/wide stereo transients.
                let right_mag = self.fft_buffer[bin].norm();
                let mag = (self.left_magnitudes[bin] + right_mag) * 0.5;
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
            let threshold = self.mean_flux + FLUX_THRESHOLD_SIGMA * sigma;
            let is_transient = flux > threshold
                && flux > self.prev_flux.max(FLUX_ABS_MIN) * FLUX_SPIKE_RATIO
                && flux > FLUX_ABS_MIN;

            if is_transient && self.cooldown_frames == 0 {
                let event_mask =
                    self.select_reset_mask(sub_flux, low_flux, mid_flux, high_flux, threshold);
                for i in 0..4 {
                    reset_mask[i] |= event_mask[i];
                    if event_mask[i] {
                        self.stats.reset_band_counts_total[i] =
                            self.stats.reset_band_counts_total[i].saturating_add(1);
                    }
                }
                self.stats.events_detected_total =
                    self.stats.events_detected_total.saturating_add(1);
                self.cooldown_frames = FLUX_RESET_COOLDOWN_FRAMES;
            }

            self.update_flux_stats(flux);
            self.prev_flux = flux;
            self.cooldown_frames = self.cooldown_frames.saturating_sub(1);
            self.last_processed_frame_start = Some(absolute_frame_start);
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

        // Kick-assist low-band reset: engage for either low-dominant hits or
        // broadband percussive events with meaningful low contribution.
        let low_dominant = low_flux > peak * 0.30 && low_flux > threshold * 0.12;
        let low_broadband_support =
            low_flux > peak * 0.22 && (mid_flux > peak * 0.25 || high_flux > peak * 0.25);
        let low_energy_spike = sub_flux + low_flux > threshold * 0.75;
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
    use super::TransientEventScheduler;
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
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0);
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
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0);
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
        let _ = scheduler.detect_stereo_reset_mask(&stereo, 0);
        scheduler.reset();
        let mask = scheduler.detect_stereo_reset_mask(&stereo, 0);
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
        let first = scheduler.detect_stereo_reset_mask(&stereo, 0);
        let second = scheduler.detect_stereo_reset_mask(&stereo, 0);
        assert!(first.is_some(), "first pass should observe transient");
        assert!(
            second.is_none(),
            "second pass with same origin should not reprocess duplicate frames"
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
        let _ = scheduler.detect_stereo_reset_mask(&stereo, 0);
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
}
