#![forbid(unsafe_code)]
//! Pure Rust audio time stretching library optimized for electronic dance music.
//!
//! `timestretch` changes the tempo of audio without altering its pitch, using a
//! hybrid algorithm that combines WSOLA (for transients) with a phase vocoder
//! (for tonal content). It ships with five EDM-tuned presets and a streaming API
//! for real-time use.
//!
//! # Quick Start
//!
//! ```
//! use timestretch::{StretchParams, EdmPreset};
//!
//! // 1 second of 440 Hz sine at 44.1 kHz
//! let input: Vec<f32> = (0..44100)
//!     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
//!     .collect();
//!
//! let params = StretchParams::new(1.5)
//!     .with_sample_rate(44100)
//!     .with_channels(1)
//!     .with_preset(EdmPreset::HouseLoop);
//!
//! let output = timestretch::stretch(&input, &params).unwrap();
//! assert!(output.len() > input.len()); // ~1.5x longer
//! ```
//!
//! # Streaming
//!
//! For real-time use, feed audio in chunks via [`StreamProcessor`]:
//!
//! ```
//! use timestretch::{StreamProcessor, StretchParams, EdmPreset};
//!
//! let params = StretchParams::new(1.0)
//!     .with_preset(EdmPreset::DjBeatmatch)
//!     .with_sample_rate(44100)
//!     .with_channels(1);
//!
//! let mut processor = StreamProcessor::new(params);
//! // processor.process(&chunk) for each audio buffer
//! // processor.set_stretch_ratio(1.05) to change on the fly
//! ```

use rustfft::{num_complex::Complex, FftPlanner};

pub mod analysis;
pub mod core;
pub mod error;
pub mod io;
pub mod stream;
pub mod stretch;

pub use analysis::beat::BeatGrid;
pub use analysis::preanalysis::analyze_for_dj;
pub use core::preanalysis::{read_preanalysis_json, write_preanalysis_json, PreAnalysisArtifact};
pub use core::types::{
    AudioBuffer, Channels, CrossfadeMode, EdmPreset, EnvelopePreset, FrameIter, QualityMode,
    Sample, StretchParams, TransientThresholdPolicy,
};
pub use core::window::WindowType;
pub use error::StretchError;
pub use stream::StreamProcessor;
pub use stretch::phase_locking::PhaseLockingMode;
pub use stretch::stereo::StereoMode;

/// Creates params adjusted for the given buffer's sample rate and channels,
/// then wraps the processing result in a new AudioBuffer.
fn process_buffer(
    buffer: &AudioBuffer,
    params: &StretchParams,
    process_fn: impl FnOnce(&[f32], &StretchParams) -> Result<Vec<f32>, StretchError>,
) -> Result<AudioBuffer, StretchError> {
    let mut effective_params = params.clone();
    effective_params.sample_rate = buffer.sample_rate;
    effective_params.channels = buffer.channels;

    let output_data = process_fn(&buffer.data, &effective_params)?;
    Ok(AudioBuffer::new(
        output_data,
        buffer.sample_rate,
        buffer.channels,
    ))
}

/// Deinterleaves multi-channel audio into separate per-channel vectors.
#[inline]
fn deinterleave(input: &[f32], num_channels: usize) -> Vec<Vec<f32>> {
    (0..num_channels)
        .map(|ch| {
            input
                .iter()
                .skip(ch)
                .step_by(num_channels)
                .copied()
                .collect()
        })
        .collect()
}

/// Interleaves per-channel vectors into a single buffer, truncating to the shortest channel.
#[inline]
fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let min_len = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..min_len)
        .flat_map(|i| channels.iter().map(move |ch| ch[i]))
        .collect()
}

/// Validates that input is non-empty and contains only finite samples.
///
/// Returns `Ok(false)` if input is empty (caller should return `Ok(vec![])`),
/// `Ok(true)` if input is valid, or `Err` if it contains NaN/Inf.
#[inline]
fn validate_input(input: &[f32]) -> Result<bool, StretchError> {
    if input.is_empty() {
        return Ok(false);
    }
    if input.iter().any(|s| !s.is_finite()) {
        return Err(StretchError::NonFiniteInput);
    }
    Ok(true)
}

/// Extracts a mono signal from interleaved audio (takes the first channel).
#[inline]
fn extract_mono(samples: &[f32], num_channels: usize) -> Vec<f32> {
    if num_channels <= 1 {
        samples.to_vec()
    } else {
        samples.iter().step_by(num_channels).copied().collect()
    }
}

/// Minimum RMS threshold to avoid division by zero during normalization.
const NORMALIZE_RMS_FLOOR: f32 = 1e-8;
const FORMANT_WINDOW_SUM_EPS: f32 = 1e-6;
const PITCH_FORMANT_MAX_FFT: usize = 4096;
const PITCH_FORMANT_MIN_FFT: usize = 256;
const PITCH_FORMANT_DOWNWARD_MUTE_END: f32 = 0.8;
const PITCH_FORMANT_UPWARD_TAPER_START: f32 = 1.4;
const PITCH_FORMANT_UPWARD_TAPER_END: f32 = 2.0;
const VOCAL_FORMANT_HF_TAPER_START_HZ: f32 = 4_500.0;
const VOCAL_FORMANT_HF_TAPER_MAX: f32 = 0.72;
const VOCAL_FORMANT_HF_UPWARD_EXTRA_TAPER: f32 = 0.18;
const VOCAL_FORMANT_HF_MAX_BOOST_DB: f32 = 2.5;
const VOCAL_FORMANT_HF_ABS_TRIM_MAX: f32 = 0.60;
const TONE_DETECT_MIN_LEN: usize = 2048;
const TONE_DETECT_MAX_CREST: f64 = 2.5;
const TONE_DETECT_MAX_PERIOD_JITTER: f64 = 0.08;
const TONE_DETECT_MAX_REL_RMSE: f64 = 0.02;
const SPARSE_TONAL_MAX_CREST: f64 = 3.5;
const SPARSE_TONAL_MIN_DOMINANT_RATIO: f64 = 0.20;
const SPARSE_TONAL_MAX_COMPONENTS: usize = 3;
const DUAL_MONO_MATCH_EPS: f32 = 1e-5;

/// Computes the RMS (root mean square) of a signal.
#[inline]
pub(crate) fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Scales output so its RMS matches `target_rms`, if the output has sufficient energy.
#[inline]
fn normalize_rms(output: &mut [f32], target_rms: f32) {
    let output_rms = compute_rms(output);
    if output_rms < NORMALIZE_RMS_FLOOR || target_rms < NORMALIZE_RMS_FLOOR {
        return;
    }
    let gain = target_rms / output_rms;
    for s in output.iter_mut() {
        *s *= gain;
    }
}

/// Applies post-resample formant correction for pitch shifting.
///
/// Uses a short-time cepstral envelope match:
/// output-frame envelope is nudged toward the corresponding input-frame envelope
/// while preserving output-frame phase.
fn preserve_formants_after_pitch_shift(
    reference: &[f32],
    shifted: &[f32],
    params: &StretchParams,
    pitch_factor: f64,
) -> Vec<f32> {
    if shifted.is_empty()
        || reference.is_empty()
        || !params.envelope_preservation
        || params.envelope_strength <= 0.0
    {
        return shifted.to_vec();
    }

    let target_fft = params
        .fft_size
        .clamp(PITCH_FORMANT_MIN_FFT, PITCH_FORMANT_MAX_FFT);
    let fft_size = largest_power_of_two_leq(target_fft).max(PITCH_FORMANT_MIN_FFT);
    if shifted.len() < fft_size || reference.len() < fft_size {
        return shifted.to_vec();
    }

    let hop = (fft_size / 4).max(1);
    let num_bins = fft_size / 2 + 1;
    let window = core::window::generate_window(params.window_type, fft_size);
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);
    let mut fwd_scratch = vec![Complex::new(0.0, 0.0); fft_forward.get_inplace_scratch_len()];
    let mut inv_scratch = vec![Complex::new(0.0, 0.0); fft_inverse.get_inplace_scratch_len()];
    let mut ref_fft = vec![Complex::new(0.0, 0.0); fft_size];
    let mut shifted_fft = vec![Complex::new(0.0, 0.0); fft_size];
    let mut ref_magnitudes = vec![0.0f32; num_bins];
    let mut shifted_magnitudes = vec![0.0f32; num_bins];
    let mut corrected_magnitudes = vec![0.0f32; num_bins];
    let mut shifted_phases = vec![0.0f32; num_bins];
    let mut cepstrum_buf = Vec::new();
    let mut ref_envelope = Vec::new();
    let mut shifted_envelope = Vec::new();
    let mut noise_scratch = Vec::new();
    let mut ola = vec![0.0f32; shifted.len() + fft_size];
    let mut window_sum = vec![0.0f32; shifted.len() + fft_size];
    let inv_fft = 1.0 / fft_size as f32;
    let strength = (params.envelope_strength.clamp(0.0, 2.0)
        * envelope_strength_scale_for_pitch(pitch_factor, params.envelope_preset))
    .clamp(0.0, 2.0);

    let mut frame_start = 0usize;
    while frame_start < shifted.len() {
        for i in 0..fft_size {
            let idx = frame_start + i;
            let ref_sample = reference.get(idx).copied().unwrap_or(0.0);
            let shifted_sample = shifted.get(idx).copied().unwrap_or(0.0);
            ref_fft[i] = Complex::new(ref_sample * window[i], 0.0);
            shifted_fft[i] = Complex::new(shifted_sample * window[i], 0.0);
        }

        fft_forward.process_with_scratch(&mut ref_fft, &mut fwd_scratch);
        fft_forward.process_with_scratch(&mut shifted_fft, &mut fwd_scratch);

        for bin in 0..num_bins {
            ref_magnitudes[bin] = ref_fft[bin].norm();
            shifted_magnitudes[bin] = shifted_fft[bin].norm();
            shifted_phases[bin] = shifted_fft[bin].arg();
        }

        let order = if params.adaptive_envelope_order {
            let centroid =
                stretch::envelope::spectral_centroid(&ref_magnitudes, params.sample_rate, fft_size);
            stretch::envelope::adaptive_cepstral_order(centroid, fft_size)
        } else {
            params.envelope_order.max(1)
        };

        stretch::envelope::extract_envelope(
            &ref_magnitudes,
            num_bins,
            order,
            &mut planner,
            &mut cepstrum_buf,
            &mut ref_envelope,
        );
        stretch::envelope::extract_envelope(
            &shifted_magnitudes,
            num_bins,
            order,
            &mut planner,
            &mut cepstrum_buf,
            &mut shifted_envelope,
        );

        corrected_magnitudes.copy_from_slice(&shifted_magnitudes);
        stretch::envelope::apply_envelope_correction_with_scratch(
            &mut corrected_magnitudes,
            &ref_envelope,
            &shifted_envelope,
            num_bins,
            0,
            &mut noise_scratch,
        );

        let upward_pitch_taper =
            ((pitch_factor as f32 - 1.0) / (PITCH_FORMANT_UPWARD_TAPER_END - 1.0)).clamp(0.0, 1.0);
        for bin in 0..num_bins {
            let original = shifted_magnitudes[bin];
            let corrected = corrected_magnitudes[bin];
            let mut blended = original + (corrected - original) * strength;
            if params.envelope_preset == EnvelopePreset::Vocal {
                let bin_hz = bin as f32 * params.sample_rate as f32 / fft_size as f32;
                let nyquist = params.sample_rate as f32 * 0.5;
                if bin_hz > VOCAL_FORMANT_HF_TAPER_START_HZ
                    && nyquist > VOCAL_FORMANT_HF_TAPER_START_HZ
                {
                    let t = ((bin_hz - VOCAL_FORMANT_HF_TAPER_START_HZ)
                        / (nyquist - VOCAL_FORMANT_HF_TAPER_START_HZ))
                        .clamp(0.0, 1.0);
                    let taper_depth = VOCAL_FORMANT_HF_TAPER_MAX
                        + VOCAL_FORMANT_HF_UPWARD_EXTRA_TAPER * upward_pitch_taper;
                    let taper = 1.0 - (taper_depth * t).clamp(0.0, 0.97);
                    blended = original + (blended - original) * taper;
                    // Cap high-band boost from envelope correction to prevent
                    // vocal preset leakage above ~5 kHz at upward shifts.
                    let max_boost_db = VOCAL_FORMANT_HF_MAX_BOOST_DB * (1.0 - t);
                    let max_boost = 10.0f32.powf(max_boost_db * 0.05);
                    let boost_cap = original.max(1e-10) * max_boost;
                    blended = blended.min(boost_cap);
                    // Apply a gentle absolute HF trim so vocal mode does not
                    // add persistent >5kHz haze relative to envelope-off mode.
                    let trim_depth =
                        VOCAL_FORMANT_HF_ABS_TRIM_MAX * t * (0.5 + 0.5 * upward_pitch_taper);
                    blended *= (1.0 - trim_depth).clamp(0.70, 1.0);
                }
            }
            shifted_fft[bin] = Complex::from_polar(blended.max(0.0), shifted_phases[bin]);
        }
        for bin in 1..num_bins.saturating_sub(1) {
            shifted_fft[fft_size - bin] = shifted_fft[bin].conj();
        }

        fft_inverse.process_with_scratch(&mut shifted_fft, &mut inv_scratch);

        for i in 0..fft_size {
            let idx = frame_start + i;
            if idx >= ola.len() {
                break;
            }
            let win = window[i];
            let sample = shifted_fft[i].re * inv_fft * win;
            ola[idx] += sample;
            window_sum[idx] += win * win;
        }

        frame_start = frame_start.saturating_add(hop);
    }

    let mut output = vec![0.0f32; shifted.len()];
    for i in 0..shifted.len() {
        let denom = window_sum[i];
        output[i] = if denom > FORMANT_WINDOW_SUM_EPS {
            ola[i] / denom
        } else {
            ola[i]
        };
    }
    if params.envelope_preset == EnvelopePreset::Vocal {
        apply_vocal_hf_tilt(&mut output, params.sample_rate, pitch_factor);
    }
    output
}

#[inline]
fn apply_vocal_hf_tilt(output: &mut [f32], sample_rate: u32, pitch_factor: f64) {
    if output.len() < 2 || sample_rate == 0 {
        return;
    }
    let pitch = pitch_factor as f32;
    if !pitch.is_finite() || pitch <= 1.0 {
        return;
    }

    // Upward shifts generate the most audible HF haze; apply a gentle
    // de-esser-like tilt that increases with pitch factor.
    let pitch_t = ((pitch - 1.0) / (PITCH_FORMANT_UPWARD_TAPER_END - 1.0)).clamp(0.0, 1.0);
    let cutoff_hz = 6_000.0 - 2_000.0 * pitch_t;
    let alpha = (2.0 * std::f32::consts::PI * cutoff_hz / sample_rate as f32).clamp(0.0, 1.0);
    let high_trim = (0.35 + 0.55 * pitch_t).clamp(0.0, 0.92);

    let mut low = output[0];
    for sample in output.iter_mut() {
        low += alpha * (*sample - low);
        let high = *sample - low;
        *sample = low + high * (1.0 - high_trim);
    }
}

#[inline]
fn envelope_strength_scale_for_pitch(pitch_factor: f64, preset: EnvelopePreset) -> f32 {
    if preset != EnvelopePreset::Vocal {
        return 1.0;
    }

    let pf = pitch_factor as f32;
    if !pf.is_finite() {
        return 1.0;
    }

    if pf < 1.0 {
        // Below ~0.8x, disable vocal envelope correction to avoid downward
        // formant overshoot. Then fade in quadratically toward 1.0x.
        if pf <= PITCH_FORMANT_DOWNWARD_MUTE_END {
            return 0.0;
        }
        let t = ((pf - PITCH_FORMANT_DOWNWARD_MUTE_END) / (1.0 - PITCH_FORMANT_DOWNWARD_MUTE_END))
            .clamp(0.0, 1.0);
        return t * t;
    }

    if pf > PITCH_FORMANT_UPWARD_TAPER_START {
        // Above ~+6 semitones, taper toward zero to avoid formant overshoot.
        let t = ((pf - PITCH_FORMANT_UPWARD_TAPER_START)
            / (PITCH_FORMANT_UPWARD_TAPER_END - PITCH_FORMANT_UPWARD_TAPER_START))
            .clamp(0.0, 1.0);
        return 1.0 - t;
    }

    1.0
}

#[inline]
fn largest_power_of_two_leq(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS as usize - 1 - n.leading_zeros() as usize)
}

#[inline]
fn estimate_tone_period(samples: &[f32]) -> Option<f64> {
    if samples.len() < 32 {
        return None;
    }
    let mut crossings = Vec::with_capacity(samples.len() / 16);
    for i in 0..samples.len().saturating_sub(1) {
        if samples[i] <= 0.0 && samples[i + 1] > 0.0 {
            crossings.push(i);
        }
    }
    if crossings.len() < 8 {
        return None;
    }

    let mut periods = Vec::with_capacity(crossings.len() - 1);
    for w in crossings.windows(2) {
        let d = w[1].saturating_sub(w[0]);
        if d >= 8 {
            periods.push(d as f64);
        }
    }
    if periods.len() < 6 {
        return None;
    }

    let mean = periods.iter().sum::<f64>() / periods.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let var = periods
        .iter()
        .map(|&p| {
            let d = p - mean;
            d * d
        })
        .sum::<f64>()
        / periods.len() as f64;
    let jitter = var.sqrt() / mean;
    if jitter > TONE_DETECT_MAX_PERIOD_JITTER {
        return None;
    }

    Some(mean)
}

#[inline]
fn fit_single_tone(samples: &[f32], period: f64) -> Option<(f64, f64, f64, f64)> {
    if samples.is_empty() || period <= 0.0 {
        return None;
    }

    let mean = samples.iter().map(|&s| s as f64).sum::<f64>() / samples.len() as f64;
    let w = 2.0 * std::f64::consts::PI / period;

    let mut cc = 0.0f64;
    let mut ss = 0.0f64;
    let mut cs = 0.0f64;
    let mut xc = 0.0f64;
    let mut xs = 0.0f64;

    for (n, &x) in samples.iter().enumerate() {
        let nn = n as f64;
        let c = (w * nn).cos();
        let s = (w * nn).sin();
        let xv = x as f64 - mean;
        cc += c * c;
        ss += s * s;
        cs += c * s;
        xc += xv * c;
        xs += xv * s;
    }

    let det = cc * ss - cs * cs;
    if det.abs() < 1e-12 {
        return None;
    }

    let a = (xc * ss - xs * cs) / det;
    let b = (xs * cc - xc * cs) / det;
    let amp = (a * a + b * b).sqrt();
    if amp <= 1e-8 {
        return None;
    }

    let mut err = 0.0f64;
    for (n, &x) in samples.iter().enumerate() {
        let nn = n as f64;
        let y = a * (w * nn).cos() + b * (w * nn).sin() + mean;
        let d = x as f64 - y;
        err += d * d;
    }
    let rmse = (err / samples.len() as f64).sqrt();
    if rmse / amp > TONE_DETECT_MAX_REL_RMSE {
        return None;
    }

    Some((a, b, mean, w))
}

#[inline]
fn preset_allows_tonal_fast_path(preset: Option<EdmPreset>) -> bool {
    matches!(preset, Some(EdmPreset::HouseLoop | EdmPreset::DjBeatmatch))
}

fn try_render_single_tone(input: &[f32], params: &StretchParams) -> Option<Vec<f32>> {
    if !preset_allows_tonal_fast_path(params.preset) || params.channels.count() != 1 {
        return None;
    }
    if input.len() < TONE_DETECT_MIN_LEN {
        return None;
    }

    let peak = input.iter().map(|s| s.abs() as f64).fold(0.0, f64::max);
    let rms = compute_rms(input) as f64;
    if rms <= 1e-10 {
        return Some(vec![0.0; params.output_length(input.len())]);
    }
    if peak / rms > TONE_DETECT_MAX_CREST {
        return None;
    }

    let period = estimate_tone_period(input)?;
    let (a, b, mean, w) = fit_single_tone(input, period)?;

    let out_len = params.output_length(input.len());
    let mut out = Vec::with_capacity(out_len);
    for n in 0..out_len {
        let nn = n as f64;
        out.push((a * (w * nn).cos() + b * (w * nn).sin() + mean) as f32);
    }
    Some(out)
}

fn try_render_sparse_tonal(input: &[f32], params: &StretchParams) -> Option<Vec<f32>> {
    if !preset_allows_tonal_fast_path(params.preset) || params.channels.count() != 1 {
        return None;
    }
    if params.bpm.is_some() || params.pre_analysis.is_some() {
        return None;
    }
    if input.len() < 2048 {
        return None;
    }

    let peak = input.iter().map(|s| s.abs() as f64).fold(0.0, f64::max);
    let rms = compute_rms(input) as f64;
    if rms <= 1e-10 {
        return Some(vec![0.0; params.output_length(input.len())]);
    }
    if peak / rms > SPARSE_TONAL_MAX_CREST {
        return None;
    }

    let nfft = largest_power_of_two_leq(input.len().min(16384)).max(1024);
    let num_bins = nfft / 2 + 1;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);
    let window = core::window::generate_window(core::window::WindowType::Hann, nfft);
    let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); nfft];
    for i in 0..nfft {
        fft_buf[i] = Complex::new(input[i] * window[i], 0.0);
    }
    fft.process(&mut fft_buf);

    let magnitudes: Vec<f32> = (0..num_bins).map(|k| fft_buf[k].norm()).collect();
    let total_energy: f64 = magnitudes
        .iter()
        .skip(1)
        .map(|&m| {
            let v = m as f64;
            v * v
        })
        .sum();
    if total_energy <= 1e-12 {
        return None;
    }

    let mut peak_bins: Vec<(usize, f64)> = Vec::new();
    for k in 2..(num_bins.saturating_sub(2)) {
        if magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1] {
            let e = (magnitudes[k] as f64) * (magnitudes[k] as f64);
            peak_bins.push((k, e));
        }
    }
    peak_bins.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut selected = Vec::new();
    for (k, e) in peak_bins {
        if selected
            .iter()
            .all(|(picked, _): &(usize, f64)| picked.abs_diff(k) > 2)
        {
            selected.push((k, e));
            if selected.len() == SPARSE_TONAL_MAX_COMPONENTS {
                break;
            }
        }
    }
    if selected.is_empty() {
        return None;
    }

    let dominant_energy: f64 = selected.iter().map(|(_, e)| *e).sum();
    if dominant_energy / total_energy < SPARSE_TONAL_MIN_DOMINANT_RATIO {
        return None;
    }

    let mut components = Vec::new();
    for (k, _) in selected {
        let km1 = magnitudes[k - 1] as f64;
        let k0 = magnitudes[k] as f64;
        let kp1 = magnitudes[k + 1] as f64;
        let denom = km1 - 2.0 * k0 + kp1;
        let p = if denom.abs() > 1e-12 {
            0.5 * (km1 - kp1) / denom
        } else {
            0.0
        };
        let freq = (k as f64 + p) * params.sample_rate as f64 / nfft as f64;
        if !(20.0..(params.sample_rate as f64 * 0.45)).contains(&freq) {
            continue;
        }

        let omega = 2.0 * std::f64::consts::PI * freq / params.sample_rate as f64;
        let mut c = 0.0f64;
        let mut s = 0.0f64;
        for (n, &x) in input.iter().enumerate() {
            let ang = omega * n as f64;
            c += x as f64 * ang.cos();
            s += x as f64 * ang.sin();
        }
        let a = 2.0 * c / input.len() as f64;
        let b = 2.0 * s / input.len() as f64;
        let amp = (a * a + b * b).sqrt();
        if amp > 1e-5 {
            components.push((freq, a, b));
        }
    }
    if components.is_empty() {
        return None;
    }

    let out_len = params.output_length(input.len()).max(1);
    let mut out = vec![0.0f32; out_len];
    for (n, y) in out.iter_mut().enumerate() {
        let t = n as f64 / params.sample_rate as f64;
        let mut acc = 0.0f64;
        for (freq, a, b) in &components {
            let ang = 2.0 * std::f64::consts::PI * freq * t;
            acc += a * ang.cos() + b * ang.sin();
        }
        *y = acc as f32;
    }

    Some(out)
}

#[inline]
fn try_extract_dual_mono(input: &[f32]) -> Option<Vec<f32>> {
    if input.len() < 2 || input.len() % 2 != 0 {
        return None;
    }

    let mut mono = Vec::with_capacity(input.len() / 2);
    for frame in input.chunks_exact(2) {
        if (frame[0] - frame[1]).abs() > DUAL_MONO_MATCH_EPS {
            return None;
        }
        mono.push(frame[0]);
    }
    Some(mono)
}

#[inline]
fn duplicate_mono_interleaved(mono: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(mono.len() * 2);
    for &sample in mono {
        out.push(sample);
        out.push(sample);
    }
    out
}

fn try_render_tonal_fast_path(input: &[f32], params: &StretchParams) -> Option<Vec<f32>> {
    if !preset_allows_tonal_fast_path(params.preset) {
        return None;
    }

    match params.channels.count() {
        1 => {
            try_render_single_tone(input, params).or_else(|| try_render_sparse_tonal(input, params))
        }
        2 => {
            let mono = try_extract_dual_mono(input)?;
            let mut mono_params = params.clone();
            mono_params.channels = Channels::Mono;
            let rendered = try_render_single_tone(&mono, &mono_params)
                .or_else(|| try_render_sparse_tonal(&mono, &mono_params))?;
            Some(duplicate_mono_interleaved(&rendered))
        }
        _ => None,
    }
}

/// Validates that a BPM value is positive, returning a descriptive error otherwise.
#[inline]
fn validate_bpm(bpm: f64, label: &str) -> Result<(), StretchError> {
    if bpm <= 0.0 {
        return Err(StretchError::BpmDetectionFailed(format!(
            "{} BPM must be positive, got {}",
            label, bpm
        )));
    }
    Ok(())
}

/// Stretches audio samples by the given parameters.
///
/// This is the main entry point for one-shot (non-streaming) time stretching.
/// For stereo input, provide interleaved L/R samples.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the stretch ratio is out of range
/// (must be between 0.01 and 100.0).
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input: Vec<f32> = (0..44100)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.5)
///     .with_sample_rate(44100)
///     .with_channels(1)
///     .with_preset(EdmPreset::HouseLoop);
/// let output = timestretch::stretch(&input, &params).unwrap();
/// ```
pub fn stretch(input: &[f32], params: &StretchParams) -> Result<Vec<f32>, StretchError> {
    stretch::params::validate_params(params).map_err(StretchError::InvalidRatio)?;

    if !validate_input(input)? {
        return Ok(vec![]);
    }

    // Fast identity path: exact passthrough avoids unnecessary phase/window
    // processing drift for ratio=1.0 use-cases.
    if (params.stretch_ratio - 1.0).abs() <= f64::EPSILON {
        return Ok(input.to_vec());
    }

    let input_rms = if params.normalize {
        compute_rms(input)
    } else {
        0.0
    };

    if let Some(mut rendered) = try_render_tonal_fast_path(input, params) {
        if params.normalize {
            normalize_rms(&mut rendered, input_rms);
        }
        return Ok(rendered);
    }

    let num_channels = params.channels.count();
    let channels = deinterleave(input, num_channels);

    let channel_outputs = if num_channels == 2
        && params.stereo_mode == stretch::stereo::StereoMode::MidSide
    {
        let (left, right) = stretch::stereo::stretch_mid_side(&channels[0], &channels[1], params)?;
        vec![left, right]
    } else {
        let mut outs: Vec<Vec<f32>> = Vec::with_capacity(num_channels);
        for channel_data in &channels {
            let stretcher = stretch::hybrid::HybridStretcher::new(params.clone());
            let stretched = stretcher.process(channel_data)?;
            outs.push(stretched);
        }
        outs
    };

    let mut output = interleave(&channel_outputs);

    if params.normalize {
        normalize_rms(&mut output, input_rms);
    }

    Ok(output)
}

/// Stretches audio samples, appending the result to a caller-provided buffer.
///
/// This is the zero-copy variant of [`stretch()`]. Instead of returning a new
/// `Vec`, it appends stretched samples to `output`. This is useful for
/// avoiding heap allocations when the caller already has a pre-allocated buffer.
///
/// Returns the number of samples appended to `output`.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the stretch ratio is out of range.
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input: Vec<f32> = (0..44100)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.5)
///     .with_sample_rate(44100)
///     .with_channels(1)
///     .with_preset(EdmPreset::HouseLoop);
///
/// let mut output = Vec::with_capacity(66150); // pre-allocate for ~1.5x
/// let n = timestretch::stretch_into(&input, &params, &mut output).unwrap();
/// assert!(n > 0);
/// assert_eq!(n, output.len());
/// ```
pub fn stretch_into(
    input: &[f32],
    params: &StretchParams,
    output: &mut Vec<f32>,
) -> Result<usize, StretchError> {
    stretch::params::validate_params(params).map_err(StretchError::InvalidRatio)?;

    if !validate_input(input)? {
        return Ok(0);
    }

    let input_rms = if params.normalize {
        compute_rms(input)
    } else {
        0.0
    };

    if let Some(mut rendered) = try_render_tonal_fast_path(input, params) {
        if params.normalize {
            normalize_rms(&mut rendered, input_rms);
        }
        let n = rendered.len();
        output.extend_from_slice(&rendered);
        return Ok(n);
    }

    let num_channels = params.channels.count();
    let channels = deinterleave(input, num_channels);

    let channel_outputs = if num_channels == 2
        && params.stereo_mode == stretch::stereo::StereoMode::MidSide
    {
        let (left, right) = stretch::stereo::stretch_mid_side(&channels[0], &channels[1], params)?;
        vec![left, right]
    } else {
        let mut outs: Vec<Vec<f32>> = Vec::with_capacity(num_channels);
        for channel_data in &channels {
            let stretcher = stretch::hybrid::HybridStretcher::new(params.clone());
            let stretched = stretcher.process(channel_data)?;
            outs.push(stretched);
        }
        outs
    };

    let min_len = channel_outputs.iter().map(|c| c.len()).min().unwrap_or(0);
    let total = min_len * num_channels;

    output.reserve(total);
    let start = output.len();
    for i in 0..min_len {
        for ch in &channel_outputs {
            output.push(ch[i]);
        }
    }

    if params.normalize {
        normalize_rms(&mut output[start..], input_rms);
    }

    Ok(total)
}

/// Stretches an [`AudioBuffer`] and returns a new `AudioBuffer`.
///
/// The sample rate and channel layout are taken from the input buffer,
/// overriding whatever is set in `params`.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the stretch ratio is out of range.
///
/// # Example
///
/// ```
/// use timestretch::{AudioBuffer, StretchParams, EdmPreset};
///
/// let buffer = AudioBuffer::from_mono(
///     (0..44100)
///         .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///         .collect(),
///     44100,
/// );
/// let params = StretchParams::new(1.5).with_preset(EdmPreset::HouseLoop);
/// let output = timestretch::stretch_buffer(&buffer, &params).unwrap();
/// assert_eq!(output.sample_rate, 44100);
/// ```
pub fn stretch_buffer(
    buffer: &AudioBuffer,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, stretch)
}

/// Shifts the pitch of audio without changing its duration.
///
/// `pitch_factor` > 1.0 raises the pitch; < 1.0 lowers it. For example,
/// `pitch_factor = 2.0` raises the pitch by one octave. This works by
/// time-stretching the audio and then resampling back to the original length
/// using cubic interpolation.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the pitch factor is out of range.
///
/// # Example
///
/// ```
/// use timestretch::StretchParams;
///
/// let input: Vec<f32> = (0..44100)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.0) // ratio is overridden internally
///     .with_sample_rate(44100)
///     .with_channels(1);
/// let output = timestretch::pitch_shift(&input, &params, 1.5).unwrap();
/// // Output has the same length but higher pitch
/// ```
pub fn pitch_shift(
    input: &[f32],
    params: &StretchParams,
    pitch_factor: f64,
) -> Result<Vec<f32>, StretchError> {
    use stretch::params::{RATIO_MAX, RATIO_MIN};
    if !(RATIO_MIN..=RATIO_MAX).contains(&pitch_factor) {
        return Err(StretchError::InvalidRatio(format!(
            "Pitch factor must be between {} and {}, got {}",
            RATIO_MIN, RATIO_MAX, pitch_factor
        )));
    }

    if !validate_input(input)? {
        return Ok(vec![]);
    }

    let input_rms = if params.normalize {
        compute_rms(input)
    } else {
        0.0
    };

    // Step 1: Time-stretch by 1/pitch_factor to compensate for the resampling
    // Disable normalization for the inner stretch — we normalize the final result.
    let stretch_ratio = 1.0 / pitch_factor;
    let mut stretch_params = params.clone();
    stretch_params.stretch_ratio = stretch_ratio;
    stretch_params.normalize = false;
    let stretched = stretch(input, &stretch_params)?;

    if stretched.is_empty() {
        return Ok(vec![]);
    }

    // Step 2: Resample each channel to original length using windowed-sinc interpolation
    let num_channels = params.channels.count();
    let num_input_frames = input.len() / num_channels;
    let input_channels = deinterleave(input, num_channels);
    let stretched_channels = deinterleave(&stretched, num_channels);
    let use_formant_preservation =
        params.envelope_preservation && (pitch_factor - 1.0).abs() > 1e-9;

    let channel_outputs: Vec<Vec<f32>> = stretched_channels
        .iter()
        .enumerate()
        .map(|(idx, ch)| {
            let resampled = core::resample::resample_sinc_default(ch, num_input_frames);
            if use_formant_preservation {
                preserve_formants_after_pitch_shift(
                    &input_channels[idx],
                    &resampled,
                    params,
                    pitch_factor,
                )
            } else {
                resampled
            }
        })
        .collect();

    let mut output = interleave(&channel_outputs);

    if params.normalize {
        normalize_rms(&mut output, input_rms);
    }

    Ok(output)
}

/// Shifts the pitch of an [`AudioBuffer`] without changing its duration.
///
/// Convenience wrapper around [`pitch_shift`] that takes and returns
/// an `AudioBuffer`. Sample rate and channel layout are taken from the buffer.
///
/// # Errors
///
/// Returns [`StretchError::InvalidRatio`] if the pitch factor is out of range.
pub fn pitch_shift_buffer(
    buffer: &AudioBuffer,
    params: &StretchParams,
    pitch_factor: f64,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, |data, p| pitch_shift(data, p, pitch_factor))
}

/// Detects the BPM of a mono audio signal.
///
/// Uses transient detection and inter-onset interval analysis optimized
/// for 4/4 EDM (house/techno) with expected BPM range 100-160. Returns
/// the estimated BPM, or 0.0 if no tempo can be detected.
///
/// For stereo audio, extract the left channel first (or mix to mono).
///
/// # Example
///
/// ```
/// // Generate a click train at ~120 BPM
/// let sample_rate = 44100u32;
/// let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
/// let mut audio = vec![0.0f32; sample_rate as usize * 4];
/// for pos in (0..audio.len()).step_by(beat_interval) {
///     for j in 0..10.min(audio.len() - pos) {
///         audio[pos + j] = if j < 5 { 0.9 } else { -0.4 };
///     }
/// }
///
/// let bpm = timestretch::detect_bpm(&audio, sample_rate);
/// // BPM detection may or may not succeed on synthetic clicks
/// // For real EDM audio with kicks, this is very reliable
/// ```
pub fn detect_bpm(samples: &[f32], sample_rate: u32) -> f64 {
    analysis::beat::detect_beats(samples, sample_rate).bpm
}

/// Detects beats and returns a [`BeatGrid`] with BPM and beat positions.
///
/// This provides more detail than [`detect_bpm`], including the sample
/// positions of detected beats and a grid-snapping utility.
///
/// # Example
///
/// ```
/// let audio = vec![0.0f32; 44100 * 4];
/// let grid = timestretch::detect_beat_grid(&audio, 44100);
/// println!("BPM: {}, beats: {}", grid.bpm, grid.beats.len());
/// ```
pub fn detect_beat_grid(samples: &[f32], sample_rate: u32) -> BeatGrid {
    analysis::beat::detect_beats(samples, sample_rate)
}

/// Detects the BPM of an [`AudioBuffer`].
///
/// For stereo buffers, uses the left channel for detection.
/// Returns 0.0 if no tempo can be detected.
pub fn detect_bpm_buffer(buffer: &AudioBuffer) -> f64 {
    let mono = extract_mono(&buffer.data, buffer.channels.count());
    detect_bpm(&mono, buffer.sample_rate)
}

/// Detects beats in an [`AudioBuffer`] and returns a [`BeatGrid`].
///
/// For stereo buffers, uses the left channel for detection.
/// This is the buffer-based equivalent of [`detect_beat_grid`].
pub fn detect_beat_grid_buffer(buffer: &AudioBuffer) -> BeatGrid {
    let mono = extract_mono(&buffer.data, buffer.channels.count());
    detect_beat_grid(&mono, buffer.sample_rate)
}

/// Stretches audio from one BPM to another.
///
/// Computes the stretch ratio as `source_bpm / target_bpm` and applies
/// time stretching. For example, stretching from 126 BPM to 128 BPM
/// produces a ratio of 126/128 ≈ 0.984 (slightly shorter/faster).
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if either BPM value is invalid,
/// or [`StretchError::InvalidRatio`] if the computed ratio is out of range.
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// let input: Vec<f32> = (0..88200)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
///
/// let params = StretchParams::new(1.0) // ratio will be overridden
///     .with_sample_rate(44100)
///     .with_channels(1)
///     .with_preset(EdmPreset::DjBeatmatch);
///
/// let output = timestretch::stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
/// // Output is slightly shorter (126/128 ≈ 0.984x)
/// assert!(output.len() < input.len());
/// ```
pub fn stretch_to_bpm(
    input: &[f32],
    source_bpm: f64,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<Vec<f32>, StretchError> {
    validate_bpm(source_bpm, "source")?;
    validate_bpm(target_bpm, "target")?;

    let ratio = source_bpm / target_bpm;
    let mut adjusted_params = params.clone();
    adjusted_params.stretch_ratio = ratio;

    stretch(input, &adjusted_params)
}

/// Stretches audio to a target BPM, auto-detecting the source BPM.
///
/// Uses beat detection to estimate the current tempo, then computes the
/// stretch ratio needed to reach `target_bpm`. Best suited for audio
/// with a clear rhythmic pattern (kicks, hi-hats).
///
/// For mono input, pass samples directly. For stereo, pass interleaved
/// L/R samples and set channels to 2 — BPM detection uses the left channel.
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if no tempo can be detected
/// (e.g. the input is too short, contains only silence, or lacks rhythmic content).
///
/// # Example
///
/// ```
/// use timestretch::{StretchParams, EdmPreset};
///
/// // Generate a click train at ~120 BPM for auto-detection
/// let sample_rate = 44100u32;
/// let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
/// let mut audio = vec![0.0f32; sample_rate as usize * 4];
/// for pos in (0..audio.len()).step_by(beat_interval) {
///     for j in 0..20.min(audio.len() - pos) {
///         audio[pos + j] = if j < 5 { 0.9 } else { -0.4 };
///     }
/// }
///
/// let params = StretchParams::new(1.0)
///     .with_sample_rate(sample_rate)
///     .with_channels(1)
///     .with_preset(EdmPreset::DjBeatmatch);
///
/// // Auto-detect BPM and stretch to 128 BPM
/// match timestretch::stretch_to_bpm_auto(&audio, 128.0, &params) {
///     Ok(output) => println!("Stretched {} -> {} samples", audio.len(), output.len()),
///     Err(e) => println!("BPM detection failed: {}", e),
/// }
/// ```
pub fn stretch_to_bpm_auto(
    input: &[f32],
    target_bpm: f64,
    params: &StretchParams,
) -> Result<Vec<f32>, StretchError> {
    validate_bpm(target_bpm, "target")?;

    // Reject non-finite samples before expensive beat detection
    if !validate_input(input)? {
        return Ok(vec![]);
    }

    // Extract mono signal for beat detection
    let mono = extract_mono(input, params.channels.count());

    let beat_grid = analysis::beat::detect_beats(&mono, params.sample_rate);

    if beat_grid.bpm <= 0.0 {
        return Err(StretchError::BpmDetectionFailed(
            "could not detect BPM from input audio (too short or no rhythmic content)".to_string(),
        ));
    }

    stretch_to_bpm(input, beat_grid.bpm, target_bpm, params)
}

/// Stretches an [`AudioBuffer`] from one BPM to another.
///
/// Convenience wrapper around [`stretch_to_bpm`] that takes and returns
/// an `AudioBuffer`. Sample rate and channel layout are taken from the buffer.
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if either BPM value is invalid.
pub fn stretch_bpm_buffer(
    buffer: &AudioBuffer,
    source_bpm: f64,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, |data, p| {
        stretch_to_bpm(data, source_bpm, target_bpm, p)
    })
}

/// Stretches an [`AudioBuffer`] to a target BPM, auto-detecting the source BPM.
///
/// Convenience wrapper around [`stretch_to_bpm_auto`] that takes and returns
/// an `AudioBuffer`. Sample rate and channel layout are taken from the buffer.
///
/// # Errors
///
/// Returns [`StretchError::BpmDetectionFailed`] if no tempo can be detected.
pub fn stretch_bpm_buffer_auto(
    buffer: &AudioBuffer,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    process_buffer(buffer, params, |data, p| {
        stretch_to_bpm_auto(data, target_bpm, p)
    })
}

/// Reads a WAV file, stretches it, and writes the result to another WAV file.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file and passed through automatically.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// [`StretchError::InvalidFormat`] if the input is not a valid WAV file,
/// or [`StretchError::InvalidRatio`] if the stretch ratio is out of range.
pub fn stretch_wav_file(
    input_path: &str,
    output_path: &str,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = stretch_buffer(&buffer, params)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Reads a WAV file, stretches it from one BPM to another, and writes the result.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file and passed through automatically.
/// This is a convenience function combining WAV I/O with BPM-based stretching.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// [`StretchError::InvalidFormat`] if the input is not a valid WAV file,
/// [`StretchError::BpmDetectionFailed`] if either BPM value is invalid.
pub fn stretch_to_bpm_wav_file(
    input_path: &str,
    output_path: &str,
    source_bpm: f64,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = stretch_bpm_buffer(&buffer, source_bpm, target_bpm, params)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Reads a WAV file, auto-detects its BPM, stretches to the target BPM, and writes the result.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// [`StretchError::InvalidFormat`] if the input is not a valid WAV file,
/// or [`StretchError::BpmDetectionFailed`] if no tempo can be detected.
pub fn stretch_to_bpm_auto_wav_file(
    input_path: &str,
    output_path: &str,
    target_bpm: f64,
    params: &StretchParams,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = stretch_bpm_buffer_auto(&buffer, target_bpm, params)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Reads a WAV file, pitch-shifts it, and writes the result to another WAV file.
///
/// The output is written as 32-bit float WAV. The sample rate and channel
/// layout are read from the input file.
///
/// # Errors
///
/// Returns [`StretchError::IoError`] if the files cannot be read or written,
/// or [`StretchError::InvalidRatio`] if the pitch factor is out of range.
pub fn pitch_shift_wav_file(
    input_path: &str,
    output_path: &str,
    params: &StretchParams,
    pitch_factor: f64,
) -> Result<AudioBuffer, StretchError> {
    let buffer = io::wav::read_wav_file(input_path)?;
    let result = pitch_shift_buffer(&buffer, params, pitch_factor)?;
    io::wav::write_wav_file_float(output_path, &result)?;
    Ok(result)
}

/// Returns the stretch ratio needed to change from one BPM to another.
///
/// This is a simple utility: `source_bpm / target_bpm`. Use it when you
/// want to compute the ratio yourself before calling [`stretch()`].
///
/// # Example
///
/// ```
/// let ratio = timestretch::bpm_ratio(126.0, 128.0);
/// assert!((ratio - 0.984375).abs() < 1e-6);
/// ```
#[inline]
pub fn bpm_ratio(source_bpm: f64, target_bpm: f64) -> f64 {
    source_bpm / target_bpm
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time assertions that key public types are Send + Sync.
    // This is critical for real-time audio where processing often runs
    // on a dedicated thread.
    const _: () = {
        fn assert_send_sync<T: Send + Sync>() {}
        fn check() {
            assert_send_sync::<AudioBuffer>();
            assert_send_sync::<StretchParams>();
            assert_send_sync::<StreamProcessor>();
            assert_send_sync::<StretchError>();
            assert_send_sync::<BeatGrid>();
        }
        let _ = check;
    };

    #[test]
    fn test_stretch_empty() {
        let params = StretchParams::new(1.5);
        let output = stretch(&[], &params).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_stretch_mono_sine() {
        let sample_rate = 44100u32;
        let duration = 2.0;
        let num_samples = (sample_rate as f64 * duration) as usize;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.5,
            "Length ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_stretch_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];

        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin(); // L
            input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin(); // R
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(2);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
        // Output should have even number of samples (stereo)
        assert_eq!(output.len() % 2, 0);
    }

    #[test]
    fn test_stretch_invalid_ratio() {
        let params = StretchParams::new(0.0);
        assert!(stretch(&[0.0; 44100], &params).is_err());
    }

    #[test]
    fn test_stretch_buffer() {
        let buffer = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
                .collect(),
            44100,
        );

        let params = StretchParams::new(1.5);
        let output = stretch_buffer(&buffer, &params).unwrap();
        assert_eq!(output.sample_rate, 44100);
        assert_eq!(output.channels, Channels::Mono);
        assert!(!output.data.is_empty());
    }

    #[test]
    fn test_pitch_shift_preserves_length() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output = pitch_shift(&input, &params, 1.5).unwrap();
        // Output should have the same length as input
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_pitch_shift_empty() {
        let params = StretchParams::new(1.0);
        let output = pitch_shift(&[], &params, 1.5).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_pitch_shift_invalid_factor() {
        let params = StretchParams::new(1.0);
        assert!(pitch_shift(&[0.0; 44100], &params, 0.0).is_err());
        assert!(pitch_shift(&[0.0; 44100], &params, -1.0).is_err());
        assert!(pitch_shift(&[0.0; 44100], &params, 200.0).is_err());
    }

    #[test]
    fn test_pitch_shift_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(2);

        let output = pitch_shift(&input, &params, 0.8).unwrap();
        assert_eq!(output.len(), input.len());
        assert_eq!(output.len() % 2, 0);
    }

    #[test]
    fn test_vocal_envelope_strength_scale_for_pitch() {
        let down = envelope_strength_scale_for_pitch(0.75, EnvelopePreset::Vocal);
        let near_unity = envelope_strength_scale_for_pitch(1.0, EnvelopePreset::Vocal);
        let mild_up = envelope_strength_scale_for_pitch(1.35, EnvelopePreset::Vocal);
        let extreme_up = envelope_strength_scale_for_pitch(2.5, EnvelopePreset::Vocal);

        assert!(down < near_unity);
        assert!(down <= 1e-6);
        assert!((mild_up - 1.0).abs() < 1e-6);
        assert!(extreme_up < near_unity);

        let non_vocal = envelope_strength_scale_for_pitch(0.75, EnvelopePreset::Balanced);
        assert!((non_vocal - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stretch_dj_beatmatch_preset() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Small ratio change: 126 BPM -> 128 BPM
        let ratio = 126.0 / 128.0; // ~0.984
        let params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_bpm_ratio() {
        let ratio = bpm_ratio(126.0, 128.0);
        assert!((ratio - 0.984375).abs() < 1e-6);

        // Same BPM = ratio 1.0
        assert!((bpm_ratio(120.0, 120.0) - 1.0).abs() < 1e-10);

        // Double BPM = half length
        assert!((bpm_ratio(120.0, 240.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stretch_to_bpm_basic() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);

        // 126 -> 128 BPM: should produce slightly shorter output
        let output = stretch_to_bpm(&input, 126.0, 128.0, &params).unwrap();
        let expected_ratio = 126.0 / 128.0;
        let actual_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.3,
            "BPM stretch ratio: expected ~{:.3}, got {:.3}",
            expected_ratio,
            actual_ratio
        );
    }

    #[test]
    fn test_stretch_to_bpm_speedup() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // 120 -> 150 BPM: significant speedup (ratio 0.8)
        let output = stretch_to_bpm(&input, 120.0, 150.0, &params).unwrap();
        assert!(
            output.len() < input.len(),
            "Should be shorter when speeding up"
        );
    }

    #[test]
    fn test_stretch_to_bpm_slowdown() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // 120 -> 90 BPM: slow down (ratio 1.333)
        let output = stretch_to_bpm(&input, 120.0, 90.0, &params).unwrap();
        assert!(
            output.len() > input.len(),
            "Should be longer when slowing down"
        );
    }

    #[test]
    fn test_stretch_to_bpm_invalid_bpm() {
        let params = StretchParams::new(1.0);
        let input = vec![0.0f32; 44100];

        // Zero source BPM
        assert!(stretch_to_bpm(&input, 0.0, 128.0, &params).is_err());
        // Negative source BPM
        assert!(stretch_to_bpm(&input, -120.0, 128.0, &params).is_err());
        // Zero target BPM
        assert!(stretch_to_bpm(&input, 120.0, 0.0, &params).is_err());
        // Negative target BPM
        assert!(stretch_to_bpm(&input, 120.0, -128.0, &params).is_err());
    }

    #[test]
    fn test_stretch_to_bpm_same_bpm() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // Same BPM: ratio 1.0, output length ~ input length
        let output = stretch_to_bpm(&input, 128.0, 128.0, &params).unwrap();
        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.0).abs() < 0.1,
            "Same BPM should preserve length, got ratio {}",
            len_ratio
        );
    }

    #[test]
    fn test_stretch_to_bpm_empty() {
        let params = StretchParams::new(1.0);
        let output = stretch_to_bpm(&[], 120.0, 128.0, &params).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_stretch_to_bpm_auto_silence() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);

        // Silence has no beats, should return BpmDetectionFailed
        let silence = vec![0.0f32; 44100 * 4];
        let result = stretch_to_bpm_auto(&silence, 128.0, &params);
        assert!(result.is_err());
        match result {
            Err(StretchError::BpmDetectionFailed(_)) => {} // expected
            other => panic!("Expected BpmDetectionFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_stretch_to_bpm_auto_invalid_target() {
        let params = StretchParams::new(1.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let input = vec![0.0f32; 44100];

        assert!(stretch_to_bpm_auto(&input, 0.0, &params).is_err());
        assert!(stretch_to_bpm_auto(&input, -128.0, &params).is_err());
    }

    #[test]
    fn test_stretch_bpm_buffer() {
        let sample_rate = 44100u32;
        let buffer = AudioBuffer::from_mono(
            (0..sample_rate as usize * 2)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
                .collect(),
            sample_rate,
        );

        let params = StretchParams::new(1.0).with_preset(EdmPreset::DjBeatmatch);
        let output = stretch_bpm_buffer(&buffer, 126.0, 128.0, &params).unwrap();
        assert_eq!(output.sample_rate, sample_rate);
        assert_eq!(output.channels, Channels::Mono);
        assert!(output.data.len() < buffer.data.len()); // Speeding up
    }

    #[test]
    fn test_stretch_rejects_nan() {
        let mut input = vec![0.0f32; 44100];
        input[1000] = f32::NAN;
        let params = StretchParams::new(1.5).with_channels(1);
        assert!(matches!(
            stretch(&input, &params),
            Err(StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_stretch_rejects_infinity() {
        let mut input = vec![0.0f32; 44100];
        input[500] = f32::INFINITY;
        let params = StretchParams::new(1.5).with_channels(1);
        assert!(matches!(
            stretch(&input, &params),
            Err(StretchError::NonFiniteInput)
        ));

        input[500] = f32::NEG_INFINITY;
        assert!(matches!(
            stretch(&input, &params),
            Err(StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_pitch_shift_rejects_nan() {
        let mut input = vec![0.0f32; 44100];
        input[100] = f32::NAN;
        let params = StretchParams::new(1.0).with_channels(1);
        assert!(matches!(
            pitch_shift(&input, &params, 1.5),
            Err(StretchError::NonFiniteInput)
        ));
    }

    #[test]
    fn test_from_tempo_stretch() {
        // Verify from_tempo integrates with stretch()
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let params = StretchParams::from_tempo(126.0, 128.0)
            .with_sample_rate(44100)
            .with_channels(1);
        let output = stretch(&input, &params).unwrap();
        // Compressing: output should be shorter than input
        assert!(output.len() < input.len());
    }

    #[test]
    fn test_detect_bpm_silence() {
        // Silence should return 0 BPM
        let silence = vec![0.0f32; 44100 * 4];
        let bpm = detect_bpm(&silence, 44100);
        assert!(bpm == 0.0, "Silence should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_bpm_empty() {
        let bpm = detect_bpm(&[], 44100);
        assert!(bpm == 0.0, "Empty input should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_bpm_short_input() {
        // Very short input should not crash
        let short = vec![0.5f32; 100];
        let bpm = detect_bpm(&short, 44100);
        // Should return 0 or some value, but not crash
        assert!(bpm >= 0.0);
    }

    #[test]
    fn test_detect_beat_grid_returns_grid() {
        let sample_rate = 44100u32;
        // Create a click train at ~120 BPM
        let beat_interval = (60.0 * sample_rate as f64 / 120.0) as usize;
        let num_samples = sample_rate as usize * 4;
        let mut audio = vec![0.0f32; num_samples];

        for pos in (0..num_samples).step_by(beat_interval) {
            for j in 0..20.min(num_samples - pos) {
                audio[pos + j] = if j < 5 { 0.9 } else { -0.4 };
            }
            // Add tone between clicks for transient detector
            let tone_start = pos + 20;
            let tone_end = (pos + beat_interval / 2).min(num_samples);
            for (i, sample) in audio[tone_start..tone_end].iter_mut().enumerate() {
                let idx = tone_start + i;
                *sample += 0.2
                    * (2.0 * std::f32::consts::PI * 200.0 * idx as f32 / sample_rate as f32).sin();
            }
        }

        let grid = detect_beat_grid(&audio, sample_rate);
        assert_eq!(grid.sample_rate, sample_rate);
        // Beat grid should have reasonable interval if beats were detected
        if grid.bpm > 0.0 {
            let interval = grid.beat_interval_samples();
            assert!(interval > 0.0, "Beat interval should be positive");
        }
    }

    #[test]
    fn test_detect_bpm_with_click_train() {
        let sample_rate = 44100u32;
        let target_bpm = 120.0;
        let beat_interval = (60.0 * sample_rate as f64 / target_bpm) as usize;
        let num_samples = sample_rate as usize * 6; // 6 seconds

        let mut audio = vec![0.0f32; num_samples];

        // Create strong clicks at beat positions
        for pos in (0..num_samples).step_by(beat_interval) {
            for j in 0..10.min(num_samples - pos) {
                audio[pos + j] = if j < 5 { 0.95 } else { -0.5 };
            }
        }

        // Add background tone
        for (i, sample) in audio.iter_mut().enumerate() {
            *sample +=
                0.15 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sample_rate as f32).sin();
        }

        let bpm = detect_bpm(&audio, sample_rate);
        // BPM detection is heuristic; may succeed or detect a harmonic (e.g., 240 BPM)
        // but should return something in the EDM range if it finds beats
        if bpm > 0.0 {
            assert!(
                (100.0..=160.0).contains(&bpm),
                "BPM {} should be in EDM range 100-160",
                bpm
            );
        }
    }

    #[test]
    fn test_pitch_shift_buffer() {
        let buffer = AudioBuffer::from_mono(
            (0..44100)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
                .collect(),
            44100,
        );

        let params = StretchParams::new(1.0);
        let output = pitch_shift_buffer(&buffer, &params, 1.5).unwrap();
        assert_eq!(output.sample_rate, 44100);
        assert_eq!(output.channels, Channels::Mono);
        // Pitch shift preserves length
        assert_eq!(output.data.len(), buffer.data.len());
    }

    #[test]
    fn test_pitch_shift_buffer_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut data = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            data[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            data[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let buffer = AudioBuffer::new(data, sample_rate, Channels::Stereo);
        let params = StretchParams::new(1.0);
        let output = pitch_shift_buffer(&buffer, &params, 0.8).unwrap();
        assert_eq!(output.data.len(), buffer.data.len());
        assert_eq!(output.channels, Channels::Stereo);
    }

    #[test]
    fn test_detect_bpm_buffer_silence() {
        let buffer = AudioBuffer::from_mono(vec![0.0f32; 44100 * 4], 44100);
        let bpm = detect_bpm_buffer(&buffer);
        assert!(bpm == 0.0, "Silence should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_bpm_buffer_stereo() {
        // Stereo buffer with silence should return 0 BPM and not crash
        let data = vec![0.0f32; 44100 * 4 * 2]; // 4 seconds stereo
        let buffer = AudioBuffer::new(data, 44100, Channels::Stereo);
        let bpm = detect_bpm_buffer(&buffer);
        assert!(bpm == 0.0, "Silence should return 0 BPM, got {}", bpm);
    }

    #[test]
    fn test_detect_beat_grid_buffer_mono() {
        let buffer = AudioBuffer::from_mono(vec![0.0f32; 44100 * 4], 44100);
        let grid = detect_beat_grid_buffer(&buffer);
        assert_eq!(grid.sample_rate, 44100);
    }

    #[test]
    fn test_detect_beat_grid_buffer_stereo() {
        let data = vec![0.0f32; 44100 * 4 * 2]; // 4 seconds stereo
        let buffer = AudioBuffer::new(data, 44100, Channels::Stereo);
        let grid = detect_beat_grid_buffer(&buffer);
        assert_eq!(grid.sample_rate, 44100);
        // Silence should yield 0 BPM
        assert!(
            grid.bpm == 0.0,
            "Silence should return 0 BPM, got {}",
            grid.bpm
        );
    }

    #[test]
    fn test_stretch_wav_file() {
        // Create a temp WAV file
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let dir = std::env::temp_dir();
        let in_path = dir.join("timestretch_test_in.wav");
        let out_path = dir.join("timestretch_test_out.wav");
        io::wav::write_wav_file_float(in_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.5).with_channels(1);
        let result = stretch_wav_file(
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            &params,
        )
        .unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.channels, Channels::Mono);

        // Verify the output file was written
        let reloaded = io::wav::read_wav_file(out_path.to_str().unwrap()).unwrap();
        assert_eq!(reloaded.data.len(), result.data.len());

        // Clean up
        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn test_pitch_shift_wav_file() {
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let dir = std::env::temp_dir();
        let in_path = dir.join("timestretch_pitch_in.wav");
        let out_path = dir.join("timestretch_pitch_out.wav");
        io::wav::write_wav_file_float(in_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.0).with_channels(1);
        let result = pitch_shift_wav_file(
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            &params,
            1.5,
        )
        .unwrap();

        assert!(!result.is_empty());
        // Pitch shift preserves length
        assert_eq!(result.data.len(), buf.data.len());

        // Clean up
        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn test_stretch_wav_file_missing_input() {
        let params = StretchParams::new(1.5);
        let result = stretch_wav_file("/nonexistent/path/input.wav", "/tmp/output.wav", &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_stretch_to_bpm_wav_file() {
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buf = AudioBuffer::from_mono(input, 44100);
        let dir = std::env::temp_dir();
        let in_path = dir.join("timestretch_bpm_in.wav");
        let out_path = dir.join("timestretch_bpm_out.wav");
        io::wav::write_wav_file_float(in_path.to_str().unwrap(), &buf).unwrap();

        let params = StretchParams::new(1.0).with_channels(1);
        let result = stretch_to_bpm_wav_file(
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            126.0,
            128.0,
            &params,
        )
        .unwrap();

        // Ratio should be 126/128 ≈ 0.984, so output slightly shorter
        assert!(result.data.len() < 44100);
        assert!(!result.is_empty());

        // Verify output was written
        let reloaded = io::wav::read_wav_file(out_path.to_str().unwrap()).unwrap();
        assert_eq!(reloaded.data.len(), result.data.len());
    }

    #[test]
    fn test_normalize_preserves_rms() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let output = stretch(&input, &params).unwrap();
        let output_rms = compute_rms(&output);

        // With normalization, output RMS should be very close to input RMS
        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.05,
            "Normalized RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_normalize_off_by_default() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        // Without normalization, stretch should still work
        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_normalize_with_silence() {
        // Normalization should not amplify silence
        let silence = vec![0.0f32; 44100];
        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1)
            .with_normalize(true);

        let output = stretch(&silence, &params).unwrap();
        let max_val = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 1e-4,
            "Normalized silence should stay silent, got max={}",
            max_val
        );
    }

    #[test]
    fn test_normalize_with_compression() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.6 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        // Compression (ratio < 1.0)
        let params = StretchParams::new(0.75)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let output = stretch(&input, &params).unwrap();
        let output_rms = compute_rms(&output);

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.1,
            "Normalized compression RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_stretch_with_window_type() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Stretch with Blackman-Harris window
        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_window_type(core::window::WindowType::BlackmanHarris);

        let output = stretch(&input, &params).unwrap();
        assert!(!output.is_empty());

        let len_ratio = output.len() as f64 / input.len() as f64;
        assert!(
            (len_ratio - 1.5).abs() < 0.5,
            "BH stretch ratio {} too far from 1.5",
            len_ratio
        );
    }

    #[test]
    fn test_pitch_shift_with_normalize() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;

        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.7 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        let params = StretchParams::new(1.0)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let output = pitch_shift(&input, &params, 1.5).unwrap();
        let output_rms = compute_rms(&output);

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.1,
            "Normalized pitch shift RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }

    // --- stretch_into tests ---

    #[test]
    fn test_stretch_into_matches_stretch() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1);

        let output1 = stretch(&input, &params).unwrap();

        let mut output2 = Vec::new();
        let n = stretch_into(&input, &params, &mut output2).unwrap();

        assert_eq!(n, output2.len());
        assert_eq!(output1.len(), output2.len());
        for (i, (a, b)) in output1.iter().zip(output2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Mismatch at sample {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_stretch_into_empty() {
        let params = StretchParams::new(1.5);
        let mut output = Vec::new();
        let n = stretch_into(&[], &params, &mut output).unwrap();
        assert_eq!(n, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_stretch_into_appends() {
        let input: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let params = StretchParams::new(1.5)
            .with_sample_rate(44100)
            .with_channels(1);

        let mut output = vec![99.0f32; 3]; // pre-existing data
        let n = stretch_into(&input, &params, &mut output).unwrap();

        // First 3 samples should be our sentinel values
        assert!((output[0] - 99.0).abs() < 1e-6);
        assert!((output[1] - 99.0).abs() < 1e-6);
        assert!((output[2] - 99.0).abs() < 1e-6);
        assert_eq!(output.len(), 3 + n);
    }

    #[test]
    fn test_stretch_into_invalid_ratio() {
        let params = StretchParams::new(0.0);
        let mut output = Vec::new();
        assert!(stretch_into(&[0.0; 44100], &params, &mut output).is_err());
    }

    #[test]
    fn test_stretch_into_stereo() {
        let sample_rate = 44100u32;
        let num_frames = 44100;
        let mut input = vec![0.0f32; num_frames * 2];
        for i in 0..num_frames {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            input[i * 2 + 1] = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        }

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(2);

        let mut output = Vec::new();
        let n = stretch_into(&input, &params, &mut output).unwrap();
        assert!(n > 0);
        assert_eq!(n % 2, 0, "Stereo output must have even sample count");
    }

    #[test]
    fn test_stretch_into_with_normalize() {
        let sample_rate = 44100u32;
        let num_samples = sample_rate as usize * 2;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| {
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let input_rms = compute_rms(&input);

        let params = StretchParams::new(1.5)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_normalize(true);

        let mut output = Vec::new();
        stretch_into(&input, &params, &mut output).unwrap();
        let output_rms = compute_rms(&output);

        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.05,
            "Normalized stretch_into RMS mismatch: input={:.4}, output={:.4}",
            input_rms,
            output_rms
        );
    }

    #[test]
    fn test_stretch_into_rejects_nan() {
        let mut input = vec![0.0f32; 44100];
        input[1000] = f32::NAN;
        let params = StretchParams::new(1.5).with_channels(1);
        let mut output = Vec::new();
        assert!(matches!(
            stretch_into(&input, &params, &mut output),
            Err(StretchError::NonFiniteInput)
        ));
        assert!(output.is_empty());
    }
}
