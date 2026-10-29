//! Musical key detection via chroma features and Krumhansl-Kessler profiles.
//!
//! Pipeline: STFT magnitudes restricted to the 100–5000 Hz band (below,
//! kick/bass energy is weakly pitched; above, broadband percussive content
//! dominates) are power-compressed. A global tuning offset is estimated
//! from the circular mean of each bin's distance to the nearest
//! equal-tempered semitone, so tracks mastered off A440 don't smear energy
//! across pitch-class boundaries. Per-frame chroma is HPCP-style
//! (Gómez 2006): spectral peaks only, each attributed to its candidate
//! fundamentals `f/h` with decaying weight, which returns harmonic-series
//! energy (e.g. the 5th harmonic, a major third) to the fundamental's
//! pitch class. Frame vectors are L2-normalized, averaged over the track,
//! and scored against the 24 rotated Krumhansl-Kessler major/minor
//! profiles by Pearson correlation.
//!
//! Because the profiles separate parallel modes mainly at the 3rd/6th/7th
//! scale degrees — weak evidence in percussion-heavy mixes — a final
//! heuristic re-checks those degrees on a second, harmonically-emphasized
//! view of the same spectrogram (HPSS-style Wiener mask: temporally stable
//! energy kept, broadband hits attenuated) and flips the winner's mode
//! when they clearly contradict it. The masked view is deliberately not
//! used for template scoring: masking can suppress short chord stabs and
//! shift the detected root.

use crate::analysis::hpss::{median_filter_horizontal, median_filter_vertical};
use crate::core::preanalysis::{KeyEstimate, KeyMode};
use crate::core::window::{WindowType, generate_window};
use rustfft::{FftPlanner, num_complex::Complex};

/// Analysis frame size. At 44.1 kHz this gives ~5.4 Hz bins, enough to
/// separate adjacent semitones down to the ~100 Hz band edge.
const KEY_FFT_SIZE: usize = 8192;
/// Analysis hop (50% overlap). Key content moves slowly; finer time
/// resolution buys nothing here.
const KEY_HOP_SIZE: usize = 4096;

/// Chroma band limits in Hz.
const CHROMA_FMIN_HZ: f32 = 100.0;
const CHROMA_FMAX_HZ: f32 = 5000.0;

/// Power-law magnitude compression exponent; tames broadband energy so
/// tonal partials dominate the chroma.
const MAGNITUDE_COMPRESSION: f32 = 0.6;

/// Frames whose compressed-magnitude sum is below this fraction of the
/// loudest frame are treated as silence and skipped.
const FRAME_ENERGY_GATE_RATIO: f32 = 0.01;

/// Time-axis median width (frames) for the harmonic estimate: ~460 ms at
/// the 8192/4096 configuration, long enough to span drum hits but short
/// enough to keep chord stabs.
const HARMONIC_MEDIAN_FRAMES: usize = 3;
/// Frequency-axis median width (bins) for the percussive estimate
/// (~92 Hz at 44.1 kHz): wide against tonal peaks, narrow against
/// broadband hits.
const PERCUSSIVE_MEDIAN_BINS: usize = 17;

/// Number of candidate fundamentals `f/h` each spectral peak is
/// attributed to (HPCP-style subharmonic summation).
const HARMONIC_ATTRIBUTION_COUNT: usize = 6;
/// Weight decay per harmonic number for subharmonic attribution.
const HARMONIC_DECAY: f32 = 0.75;
/// Lowest fundamental a peak may be attributed to (A1). Below this,
/// candidates stop being musically plausible root notes.
const FUNDAMENTAL_MIN_HZ: f32 = 55.0;
/// Peaks below this fraction of the frame's strongest bin are ignored.
const PEAK_FLOOR_RATIO: f32 = 1e-3;

/// Mode-flip gate: evidence for the opposite mode at the 3rd/6th/7th
/// degrees must exceed the winner's by this factor.
const MODE_FLIP_DOMINANCE: f64 = 1.5;
/// Mode-flip gate: the flipped key's own template score must reach this
/// fraction of the winning score, so the flip never lands on a key the
/// profiles flatly contradict.
const MODE_FLIP_MIN_SCORE_RATIO: f64 = 0.6;

/// Krumhansl-Kessler probe-tone profiles for C major / C minor, pitch
/// classes C..B (Krumhansl & Kessler 1982).
const KK_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KK_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Estimates the musical key of a mono signal.
///
/// Returns `None` when the signal is too short for a single analysis frame
/// or contains no tonal energy to correlate. Confidence is the relative
/// margin of the winning key over the runner-up (the runner-up is often
/// the relative major/minor, so absolute values are modest even on
/// clearly tonal material).
pub fn detect_key(samples: &[f32], sample_rate: u32) -> Option<KeyEstimate> {
    if sample_rate == 0 || samples.len() < KEY_FFT_SIZE {
        return None;
    }

    let bin_hz = sample_rate as f32 / KEY_FFT_SIZE as f32;
    let bin_lo = ((CHROMA_FMIN_HZ / bin_hz).ceil() as usize).max(1);
    let bin_hi = ((CHROMA_FMAX_HZ / bin_hz).floor() as usize).min(KEY_FFT_SIZE / 2);
    if bin_lo >= bin_hi {
        return None;
    }

    // Semitone position of each bin relative to A4 = 440 Hz = semitone 57
    // (C-based numbering, so pitch class = semitone mod 12 with C = 0).
    let bin_semitones: Vec<f32> = (bin_lo..bin_hi)
        .map(|b| 12.0 * (b as f32 * bin_hz / 440.0).log2() + 57.0)
        .collect();

    // One STFT pass. Two views of the same spectrogram: the plain view
    // drives template scoring (robust — masking can suppress short chord
    // stabs and shift the root), the harmonically-masked view drives the
    // parallel-mode check (percussion dilutes exactly the 3rd/6th/7th
    // evidence that check needs).
    let mut frame_mags = compute_frame_magnitudes(samples, bin_lo, bin_hi);
    if frame_mags.is_empty() {
        return None;
    }
    let mut masked_mags = frame_mags.clone();
    apply_harmonic_mask(&mut masked_mags);
    compress_magnitudes(&mut frame_mags);

    let tuning_offset = estimate_tuning_offset(&frame_mags, &bin_semitones);

    let binning = ChromaBinning {
        bin_lo,
        bin_hz,
        tuning_offset,
    };
    let mean_chroma = mean_chroma_of(&frame_mags, &binning)?;
    let mode_chroma = mean_chroma_of(&masked_mags, &binning).unwrap_or(mean_chroma);

    score_against_profiles(&mean_chroma, &mode_chroma)
}

/// Shared parameters for mapping spectral peaks to pitch classes.
#[derive(Clone, Copy)]
struct ChromaBinning {
    /// First stored bin's index in the full FFT spectrum.
    bin_lo: usize,
    /// Bin width in Hz.
    bin_hz: f32,
    /// Global tuning offset in semitones.
    tuning_offset: f32,
}

/// Averages L2-normalized per-frame HPCP chroma over all frames with
/// meaningful energy. `None` when every frame is silent.
fn mean_chroma_of(frame_mags: &[Vec<f32>], binning: &ChromaBinning) -> Option<[f64; 12]> {
    let frame_sums: Vec<f32> = frame_mags.iter().map(|m| m.iter().sum()).collect();
    let max_sum = frame_sums.iter().copied().fold(0.0f32, f32::max);
    if max_sum <= 0.0 {
        return None;
    }
    let gate = max_sum * FRAME_ENERGY_GATE_RATIO;

    let mut mean_chroma = [0.0f64; 12];
    let mut used_frames = 0usize;
    for (mags, &sum) in frame_mags.iter().zip(&frame_sums) {
        if sum < gate {
            continue;
        }
        let chroma = frame_chroma_hpcp(mags, binning);
        let norm = chroma.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            continue;
        }
        for (acc, v) in mean_chroma.iter_mut().zip(&chroma) {
            *acc += f64::from(v / norm);
        }
        used_frames += 1;
    }
    if used_frames == 0 {
        return None;
    }
    for v in &mut mean_chroma {
        *v /= used_frames as f64;
    }
    Some(mean_chroma)
}

/// HPCP-style chroma for one frame: spectral peaks only, each attributed
/// to the pitch classes of its candidate fundamentals `f/h` with decaying
/// weight (Gómez 2006).
///
/// Bin-by-bin chroma credits every partial to its *own* pitch class, so
/// the 5th harmonic of the root systematically inflates the major-third
/// bin (harmonic-series leakage). Subharmonic attribution instead returns
/// most of that energy to the fundamental's pitch class; percussive
/// plateaus contribute little because only local maxima are counted.
fn frame_chroma_hpcp(mags: &[f32], binning: &ChromaBinning) -> [f32; 12] {
    let mut chroma = [0.0f32; 12];
    let frame_max = mags.iter().copied().fold(0.0f32, f32::max);
    if frame_max <= 0.0 {
        return chroma;
    }
    let floor = frame_max * PEAK_FLOOR_RATIO;

    for b in 1..mags.len().saturating_sub(1) {
        let m = mags[b];
        if m <= floor || m <= mags[b - 1] || m < mags[b + 1] {
            continue;
        }
        // Parabolic refinement of the peak position, in bins.
        let denom = mags[b - 1] - 2.0 * m + mags[b + 1];
        let delta = if denom.abs() > f32::EPSILON {
            (0.5 * (mags[b - 1] - mags[b + 1]) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let freq = (binning.bin_lo + b) as f32 + delta;
        let freq = freq * binning.bin_hz;

        for h in 1..=HARMONIC_ATTRIBUTION_COUNT {
            let fundamental = freq / h as f32;
            if fundamental < FUNDAMENTAL_MIN_HZ {
                break;
            }
            let semitone = 12.0 * (fundamental / 440.0).log2() + 57.0;
            let pc = ((semitone - binning.tuning_offset).round() as i32).rem_euclid(12) as usize;
            chroma[pc] += m * HARMONIC_DECAY.powi(h as i32 - 1);
        }
    }
    chroma
}

/// Applies power-law compression in place (for the unmasked view;
/// [`apply_harmonic_mask`] compresses as part of masking).
fn compress_magnitudes(frame_mags: &mut [Vec<f32>]) {
    for mags in frame_mags {
        for m in mags {
            *m = m.powf(MAGNITUDE_COMPRESSION);
        }
    }
}

/// Runs the STFT and returns raw magnitudes for bins `[bin_lo, bin_hi)`,
/// one vector per frame.
fn compute_frame_magnitudes(samples: &[f32], bin_lo: usize, bin_hi: usize) -> Vec<Vec<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(KEY_FFT_SIZE);
    let window = generate_window(WindowType::Hann, KEY_FFT_SIZE);

    let mut frames = Vec::new();
    let mut buffer = vec![Complex::new(0.0f32, 0.0f32); KEY_FFT_SIZE];
    let mut start = 0usize;
    while start + KEY_FFT_SIZE <= samples.len() {
        for ((slot, &s), &w) in buffer
            .iter_mut()
            .zip(&samples[start..start + KEY_FFT_SIZE])
            .zip(&window)
        {
            *slot = Complex::new(s * w, 0.0);
        }
        fft.process(&mut buffer);
        frames.push(buffer[bin_lo..bin_hi].iter().map(|c| c.norm()).collect());
        start += KEY_HOP_SIZE;
    }
    frames
}

/// Emphasizes harmonic content in place, then power-compresses.
///
/// HPSS-style soft masking on the key spectrogram itself (no resynthesis):
/// the time-axis median estimates temporally stable (harmonic) energy, the
/// frequency-axis median estimates broadband (percussive) energy, and each
/// magnitude is scaled by the Wiener mask `h² / (h² + p²)`.
fn apply_harmonic_mask(frame_mags: &mut [Vec<f32>]) {
    let harmonic = median_filter_horizontal(frame_mags, HARMONIC_MEDIAN_FRAMES);
    let percussive = median_filter_vertical(frame_mags, PERCUSSIVE_MEDIAN_BINS);
    for (mags, (h_row, p_row)) in frame_mags.iter_mut().zip(harmonic.iter().zip(&percussive)) {
        for (m, (&h, &p)) in mags.iter_mut().zip(h_row.iter().zip(p_row)) {
            let h2 = h * h;
            let p2 = p * p;
            *m = (*m * h2 / (h2 + p2 + 1e-10)).powf(MAGNITUDE_COMPRESSION);
        }
    }
}

/// Estimates the global tuning offset in semitones, in [-0.5, 0.5).
///
/// Each bin's residual to the nearest semitone is mapped to an angle on
/// the circle; the magnitude-weighted circular mean is the offset. This is
/// robust to the residual wrap-around at ±0.5 semitones.
fn estimate_tuning_offset(frame_mags: &[Vec<f32>], bin_semitones: &[f32]) -> f32 {
    let mut sum_cos = 0.0f64;
    let mut sum_sin = 0.0f64;
    for mags in frame_mags {
        for (&m, &s) in mags.iter().zip(bin_semitones) {
            let residual = s - s.round();
            let angle = f64::from(residual) * std::f64::consts::TAU;
            sum_cos += f64::from(m) * angle.cos();
            sum_sin += f64::from(m) * angle.sin();
        }
    }
    if sum_cos == 0.0 && sum_sin == 0.0 {
        return 0.0;
    }
    (sum_sin.atan2(sum_cos) / std::f64::consts::TAU) as f32
}

/// Correlates the mean chroma against all 24 rotated K-K profiles, applies
/// the parallel-mode heuristic (on the harmonically-masked `mode_chroma`),
/// and returns the winner with a margin-based confidence.
fn score_against_profiles(mean_chroma: &[f64; 12], mode_chroma: &[f64; 12]) -> Option<KeyEstimate> {
    // scores[mode][root]; mode 0 = major, 1 = minor.
    let mut scores = [[f64::NEG_INFINITY; 12]; 2];
    for (m, profile) in [&KK_MAJOR, &KK_MINOR].into_iter().enumerate() {
        for root in 0..12 {
            // Profile value for pitch class `pc` in key `root` is the C
            // profile at `pc - root`.
            let rotated: Vec<f64> = (0..12)
                .map(|pc| f64::from(profile[(pc + 12 - root) % 12]))
                .collect();
            scores[m][root] = pearson(mean_chroma, &rotated);
        }
    }

    let mut best = (0usize, 0usize);
    let mut best_score = f64::NEG_INFINITY;
    let mut second_score = f64::NEG_INFINITY;
    for (m, row) in scores.iter().enumerate() {
        for (root, &score) in row.iter().enumerate() {
            if score > best_score {
                second_score = best_score;
                best_score = score;
                best = (m, root);
            } else if score > second_score {
                second_score = score;
            }
        }
    }
    if !best_score.is_finite() || best_score <= 0.0 {
        return None;
    }

    let (mode_idx, root) = best;
    let mode = resolve_mode(
        mode_chroma,
        root,
        if mode_idx == 0 {
            KeyMode::Major
        } else {
            KeyMode::Minor
        },
        &scores,
    );

    // Confidence stays the template-ranking margin even after a mode flip:
    // the flip corrects a known blind spot of the profiles, it does not
    // make the winner more separable.
    let confidence = ((best_score - second_score) / best_score).clamp(0.0, 1.0) as f32;
    Some(KeyEstimate {
        root: root as u8,
        mode,
        confidence,
    })
}

/// Direct evidence for each mode at the mode-defining scale degrees.
///
/// Compares the chroma energy of the minor vs major 3rd, 6th, and 7th above
/// `root` (the 3rd weighted double — it is *the* mode-defining degree) and
/// returns `(major_evidence, minor_evidence)`.
fn mode_evidence(chroma: &[f64; 12], root: usize) -> (f64, f64) {
    let deg = |offset: usize| chroma[(root + offset) % 12];
    let mut major = 0.0;
    let mut minor = 0.0;
    for (minor_offset, major_offset, weight) in [(3, 4, 2.0), (8, 9, 1.0), (10, 11, 1.0)] {
        let diff = deg(minor_offset) - deg(major_offset);
        if diff > 0.0 {
            minor += weight * diff;
        } else {
            major += weight * -diff;
        }
    }
    (major, minor)
}

/// Flips the winning key's mode when the 3rd/6th/7th degrees clearly
/// contradict the template choice (K-K profiles separate parallel modes
/// weakly, so sparse mixes routinely land on the wrong mode).
///
/// A flip requires the opposite mode's degree evidence to dominate by
/// [`MODE_FLIP_DOMINANCE`], the 3rd itself to agree, and the flipped key's
/// own template score to stay within [`MODE_FLIP_MIN_SCORE_RATIO`] of the
/// winner.
///
/// The gates are deliberately conservative: dance music is often modal
/// (dorian minor has a "major" 6th, mixolydian major a "minor" 7th), and
/// the root's 5th harmonic leaks into the major-third bin, so mild degree
/// contradictions are expected on correctly-detected keys.
fn resolve_mode(
    chroma: &[f64; 12],
    root: usize,
    winner: KeyMode,
    scores: &[[f64; 12]; 2],
) -> KeyMode {
    let (major_evidence, minor_evidence) = mode_evidence(chroma, root);
    let minor_third_wins = chroma[(root + 3) % 12] > chroma[(root + 4) % 12];
    let flip_ok = |flipped_score: f64, winner_score: f64| {
        flipped_score > 0.0 && flipped_score >= winner_score * MODE_FLIP_MIN_SCORE_RATIO
    };
    match winner {
        KeyMode::Major
            if minor_third_wins
                && minor_evidence > major_evidence * MODE_FLIP_DOMINANCE
                && flip_ok(scores[1][root], scores[0][root]) =>
        {
            KeyMode::Minor
        }
        KeyMode::Minor
            if !minor_third_wins
                && major_evidence > minor_evidence * MODE_FLIP_DOMINANCE
                && flip_ok(scores[0][root], scores[1][root]) =>
        {
            KeyMode::Major
        }
        _ => winner,
    }
}

/// Pearson correlation between two 12-dimensional vectors.
fn pearson(a: &[f64; 12], b: &[f64]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / 12.0;
    let mean_b = b.iter().sum::<f64>() / 12.0;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = (var_a * var_b).sqrt();
    if denom <= f64::EPSILON {
        return 0.0;
    }
    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 44_100;

    /// Frequency of a MIDI note, optionally detuned by `cents`.
    fn midi_freq(note: i32, cents: f32) -> f32 {
        440.0 * 2.0f32.powf((note as f32 - 69.0 + cents / 100.0) / 12.0)
    }

    /// Appends `secs` of a chord built from the given MIDI notes, each with
    /// a few decaying harmonics.
    fn push_chord(signal: &mut Vec<f32>, notes: &[i32], secs: f32, cents: f32) {
        let len = (secs * SAMPLE_RATE as f32) as usize;
        let start = signal.len();
        signal.resize(start + len, 0.0);
        for &note in notes {
            let f0 = midi_freq(note, cents);
            for harmonic in 1..=4 {
                let freq = f0 * harmonic as f32;
                if freq >= SAMPLE_RATE as f32 / 2.0 {
                    break;
                }
                let amp = 0.2 / harmonic as f32 / notes.len() as f32;
                for (i, s) in signal[start..].iter_mut().enumerate() {
                    *s +=
                        amp * (std::f32::consts::TAU * freq * i as f32 / SAMPLE_RATE as f32).sin();
                }
            }
        }
    }

    /// I-IV-V-I progression in C major (tonic weighted by repetition).
    fn c_major_signal(cents: f32) -> Vec<f32> {
        let mut signal = Vec::new();
        push_chord(&mut signal, &[60, 64, 67], 1.0, cents); // C E G
        push_chord(&mut signal, &[53, 57, 60], 1.0, cents); // F A C
        push_chord(&mut signal, &[55, 59, 62], 1.0, cents); // G B D
        push_chord(&mut signal, &[60, 64, 67], 1.0, cents); // C E G
        signal
    }

    #[test]
    fn test_detects_c_major_progression() {
        let key = detect_key(&c_major_signal(0.0), SAMPLE_RATE).expect("key detected");
        assert_eq!((key.root, key.mode), (0, KeyMode::Major), "{}", key.name());
        assert!(key.confidence > 0.0);
    }

    #[test]
    fn test_detects_a_minor_progression() {
        let mut signal = Vec::new();
        push_chord(&mut signal, &[57, 60, 64], 1.0, 0.0); // A C E
        push_chord(&mut signal, &[50, 53, 57], 1.0, 0.0); // D F A
        push_chord(&mut signal, &[52, 55, 59], 1.0, 0.0); // E G B
        push_chord(&mut signal, &[57, 60, 64], 1.0, 0.0); // A C E
        let key = detect_key(&signal, SAMPLE_RATE).expect("key detected");
        assert_eq!((key.root, key.mode), (9, KeyMode::Minor), "{}", key.name());
    }

    #[test]
    fn test_tuning_correction_handles_detuned_track() {
        // +40 cents is nearly the worst case for hard pitch-class binning
        // without tuning correction.
        let key = detect_key(&c_major_signal(40.0), SAMPLE_RATE).expect("key detected");
        assert_eq!((key.root, key.mode), (0, KeyMode::Major), "{}", key.name());
    }

    #[test]
    fn test_silence_and_short_input_return_none() {
        assert!(detect_key(&vec![0.0; SAMPLE_RATE as usize * 2], SAMPLE_RATE).is_none());
        assert!(detect_key(&[0.1; 1024], SAMPLE_RATE).is_none());
        assert!(detect_key(&[], SAMPLE_RATE).is_none());
    }

    #[test]
    fn test_key_survives_percussion() {
        // Same C major progression with broadband bursts on a 120 BPM
        // grid at a level comparable to the chords (real percussion in
        // the 100-5000 Hz chroma band sits at or below the tonal
        // content); the key must stay readable.
        let mut signal = c_major_signal(0.0);
        let beat = (60.0 / 120.0 * SAMPLE_RATE as f32) as usize;
        let mut seed = 0x1234_5678u32;
        let len = signal.len();
        for start in (0..len).step_by(beat) {
            for s in signal[start..(start + 2048).min(len)].iter_mut() {
                // Cheap deterministic white noise (xorshift).
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *s += 0.35 * (seed as f32 / u32::MAX as f32 - 0.5);
            }
        }
        let key = detect_key(&signal, SAMPLE_RATE).expect("key detected");
        assert_eq!((key.root, key.mode), (0, KeyMode::Major), "{}", key.name());
    }

    #[test]
    fn test_mode_evidence_reads_scale_degrees() {
        // Strong minor third over C: minor evidence must dominate.
        let mut chroma = [1.0f64; 12];
        chroma[3] = 4.0; // Eb
        chroma[4] = 1.0; // E
        let (major, minor) = mode_evidence(&chroma, 0);
        assert!(minor > major * MODE_FLIP_DOMINANCE, "{major} vs {minor}");

        // Strong major third and sixth: major evidence must dominate.
        let mut chroma = [1.0f64; 12];
        chroma[4] = 4.0; // E
        chroma[9] = 2.0; // A
        let (major, minor) = mode_evidence(&chroma, 0);
        assert!(major > minor * MODE_FLIP_DOMINANCE, "{major} vs {minor}");
    }

    #[test]
    fn test_resolve_mode_flips_only_on_contradiction() {
        // Template says C major but the chroma has a dominant minor third
        // and flat sixth/seventh: flip to C minor.
        let mut chroma = [1.0f64; 12];
        chroma[0] = 3.0; // C
        chroma[3] = 2.5; // Eb
        chroma[4] = 1.0; // E
        chroma[8] = 1.8; // Ab
        chroma[10] = 1.6; // Bb
        let scores = [[0.5; 12]; 2];
        assert_eq!(
            resolve_mode(&chroma, 0, KeyMode::Major, &scores),
            KeyMode::Minor
        );

        // Same contradiction, but the minor template score is too weak:
        // no flip.
        let mut weak = [[0.5; 12]; 2];
        weak[1][0] = 0.1;
        assert_eq!(
            resolve_mode(&chroma, 0, KeyMode::Major, &weak),
            KeyMode::Major
        );

        // Major-agreeing chroma never flips a major winner.
        let mut major_chroma = [1.0f64; 12];
        major_chroma[4] = 3.0;
        assert_eq!(
            resolve_mode(&major_chroma, 0, KeyMode::Major, &scores),
            KeyMode::Major
        );
    }

    #[test]
    fn test_estimate_tuning_offset_recovers_detune() {
        // Single partial 30 cents sharp of A4.
        let semitones = [57.3f32];
        let mags = vec![vec![1.0f32]];
        let offset = estimate_tuning_offset(&mags, &semitones);
        assert!((offset - 0.3).abs() < 1e-3, "offset {offset}");
    }
}
