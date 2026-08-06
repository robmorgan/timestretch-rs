//! BPM detection accuracy harness: scores the detector against a corpus of
//! real tracks with known tempos and reports an accuracy summary.
//!
//! Reads `benchmarks/manifest.toml` and scores every `[[track]]` with a `bpm`
//! field (both reference tracks and `bpm_only = true` corpus entries). Corpus
//! audio may be `.wav`, `.mp3`, or `.aiff`, lives under `benchmarks/audio/`
//! (git-ignored), and is verified against `original_sha256` before scoring.
//!
//! Per track, the detected tempo is classified against the expected BPM:
//! EXACT (within tolerance), OCTAVE (within tolerance of 1/2x, 2x, 1/3x, or
//! 3x), WRONG, or FAILED (detection returned no tempo). The headline scores
//! are acc1 (% EXACT) and acc2 (% EXACT or OCTAVE); error percentiles are
//! octave-folded. A JSON report is written to `target/bpm_accuracy_report.json`
//! so runs can be diffed while iterating on the detector.
//!
//! Tracks may additionally carry a `beats` field pointing at a beat
//! annotation JSON (relative to `benchmarks/`, schema
//! `{"beats": [secs...], "downbeats": [secs...]}`; `downbeats` optional).
//! Annotated tracks are also scored with beat-level metrics: F-measure at
//! ±70 ms, octave-tolerant continuity (CMLt/AMLt-style), and a downbeat
//! F-measure when downbeat annotations are present.
//!
//! Tracks with a `key` field (ground-truth key in Camelot notation, e.g.
//! "3A" = Bb minor) are also scored for key detection. The detected key is
//! classified MIREX-style: EXACT, FIFTH (adjacent on the Camelot wheel,
//! same mode), RELATIVE (relative major/minor), PARALLEL (same root, other
//! mode), OTHER, or FAILED (no key detected). The weighted key score
//! credits near-misses: exact 1.0, fifth 0.5, relative 0.3, parallel 0.2.
//!
//! Run with:
//! `cargo test --features qa-harnesses --release --test bpm_accuracy -- --nocapture`
//!
//! Environment:
//! - `TIMESTRETCH_BPM_TOLERANCE`: relative tolerance (default 0.02 = +/-2%).
//! - `TIMESTRETCH_BPM_MAX_SECONDS`: trim each track before analysis.
//! - `TIMESTRETCH_STRICT_BPM_BENCHMARK=1`: missing files, hash mismatches,
//!   and skips become failures.
//! - `TIMESTRETCH_BPM_MIN_ACC1` / `TIMESTRETCH_BPM_MIN_ACC2`: minimum
//!   accuracy percentages (0-100); the test fails below the floor.
//! - `TIMESTRETCH_BPM_MIN_BEAT_F`: minimum mean beat F-measure percentage
//!   over annotated tracks; the test fails below the floor.
//! - `TIMESTRETCH_KEY_MIN_EXACT`: minimum key exact-match percentage over
//!   key-annotated tracks; the test fails below the floor.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const MANIFEST_PATH: &str = "benchmarks/manifest.toml";
const AUDIO_BASE: &str = "benchmarks/audio";
const REPORT_PATH: &str = "target/bpm_accuracy_report.json";
const STRICT_ENV_VAR: &str = "TIMESTRETCH_STRICT_BPM_BENCHMARK";
const TOLERANCE_ENV_VAR: &str = "TIMESTRETCH_BPM_TOLERANCE";
const MAX_SECONDS_ENV_VAR: &str = "TIMESTRETCH_BPM_MAX_SECONDS";
const MIN_ACC1_ENV_VAR: &str = "TIMESTRETCH_BPM_MIN_ACC1";
const MIN_ACC2_ENV_VAR: &str = "TIMESTRETCH_BPM_MIN_ACC2";
const MIN_BEAT_F_ENV_VAR: &str = "TIMESTRETCH_BPM_MIN_BEAT_F";
const MIN_KEY_EXACT_ENV_VAR: &str = "TIMESTRETCH_KEY_MIN_EXACT";
const DEFAULT_TOLERANCE: f64 = 0.02;
/// Standard beat-tracking hit tolerance for F-measure, in seconds.
const BEAT_TOLERANCE_SECS: f64 = 0.07;
/// Continuity phase/period tolerance as a fraction of the local
/// inter-annotation interval (standard CML/AML setting).
const CONTINUITY_TOLERANCE: f64 = 0.175;

// ---------------------------------------------------------------------------
// Manifest types (subset of the benchmarks/manifest.toml schema)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    track: Vec<Track>,
}

#[derive(Debug, Deserialize)]
struct Track {
    id: String,
    #[allow(dead_code)]
    description: String,
    original: String,
    #[serde(default)]
    original_sha256: Option<String>,
    bpm: f64,
    /// Optional beat annotation JSON, relative to `benchmarks/`.
    #[serde(default)]
    beats: Option<String>,
    /// Optional ground-truth key in Camelot notation (e.g. "3A").
    #[serde(default)]
    key: Option<String>,
}

/// Ground-truth beat annotation: beat (and optionally downbeat) times in
/// seconds, ascending.
#[derive(Debug, Deserialize)]
struct BeatAnnotation {
    beats: Vec<f64>,
    #[serde(default)]
    downbeats: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Report types (JSON output)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum BpmClass {
    Exact,
    Octave,
    Wrong,
    Failed,
}

impl BpmClass {
    fn label(self) -> &'static str {
        match self {
            BpmClass::Exact => "EXACT",
            BpmClass::Octave => "OCTAVE",
            BpmClass::Wrong => "WRONG",
            BpmClass::Failed => "FAILED",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum KeyClass {
    Exact,
    Fifth,
    Relative,
    Parallel,
    Other,
    Failed,
}

impl KeyClass {
    fn label(self) -> &'static str {
        match self {
            KeyClass::Exact => "EXACT",
            KeyClass::Fifth => "FIFTH",
            KeyClass::Relative => "RELATIVE",
            KeyClass::Parallel => "PARALLEL",
            KeyClass::Other => "OTHER",
            KeyClass::Failed => "FAILED",
        }
    }

    /// MIREX-style credit for near-misses.
    fn weight(self) -> f64 {
        match self {
            KeyClass::Exact => 1.0,
            KeyClass::Fifth => 0.5,
            KeyClass::Relative => 0.3,
            KeyClass::Parallel => 0.2,
            KeyClass::Other | KeyClass::Failed => 0.0,
        }
    }
}

/// A key as (root pitch class 0-11 with C = 0, is_minor).
type ParsedKey = (u8, bool);

/// Parses Camelot notation ("1A".."12B", case-insensitive) into a key.
fn parse_camelot(s: &str) -> Result<ParsedKey, String> {
    let s = s.trim().to_ascii_uppercase();
    if s.len() < 2 {
        return Err(format!("invalid Camelot key '{}'", s));
    }
    let (number_str, letter) = s.split_at(s.len() - 1);
    let minor = match letter {
        "A" => true,
        "B" => false,
        _ => return Err(format!("invalid Camelot letter in '{}'", s)),
    };
    let number: usize = number_str
        .parse()
        .map_err(|_| format!("invalid Camelot number in '{}'", s))?;
    if !(1..=12).contains(&number) {
        return Err(format!("Camelot number out of range in '{}'", s));
    }
    // Invert the wheel: position on the circle of fifths, then multiply by
    // 7 (the mod-12 inverse of 7) to recover the semitone pitch class.
    let fifth_index = (if minor { number + 7 } else { number + 4 }) % 12;
    Ok(((fifth_index * 7 % 12) as u8, minor))
}

fn classify_key(detected: Option<ParsedKey>, expected: ParsedKey) -> KeyClass {
    let Some((root, minor)) = detected else {
        return KeyClass::Failed;
    };
    let (exp_root, exp_minor) = expected;
    let interval = (12 + root - exp_root) % 12;
    match (minor == exp_minor, interval) {
        (true, 0) => KeyClass::Exact,
        (true, 5) | (true, 7) => KeyClass::Fifth,
        (false, 0) => KeyClass::Parallel,
        // Relative pair: minor root is 9 semitones above its relative
        // major (A minor / C major).
        (false, 9) if minor => KeyClass::Relative,
        (false, 3) if !minor => KeyClass::Relative,
        _ => KeyClass::Other,
    }
}

#[derive(Debug, Serialize)]
struct TrackResult {
    id: String,
    file: String,
    expected_bpm: f64,
    detected_bpm: f64,
    class: BpmClass,
    /// Octave-folded absolute relative error in percent (absent for
    /// WRONG/FAILED tracks).
    err_pct: Option<f64>,
    confidence: f32,
    duration_secs: f64,
    analysis_secs: f64,
    /// Beat F-measure at ±70 ms against the annotation (annotated tracks).
    beat_f: Option<f64>,
    /// Continuity ratio at the annotated metrical level (CMLt-style).
    beat_cmlt: Option<f64>,
    /// Continuity ratio at the best allowed metrical level (AMLt-style:
    /// annotated level, half at either phase, or double).
    beat_amlt: Option<f64>,
    /// Downbeat F-measure at ±70 ms (annotated tracks with downbeats).
    downbeat_f: Option<f64>,
    /// Mean signed beat error vs annotation, ms (positive = grid late).
    beat_offset_mean_ms: Option<f64>,
    /// Standard deviation of the signed beat error, ms (jitter).
    beat_offset_std_ms: Option<f64>,
    /// Linear drift of the signed beat error, ms per minute.
    beat_offset_drift_ms_per_min: Option<f64>,
    /// Ground-truth key in Camelot notation (key-annotated tracks).
    expected_key: Option<String>,
    /// Detected key in Camelot notation (key-annotated tracks; None when
    /// detection returned nothing).
    detected_key: Option<String>,
    /// Key classification (key-annotated tracks).
    key_class: Option<KeyClass>,
    /// Detector's key confidence (key-annotated tracks with a detection).
    key_confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
struct Summary {
    tracks: usize,
    scored: usize,
    skipped: usize,
    exact: usize,
    octave: usize,
    wrong: usize,
    failed: usize,
    acc1_pct: f64,
    acc2_pct: f64,
    median_err_pct: Option<f64>,
    mean_err_pct: Option<f64>,
    tolerance: f64,
    /// Number of tracks scored with beat annotations.
    beat_annotated: usize,
    /// Mean beat F-measure over annotated tracks, in percent.
    mean_beat_f_pct: Option<f64>,
    /// Mean CMLt-style continuity over annotated tracks, in percent.
    mean_beat_cmlt_pct: Option<f64>,
    /// Mean AMLt-style continuity over annotated tracks, in percent.
    mean_beat_amlt_pct: Option<f64>,
    /// Mean downbeat F-measure over tracks with downbeat annotations,
    /// in percent.
    mean_downbeat_f_pct: Option<f64>,
    /// Number of tracks scored against a key annotation.
    key_annotated: usize,
    key_exact: usize,
    key_fifth: usize,
    key_relative: usize,
    key_parallel: usize,
    key_other: usize,
    key_failed: usize,
    /// Percent of key-annotated tracks detected exactly.
    key_exact_pct: Option<f64>,
    /// MIREX-style weighted key score in percent (exact 1.0, fifth 0.5,
    /// relative 0.3, parallel 0.2).
    key_weighted_pct: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    summary: Summary,
    tracks: Vec<TrackResult>,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Tempo ratios that count as octave errors (MIREX-style credit).
const OCTAVE_RATIOS: [f64; 4] = [0.5, 2.0, 1.0 / 3.0, 3.0];

fn classify(detected: f64, expected: f64, tolerance: f64) -> BpmClass {
    if !detected.is_finite() || detected <= 0.0 {
        return BpmClass::Failed;
    }
    let near = |target: f64| (detected - target).abs() / target < tolerance;
    if near(expected) {
        BpmClass::Exact
    } else if OCTAVE_RATIOS.iter().any(|r| near(expected * r)) {
        BpmClass::Octave
    } else {
        BpmClass::Wrong
    }
}

/// Absolute relative error against the closest octave of the expected tempo,
/// in percent. `None` when detection failed.
fn octave_folded_err_pct(detected: f64, expected: f64) -> Option<f64> {
    if !detected.is_finite() || detected <= 0.0 {
        return None;
    }
    std::iter::once(1.0)
        .chain(OCTAVE_RATIOS)
        .map(|r| {
            let target = expected * r;
            (detected - target).abs() / target * 100.0
        })
        .min_by(|a, b| a.total_cmp(b))
}

// ---------------------------------------------------------------------------
// Beat-level metrics (times in seconds, ascending)
// ---------------------------------------------------------------------------

/// F-measure of detected beat times against annotated ones at a fixed
/// tolerance. Each annotation may be matched at most once.
fn beat_f_measure(detected: &[f64], truth: &[f64], tolerance: f64) -> f64 {
    if detected.is_empty() || truth.is_empty() {
        return 0.0;
    }
    let mut matched = vec![false; truth.len()];
    let mut hits = 0usize;
    for &d in detected {
        let idx = truth.partition_point(|&t| t < d);
        let mut best: Option<usize> = None;
        for cand in [idx.wrapping_sub(1), idx, idx + 1] {
            if cand < truth.len() && !matched[cand] {
                let dist = (truth[cand] - d).abs();
                if dist <= tolerance && best.is_none_or(|b: usize| dist < (truth[b] - d).abs()) {
                    best = Some(cand);
                }
            }
        }
        if let Some(b) = best {
            matched[b] = true;
            hits += 1;
        }
    }
    let precision = hits as f64 / detected.len() as f64;
    let recall = hits as f64 / truth.len() as f64;
    if precision + recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Continuity ratio (CML-style): the fraction of annotations hit by a
/// detection that is phase-accurate (within [`CONTINUITY_TOLERANCE`] of the
/// local inter-annotation interval) *and* whose preceding detection hit the
/// preceding annotation — beats only count inside continuous runs.
fn continuity_ratio(detected: &[f64], truth: &[f64]) -> f64 {
    if detected.len() < 2 || truth.len() < 2 {
        return 0.0;
    }
    // For each annotation, the closest detection.
    let closest: Vec<Option<usize>> = truth
        .iter()
        .map(|&t| {
            let idx = detected.partition_point(|&d| d < t);
            let mut best: Option<usize> = None;
            for cand in [idx.wrapping_sub(1), idx] {
                if cand < detected.len()
                    && best
                        .is_none_or(|b: usize| (detected[cand] - t).abs() < (detected[b] - t).abs())
                {
                    best = Some(cand);
                }
            }
            best
        })
        .collect();

    let interval_at = |j: usize| -> f64 {
        if j + 1 < truth.len() {
            truth[j + 1] - truth[j]
        } else {
            truth[j] - truth[j - 1]
        }
    };

    let hit = |j: usize| -> bool {
        match closest[j] {
            Some(d) => (detected[d] - truth[j]).abs() <= CONTINUITY_TOLERANCE * interval_at(j),
            None => false,
        }
    };

    let mut correct = 0usize;
    for j in 0..truth.len() {
        if !hit(j) {
            continue;
        }
        // Continuity: the previous annotation must also be hit, by the
        // previous detection (first annotation only needs its own hit).
        if j == 0 {
            correct += 1;
            continue;
        }
        let contiguous = hit(j - 1)
            && match (closest[j], closest[j - 1]) {
                (Some(a), Some(b)) => a == b + 1,
                _ => false,
            };
        if contiguous {
            correct += 1;
        }
    }
    correct as f64 / truth.len() as f64
}

/// Signed beat-timing diagnostics against an annotation.
///
/// For each annotated beat, takes the closest detection within half the
/// local inter-annotation interval and records the signed error
/// (detected − truth). Returns `(mean_ms, std_ms, drift_ms_per_min)` —
/// the numbers that classify a misaligned grid: constant mean with low
/// std = phase offset, near-zero mean with high std = jitter, large
/// drift = period error / segment wander. `None` with fewer than 8
/// matched beats.
fn beat_offset_stats(detected: &[f64], truth: &[f64]) -> Option<(f64, f64, f64)> {
    if detected.is_empty() || truth.len() < 2 {
        return None;
    }
    let mut errs: Vec<(f64, f64)> = Vec::new(); // (truth time, signed error) in seconds
    for (j, &t) in truth.iter().enumerate() {
        let idx = detected.partition_point(|&d| d < t);
        let mut best: Option<f64> = None;
        for cand in [idx.wrapping_sub(1), idx] {
            if cand < detected.len() {
                let e = detected[cand] - t;
                if best.is_none_or(|b: f64| e.abs() < b.abs()) {
                    best = Some(e);
                }
            }
        }
        let interval = if j + 1 < truth.len() {
            truth[j + 1] - truth[j]
        } else {
            truth[j] - truth[j - 1]
        };
        if let Some(e) = best {
            if e.abs() <= interval * 0.5 {
                errs.push((t, e));
            }
        }
    }
    if errs.len() < 8 {
        return None;
    }
    let n = errs.len() as f64;
    let mean = errs.iter().map(|(_, e)| e).sum::<f64>() / n;
    let var = errs.iter().map(|(_, e)| (e - mean).powi(2)).sum::<f64>() / n;
    let t_mean = errs.iter().map(|(t, _)| t).sum::<f64>() / n;
    let mut cov = 0.0f64;
    let mut t_var = 0.0f64;
    for (t, e) in &errs {
        cov += (t - t_mean) * (e - mean);
        t_var += (t - t_mean).powi(2);
    }
    let slope = if t_var > 0.0 { cov / t_var } else { 0.0 };
    Some((mean * 1e3, var.sqrt() * 1e3, slope * 60.0 * 1e3))
}

/// Allowed metrical-level variants of an annotation: the annotated level,
/// half density at both phases, and double density.
fn metrical_variants(truth: &[f64]) -> Vec<Vec<f64>> {
    let half_even: Vec<f64> = truth.iter().step_by(2).copied().collect();
    let half_odd: Vec<f64> = truth.iter().skip(1).step_by(2).copied().collect();
    let mut double = Vec::with_capacity(truth.len() * 2);
    for w in truth.windows(2) {
        double.push(w[0]);
        double.push((w[0] + w[1]) * 0.5);
    }
    if let Some(&last) = truth.last() {
        double.push(last);
    }
    vec![truth.to_vec(), half_even, half_odd, double]
}

/// AMLt-style continuity: best continuity ratio over the allowed levels.
fn continuity_ratio_allowed_levels(detected: &[f64], truth: &[f64]) -> f64 {
    metrical_variants(truth)
        .iter()
        .map(|variant| continuity_ratio(detected, variant))
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Audio decoding
// ---------------------------------------------------------------------------

struct DecodedAudio {
    /// Interleaved f32 samples.
    data: Vec<f32>,
    sample_rate: u32,
    channels: usize,
}

fn decode_audio(path: &Path) -> Result<DecodedAudio, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "wav" => {
            let buffer = timestretch::io::wav::read_wav_file(
                path.to_str().ok_or_else(|| "invalid path".to_string())?,
            )
            .map_err(|e| format!("WAV decode failed: {}", e))?;
            Ok(DecodedAudio {
                sample_rate: buffer.sample_rate,
                channels: buffer.channels.count(),
                data: buffer.data,
            })
        }
        "mp3" | "aiff" | "aif" => decode_with_symphonia(path),
        other => Err(format!("unsupported extension '{}'", other)),
    }
}

fn decode_with_symphonia(path: &Path) -> Result<DecodedAudio, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("unable to open {}: {}", path.display(), e))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("format probe failed: {}", e))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "no decodable audio track".to_string())?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder for codec: {}", e))?;

    let mut data: Vec<f32> = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 0usize;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // Both variants signal normal end-of-stream for these formats.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("packet read failed: {}", e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                channels = spec.channels.count();
                let frames = decoded.frames();
                let needs_alloc = sample_buf
                    .as_ref()
                    .map(|b| b.capacity() < frames * channels)
                    .unwrap_or(true);
                if needs_alloc {
                    sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
                }
                let buf = sample_buf.as_mut().unwrap();
                buf.copy_interleaved_ref(decoded);
                data.extend_from_slice(buf.samples());
            }
            // A corrupt packet (common in the wild for MP3) is not fatal.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode failed: {}", e)),
        }
    }

    if data.is_empty() || sample_rate == 0 || channels == 0 {
        return Err("no audio decoded".to_string());
    }
    Ok(DecodedAudio {
        data,
        sample_rate,
        channels,
    })
}

// ---------------------------------------------------------------------------
// Scoring pipeline
// ---------------------------------------------------------------------------

/// Decodes, downmixes, and analyzes one file, returning its scored result.
/// When `annotation` is present, the detected grid is additionally scored
/// with beat-level metrics (annotations past a trim cut are dropped).
fn score_file(
    id: &str,
    path: &Path,
    expected_bpm: f64,
    tolerance: f64,
    max_seconds: Option<f64>,
    annotation: Option<&BeatAnnotation>,
    expected_key: Option<&str>,
) -> Result<TrackResult, String> {
    let expected_key_parsed = expected_key
        .map(|s| parse_camelot(s).map(|parsed| (s.trim().to_ascii_uppercase(), parsed)))
        .transpose()?;
    let audio = decode_audio(path)?;
    let data = maybe_trim_interleaved(&audio.data, audio.sample_rate, audio.channels, max_seconds);
    let frames = data.len() / audio.channels.max(1);
    let duration_secs = frames as f64 / audio.sample_rate as f64;

    let mid = timestretch::downmix_to_mid(&data, audio.channels);
    let (artifact, report) = timestretch::analyze_for_dj_with_report(&mid, audio.sample_rate);

    let class = classify(artifact.bpm, expected_bpm, tolerance);
    let err_pct = match class {
        BpmClass::Exact | BpmClass::Octave => octave_folded_err_pct(artifact.bpm, expected_bpm),
        BpmClass::Wrong | BpmClass::Failed => None,
    };

    let (beat_f, beat_cmlt, beat_amlt, downbeat_f, offset_stats) = match annotation {
        Some(ann) => {
            let secs = 1.0 / audio.sample_rate as f64;
            let detected: Vec<f64> = artifact
                .beat_positions_fractional
                .iter()
                .map(|&p| p * secs)
                .collect();
            let truth: Vec<f64> = ann
                .beats
                .iter()
                .copied()
                .filter(|&t| t <= duration_secs)
                .collect();
            let f = beat_f_measure(&detected, &truth, BEAT_TOLERANCE_SECS);
            let cmlt = continuity_ratio(&detected, &truth);
            let amlt = continuity_ratio_allowed_levels(&detected, &truth);
            let offset_stats = beat_offset_stats(&detected, &truth);
            let downbeat_f = if ann.downbeats.is_empty() {
                None
            } else {
                let detected_db: Vec<f64> = artifact
                    .downbeat_beat_indices
                    .iter()
                    .filter_map(|&i| artifact.beat_positions_fractional.get(i))
                    .map(|&p| p * secs)
                    .collect();
                let truth_db: Vec<f64> = ann
                    .downbeats
                    .iter()
                    .copied()
                    .filter(|&t| t <= duration_secs)
                    .collect();
                Some(beat_f_measure(&detected_db, &truth_db, BEAT_TOLERANCE_SECS))
            };
            (Some(f), Some(cmlt), Some(amlt), downbeat_f, offset_stats)
        }
        None => (None, None, None, None, None),
    };

    let (expected_key, detected_key, key_class, key_confidence) = match expected_key_parsed {
        Some((camelot, parsed)) => {
            let detected = artifact
                .key
                .map(|k| (k.root, k.mode == timestretch::KeyMode::Minor));
            (
                Some(camelot),
                artifact.key.map(|k| k.camelot()),
                Some(classify_key(detected, parsed)),
                artifact.key.map(|k| k.confidence),
            )
        }
        None => (None, None, None, None),
    };

    Ok(TrackResult {
        id: id.to_string(),
        file: path.display().to_string(),
        expected_bpm,
        detected_bpm: artifact.bpm,
        class,
        err_pct,
        confidence: artifact.confidence,
        duration_secs,
        analysis_secs: report.analysis_elapsed_secs,
        beat_f,
        beat_cmlt,
        beat_amlt,
        downbeat_f,
        beat_offset_mean_ms: offset_stats.map(|(mean, _, _)| mean),
        beat_offset_std_ms: offset_stats.map(|(_, std, _)| std),
        beat_offset_drift_ms_per_min: offset_stats.map(|(_, _, drift)| drift),
        expected_key,
        detected_key,
        key_class,
        key_confidence,
    })
}

fn print_metric(result: &TrackResult) {
    let fmt_ratio = |v: Option<f64>| {
        v.map(|v| format!("{:.3}", v))
            .unwrap_or_else(|| "n/a".to_string())
    };
    println!(
        "METRIC track=\"{}\" expected={:.1} detected={:.2} err_pct={} class={} \
         confidence={:.3} duration_secs={:.1} analysis_realtime_factor={:.1} \
         beat_f={} beat_cmlt={} beat_amlt={} downbeat_f={}",
        result.id,
        result.expected_bpm,
        result.detected_bpm,
        result
            .err_pct
            .map(|e| format!("{:.2}", e))
            .unwrap_or_else(|| "n/a".to_string()),
        result.class.label(),
        result.confidence,
        result.duration_secs,
        result.duration_secs / result.analysis_secs.max(1e-9),
        fmt_ratio(result.beat_f),
        fmt_ratio(result.beat_cmlt),
        fmt_ratio(result.beat_amlt),
        fmt_ratio(result.downbeat_f),
    );
    if let (Some(mean), Some(std), Some(drift)) = (
        result.beat_offset_mean_ms,
        result.beat_offset_std_ms,
        result.beat_offset_drift_ms_per_min,
    ) {
        println!(
            "METRIC track=\"{}\" beat_offset_mean_ms={:.1} beat_offset_std_ms={:.1} \
             beat_offset_drift_ms_per_min={:.2}",
            result.id, mean, std, drift,
        );
    }
    if let Some(class) = result.key_class {
        println!(
            "METRIC track=\"{}\" expected_key={} detected_key={} key_class={} key_confidence={}",
            result.id,
            result.expected_key.as_deref().unwrap_or("n/a"),
            result.detected_key.as_deref().unwrap_or("none"),
            class.label(),
            result
                .key_confidence
                .map(|c| format!("{:.3}", c))
                .unwrap_or_else(|| "n/a".to_string()),
        );
    }
}

fn summarize(results: &[TrackResult], skipped: usize, tolerance: f64) -> Summary {
    let count = |class: BpmClass| results.iter().filter(|r| r.class == class).count();
    let exact = count(BpmClass::Exact);
    let octave = count(BpmClass::Octave);
    let scored = results.len();
    let pct = |n: usize| {
        if scored == 0 {
            0.0
        } else {
            n as f64 / scored as f64 * 100.0
        }
    };

    let mut errs: Vec<f64> = results.iter().filter_map(|r| r.err_pct).collect();
    errs.sort_by(|a, b| a.total_cmp(b));
    let median_err_pct = if errs.is_empty() {
        None
    } else if errs.len() % 2 == 1 {
        Some(errs[errs.len() / 2])
    } else {
        Some((errs[errs.len() / 2 - 1] + errs[errs.len() / 2]) / 2.0)
    };
    let mean_err_pct = if errs.is_empty() {
        None
    } else {
        Some(errs.iter().sum::<f64>() / errs.len() as f64)
    };

    let mean_pct = |values: Vec<f64>| -> Option<f64> {
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64 * 100.0)
        }
    };
    let beat_annotated = results.iter().filter(|r| r.beat_f.is_some()).count();

    let key_count = |class: KeyClass| {
        results
            .iter()
            .filter(|r| r.key_class == Some(class))
            .count()
    };
    let key_annotated = results.iter().filter(|r| r.key_class.is_some()).count();
    let key_exact = key_count(KeyClass::Exact);
    let key_weighted: f64 = results
        .iter()
        .filter_map(|r| r.key_class)
        .map(KeyClass::weight)
        .sum();
    let key_pct = |v: f64| {
        if key_annotated == 0 {
            None
        } else {
            Some(v / key_annotated as f64 * 100.0)
        }
    };

    Summary {
        tracks: scored + skipped,
        scored,
        skipped,
        exact,
        octave,
        wrong: count(BpmClass::Wrong),
        failed: count(BpmClass::Failed),
        acc1_pct: pct(exact),
        acc2_pct: pct(exact + octave),
        median_err_pct,
        mean_err_pct,
        tolerance,
        beat_annotated,
        mean_beat_f_pct: mean_pct(results.iter().filter_map(|r| r.beat_f).collect()),
        mean_beat_cmlt_pct: mean_pct(results.iter().filter_map(|r| r.beat_cmlt).collect()),
        mean_beat_amlt_pct: mean_pct(results.iter().filter_map(|r| r.beat_amlt).collect()),
        mean_downbeat_f_pct: mean_pct(results.iter().filter_map(|r| r.downbeat_f).collect()),
        key_annotated,
        key_exact,
        key_fifth: key_count(KeyClass::Fifth),
        key_relative: key_count(KeyClass::Relative),
        key_parallel: key_count(KeyClass::Parallel),
        key_other: key_count(KeyClass::Other),
        key_failed: key_count(KeyClass::Failed),
        key_exact_pct: key_pct(key_exact as f64),
        key_weighted_pct: key_pct(key_weighted),
    }
}

fn print_summary(summary: &Summary) {
    let fmt_opt = |v: Option<f64>| {
        v.map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "n/a".to_string())
    };
    println!(
        "SUMMARY tracks={} scored={} skipped={} acc1={:.1}% acc2={:.1}% \
         median_err={}% mean_err={}% exact={} octave={} wrong={} failed={} tolerance={:.1}% \
         beat_annotated={} beat_f={}% beat_cmlt={}% beat_amlt={}% downbeat_f={}%",
        summary.tracks,
        summary.scored,
        summary.skipped,
        summary.acc1_pct,
        summary.acc2_pct,
        fmt_opt(summary.median_err_pct),
        fmt_opt(summary.mean_err_pct),
        summary.exact,
        summary.octave,
        summary.wrong,
        summary.failed,
        summary.tolerance * 100.0,
        summary.beat_annotated,
        fmt_opt(summary.mean_beat_f_pct),
        fmt_opt(summary.mean_beat_cmlt_pct),
        fmt_opt(summary.mean_beat_amlt_pct),
        fmt_opt(summary.mean_downbeat_f_pct),
    );
    if summary.key_annotated > 0 {
        println!(
            "SUMMARY key_annotated={} key_exact={}% key_weighted={}% \
             exact={} fifth={} relative={} parallel={} other={} failed={}",
            summary.key_annotated,
            fmt_opt(summary.key_exact_pct),
            fmt_opt(summary.key_weighted_pct),
            summary.key_exact,
            summary.key_fifth,
            summary.key_relative,
            summary.key_parallel,
            summary.key_other,
            summary.key_failed,
        );
    }
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

fn strict_benchmark_mode() -> bool {
    let value = std::env::var(STRICT_ENV_VAR).unwrap_or_default();
    let normalized = value.trim().to_ascii_lowercase();
    !normalized.is_empty() && normalized != "0" && normalized != "false" && normalized != "no"
}

fn env_f64(var: &str) -> Option<f64> {
    let value = std::env::var(var).ok()?;
    let parsed = value.trim().parse::<f64>().ok()?;
    (parsed.is_finite() && parsed > 0.0).then_some(parsed)
}

fn tolerance() -> f64 {
    env_f64(TOLERANCE_ENV_VAR).unwrap_or(DEFAULT_TOLERANCE)
}

fn maybe_trim_interleaved(
    data: &[f32],
    sample_rate: u32,
    channels: usize,
    max_seconds: Option<f64>,
) -> Vec<f32> {
    let Some(max_seconds) = max_seconds else {
        return data.to_vec();
    };
    let max_frames = (sample_rate as f64 * max_seconds).round() as usize;
    let max_samples = max_frames.saturating_mul(channels);
    let keep = data.len().min(max_samples);
    data[..keep].to_vec()
}

// ---------------------------------------------------------------------------
// Manifest path/hash helpers (same conventions as qa/reference_quality.rs)
// ---------------------------------------------------------------------------

fn resolve_audio_path(audio_base: &Path, configured: &str) -> Result<PathBuf, String> {
    let configured = configured.trim();
    if configured.is_empty() {
        return Err("empty path".to_string());
    }

    let relative = configured
        .strip_prefix("benchmarks/audio/")
        .unwrap_or(configured);
    if relative.starts_with("audio/") {
        return Err(format!(
            "path '{}' includes 'audio/' prefix; paths must be relative to benchmarks/audio/",
            configured
        ));
    }

    let rel_path = Path::new(relative);
    if rel_path.is_absolute() {
        return Err(format!("absolute path '{}' is not allowed", configured));
    }
    if rel_path
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(format!(
            "path '{}' contains parent traversal ('..'), which is not allowed",
            configured
        ));
    }

    Ok(audio_base.join(rel_path))
}

/// Loads a track's beat annotation JSON (path relative to `benchmarks/`,
/// no absolute paths or parent traversal). `Ok(None)` when the track has
/// no annotation configured.
fn load_annotation(track: &Track) -> Result<Option<BeatAnnotation>, String> {
    let Some(configured) = track.beats.as_deref() else {
        return Ok(None);
    };
    let rel = Path::new(configured.trim());
    if rel.as_os_str().is_empty() {
        return Err("empty beats annotation path".to_string());
    }
    if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!(
            "beats annotation path '{}' must be relative to benchmarks/ without traversal",
            configured
        ));
    }
    let path = Path::new("benchmarks").join(rel);
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("unable to read annotation {}: {}", path.display(), e))?;
    let mut ann: BeatAnnotation = serde_json::from_str(&json)
        .map_err(|e| format!("invalid annotation {}: {}", path.display(), e))?;
    if ann.beats.is_empty() {
        return Err(format!("annotation {} has no beats", path.display()));
    }
    ann.beats.sort_by(|a, b| a.total_cmp(b));
    ann.downbeats.sort_by(|a, b| a.total_cmp(b));
    Ok(Some(ann))
}

fn validate_sha256(
    file_path: &Path,
    expected_sha256: Option<&str>,
    label: &str,
    strict: bool,
) -> Result<(), String> {
    let Some(expected_sha256) = expected_sha256 else {
        if strict {
            return Err(format!(
                "{} is missing required SHA-256 in strict mode",
                label
            ));
        }
        return Ok(());
    };

    let expected = expected_sha256.trim().to_ascii_lowercase();
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "{} has invalid SHA-256 '{}' in manifest",
            label, expected_sha256
        ));
    }

    let actual = compute_sha256(file_path)
        .map_err(|msg| format!("{} checksum calculation failed: {}", label, msg))?;
    if actual != expected {
        return Err(format!(
            "{} checksum mismatch: expected {}, got {} ({})",
            label,
            expected,
            actual,
            file_path.display()
        ));
    }
    Ok(())
}

fn compute_sha256(file_path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(file_path)
        .map_err(|err| format!("unable to open {}: {}", file_path.display(), err))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];

    loop {
        let n = file
            .read(&mut buf)
            .map_err(|err| format!("unable to read {}: {}", file_path.display(), err))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn bpm_accuracy() {
    let strict = strict_benchmark_mode();
    let tolerance = tolerance();
    let max_seconds = env_f64(MAX_SECONDS_ENV_VAR);

    let manifest_path = Path::new(MANIFEST_PATH);
    if !manifest_path.exists() {
        if strict {
            panic!("{} not found in strict mode", MANIFEST_PATH);
        }
        println!(
            "{} not found, skipping BPM accuracy benchmark",
            MANIFEST_PATH
        );
        return;
    }
    let manifest_str = std::fs::read_to_string(manifest_path).expect("Failed to read manifest");
    let manifest: Manifest = toml::from_str(&manifest_str).expect("Failed to parse manifest");

    let audio_base = Path::new(AUDIO_BASE);
    let mut results: Vec<TrackResult> = Vec::new();
    let mut skipped = 0usize;
    let skip = |id: &str, msg: String, skipped: &mut usize| {
        if strict {
            panic!("Track '{}': {}", id, msg);
        }
        println!("Skipping track '{}': {}", id, msg);
        *skipped += 1;
    };

    for track in &manifest.track {
        if !track.bpm.is_finite() || track.bpm <= 0.0 {
            skip(
                &track.id,
                format!("invalid expected BPM {}", track.bpm),
                &mut skipped,
            );
            continue;
        }
        let path = match resolve_audio_path(audio_base, &track.original) {
            Ok(path) => path,
            Err(msg) => {
                skip(&track.id, msg, &mut skipped);
                continue;
            }
        };
        if !path.exists() {
            skip(
                &track.id,
                format!("audio file not found ({})", path.display()),
                &mut skipped,
            );
            continue;
        }
        if let Err(msg) = validate_sha256(
            &path,
            track.original_sha256.as_deref(),
            &format!("track '{}' original", track.id),
            strict,
        ) {
            // A present-but-wrong hash is always a failure: the file is not
            // the audio the expected BPM was measured against.
            panic!("{}", msg);
        }
        let annotation = match load_annotation(track) {
            Ok(ann) => ann,
            Err(msg) => {
                skip(&track.id, msg, &mut skipped);
                continue;
            }
        };
        match score_file(
            &track.id,
            &path,
            track.bpm,
            tolerance,
            max_seconds,
            annotation.as_ref(),
            track.key.as_deref(),
        ) {
            Ok(result) => {
                print_metric(&result);
                results.push(result);
            }
            Err(msg) => skip(&track.id, msg, &mut skipped),
        }
    }

    if results.is_empty() && skipped == 0 {
        if strict {
            panic!("no scorable tracks in {} in strict mode", MANIFEST_PATH);
        }
        println!(
            "No tracks with a bpm field in {}, nothing to score",
            MANIFEST_PATH
        );
        return;
    }

    let summary = summarize(&results, skipped, tolerance);
    print_summary(&summary);

    let report = Report {
        summary,
        tracks: results,
    };
    let json = serde_json::to_string_pretty(&report).expect("Failed to serialize report");
    std::fs::create_dir_all("target").expect("Failed to create target dir");
    std::fs::write(REPORT_PATH, &json).expect("Failed to write report JSON");
    println!("JSON report written to: {}", REPORT_PATH);

    if strict {
        assert_eq!(
            report.summary.skipped, 0,
            "Strict mode does not allow skipped tracks"
        );
    }
    if let Some(min_acc1) = env_f64(MIN_ACC1_ENV_VAR) {
        assert!(
            report.summary.acc1_pct >= min_acc1,
            "acc1 {:.1}% below required minimum {:.1}%",
            report.summary.acc1_pct,
            min_acc1
        );
    }
    if let Some(min_acc2) = env_f64(MIN_ACC2_ENV_VAR) {
        assert!(
            report.summary.acc2_pct >= min_acc2,
            "acc2 {:.1}% below required minimum {:.1}%",
            report.summary.acc2_pct,
            min_acc2
        );
    }
    if let Some(min_beat_f) = env_f64(MIN_BEAT_F_ENV_VAR) {
        let mean_beat_f = report
            .summary
            .mean_beat_f_pct
            .expect("beat F floor set but no annotated tracks were scored");
        assert!(
            mean_beat_f >= min_beat_f,
            "mean beat F {:.1}% below required minimum {:.1}%",
            mean_beat_f,
            min_beat_f
        );
    }
    if let Some(min_key_exact) = env_f64(MIN_KEY_EXACT_ENV_VAR) {
        let key_exact_pct = report
            .summary
            .key_exact_pct
            .expect("key exact floor set but no key-annotated tracks were scored");
        assert!(
            key_exact_pct >= min_key_exact,
            "key exact {:.1}% below required minimum {:.1}%",
            key_exact_pct,
            min_key_exact
        );
    }
}

/// ROADMAP Stage 10 variable-tempo evidence: a synthetic 120 → 132 BPM
/// linear ramp (kick thump + beater click; beat times recorded during
/// synthesis, so ground truth is exact by construction) must be tracked
/// by the SHIPPING analysis path — `analyze_for_dj`, which includes both
/// rigid-grid adoption gates. This doubles as the end-to-end guard that
/// no adoption gate flattens a ramp onto a constant grid (`phase_lock`
/// reads a deceptive ~0.77 on this class; the smeared sanity floor is
/// what rejects it at high lock, and the corroboration agreement — 0.25
/// here — guards the low-lock entry).
#[test]
fn tempo_ramp_fixture_tracks_within_tolerance() {
    let sr = 44_100u32;
    let srf = sr as f64;
    let seconds = 45.0;
    let len = (srf * seconds) as usize;
    let mut audio = vec![0.0f32; len];
    let mut truth: Vec<f64> = Vec::new();
    let mut pos = 0.0f64;
    while (pos as usize) < len {
        let at = pos as usize;
        truth.push(pos / srf);
        for i in 0..3000.min(len - at) {
            let t = i as f64 / srf;
            audio[at + i] +=
                (0.9 * (-t * 40.0).exp() * (2.0 * std::f64::consts::PI * 60.0 * t).sin()
                    + 0.5 * (-t * 400.0).exp() * (2.0 * std::f64::consts::PI * 3000.0 * t).sin())
                    as f32;
        }
        let frac = pos / len as f64;
        let bpm = 120.0 + 12.0 * frac;
        pos += 60.0 * srf / bpm;
    }

    let artifact = timestretch::analyze_for_dj(&audio, sr);
    let detected: Vec<f64> = artifact
        .beat_positions_fractional
        .iter()
        .map(|&p| p / srf)
        .collect();

    // Score the interior (clear of analysis warm-up and the tail).
    let interior: Vec<f64> = truth
        .iter()
        .copied()
        .filter(|&t| t > 2.0 && t < seconds - 2.0)
        .collect();
    let f = beat_f_measure(&detected, &interior, BEAT_TOLERANCE_SECS);
    let continuity = continuity_ratio(&detected, &interior);
    println!("tempo ramp: beat F {f:.3}, continuity {continuity:.3}");
    assert!(
        f >= 0.85,
        "tempo ramp under-tracked: beat F {f:.3} (Stage 10 floor 0.85)"
    );
    assert!(
        continuity >= 0.75,
        "tempo ramp tracking fragmented: continuity {continuity:.3}"
    );

    // The detected grid must actually FOLLOW the ramp — a constant grid
    // (wrongly adopted rigid fit) scores near-zero F here anyway, but
    // assert the tempo trend directly so the failure reads clearly.
    let mean_interval = |beats: &[f64]| -> f64 {
        let iv: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        iv.iter().sum::<f64>() / iv.len().max(1) as f64
    };
    let early: Vec<f64> = detected.iter().copied().filter(|&t| t < 10.0).collect();
    let late: Vec<f64> = detected
        .iter()
        .copied()
        .filter(|&t| t > seconds - 10.0)
        .collect();
    let (e, l) = (mean_interval(&early), mean_interval(&late));
    assert!(
        l < e * 0.97,
        "detected grid does not follow the ramp: early interval {e:.4}s vs late {l:.4}s"
    );
}

/// End-to-end smoke test on checked-in fixtures so the pipeline is exercised
/// even when no benchmark corpus is present locally.
#[test]
fn bpm_accuracy_self_test() {
    for (file, expected) in [
        ("test_audio/click_train_128bpm.wav", 128.0),
        ("test_audio/kick_pattern_128bpm.wav", 128.0),
    ] {
        let path = Path::new(file);
        assert!(path.exists(), "checked-in fixture {} missing", file);
        // Synthetic ground truth: the fixtures are exact 128 BPM grids
        // from sample 0, so annotate beats every 60/128 s for the first
        // few seconds and exercise the beat-metric path end to end.
        let interval = 60.0 / expected;
        let annotation = BeatAnnotation {
            beats: (0..8).map(|k| k as f64 * interval).collect(),
            downbeats: Vec::new(),
        };
        let result = score_file(
            "self-test",
            path,
            expected,
            DEFAULT_TOLERANCE,
            None,
            Some(&annotation),
            None,
        )
        .unwrap_or_else(|e| panic!("{}: {}", file, e));
        print_metric(&result);
        assert_eq!(
            result.class,
            BpmClass::Exact,
            "{}: detected {:.2} BPM, expected {:.0} within {:.0}%",
            file,
            result.detected_bpm,
            expected,
            DEFAULT_TOLERANCE * 100.0
        );
        let beat_f = result.beat_f.expect("annotated self-test scores beat F");
        assert!(
            beat_f > 0.8,
            "{}: beat F-measure {:.3} too low against exact grid",
            file,
            beat_f
        );
    }
}

#[test]
fn classification() {
    let tol = 0.02;
    assert_eq!(classify(128.0, 128.0, tol), BpmClass::Exact);
    assert_eq!(classify(126.0, 128.0, tol), BpmClass::Exact); // -1.6%
    assert_eq!(classify(64.0, 128.0, tol), BpmClass::Octave); // half
    assert_eq!(classify(256.0, 128.0, tol), BpmClass::Octave); // double
    assert_eq!(classify(140.0 / 3.0, 140.0, tol), BpmClass::Octave); // third
    assert_eq!(classify(420.0, 140.0, tol), BpmClass::Octave); // triple
    assert_eq!(classify(120.0, 128.0, tol), BpmClass::Wrong); // -6.3%
    assert_eq!(classify(0.0, 128.0, tol), BpmClass::Failed);
    assert_eq!(classify(f64::NAN, 128.0, tol), BpmClass::Failed);
    assert_eq!(classify(-1.0, 128.0, tol), BpmClass::Failed);

    // Octave-folded error folds against the nearest octave.
    let err = octave_folded_err_pct(63.5, 128.0).unwrap();
    assert!((err - (0.5 / 64.0 * 100.0)).abs() < 1e-9, "err={}", err);
    assert!(octave_folded_err_pct(128.0, 128.0).unwrap() < 1e-12);
    assert_eq!(octave_folded_err_pct(0.0, 128.0), None);
}

#[test]
fn camelot_parsing() {
    // Wheel spot checks: (camelot, pitch class with C = 0, is_minor).
    for (s, root, minor) in [
        ("8B", 0, false),  // C major
        ("8A", 9, true),   // A minor
        ("1B", 11, false), // B major
        ("1A", 8, true),   // G# minor
        ("3A", 10, true),  // Bb minor
        ("6A", 7, true),   // G minor
        ("7A", 2, true),   // D minor
        ("11A", 6, true),  // F# minor
        ("7B", 5, false),  // F major
        ("12b", 4, false), // E major, case-insensitive
    ] {
        assert_eq!(parse_camelot(s).unwrap(), (root, minor), "camelot {}", s);
    }
    assert!(parse_camelot("0A").is_err());
    assert!(parse_camelot("13B").is_err());
    assert!(parse_camelot("8C").is_err());
    assert!(parse_camelot("").is_err());
}

#[test]
fn key_classification() {
    let bbm = parse_camelot("3A").unwrap();
    assert_eq!(classify_key(Some(bbm), bbm), KeyClass::Exact);
    // Fifth up/down on the wheel, same mode: Fm (4A) and Ebm (2A).
    assert_eq!(
        classify_key(Some(parse_camelot("4A").unwrap()), bbm),
        KeyClass::Fifth
    );
    assert_eq!(
        classify_key(Some(parse_camelot("2A").unwrap()), bbm),
        KeyClass::Fifth
    );
    // Relative major of Bbm is Db major (3B).
    assert_eq!(
        classify_key(Some(parse_camelot("3B").unwrap()), bbm),
        KeyClass::Relative
    );
    // Parallel: Bb major (6B).
    assert_eq!(
        classify_key(Some(parse_camelot("6B").unwrap()), bbm),
        KeyClass::Parallel
    );
    // Relative seen from the major side: C major detected as A minor.
    let c_major = parse_camelot("8B").unwrap();
    assert_eq!(
        classify_key(Some(parse_camelot("8A").unwrap()), c_major),
        KeyClass::Relative
    );
    // Unrelated key and failed detection.
    assert_eq!(
        classify_key(Some(parse_camelot("9B").unwrap()), bbm),
        KeyClass::Other
    );
    assert_eq!(classify_key(None, bbm), KeyClass::Failed);
}

#[test]
fn beat_offset_stats_classify_failure_modes() {
    let truth: Vec<f64> = (0..64).map(|k| k as f64 * 0.5).collect();

    // Perfect detection: everything ~0.
    let (mean, std, drift) = beat_offset_stats(&truth, &truth).unwrap();
    assert!(mean.abs() < 1e-9 && std < 1e-9 && drift.abs() < 1e-9);

    // Constant +30 ms: offset shows in the mean, not std or drift.
    let late: Vec<f64> = truth.iter().map(|&t| t + 0.030).collect();
    let (mean, std, drift) = beat_offset_stats(&late, &truth).unwrap();
    assert!((mean - 30.0).abs() < 1e-6, "mean {mean}");
    assert!(std < 1e-6 && drift.abs() < 1e-6);

    // Alternating ±20 ms jitter: near-zero mean, ~20 ms std.
    let jitter: Vec<f64> = truth
        .iter()
        .enumerate()
        .map(|(i, &t)| t + if i % 2 == 0 { 0.020 } else { -0.020 })
        .collect();
    let (mean, std, _) = beat_offset_stats(&jitter, &truth).unwrap();
    assert!(mean.abs() < 1e-6, "mean {mean}");
    assert!((std - 20.0).abs() < 1e-6, "std {std}");

    // 0.1% period error: drift dominates. err(t) = 0.001*t, slope
    // 0.001 s/s = 60 ms/min.
    let stretched: Vec<f64> = truth.iter().map(|&t| t * 1.001).collect();
    let (_, _, drift) = beat_offset_stats(&stretched, &truth).unwrap();
    assert!((drift - 60.0).abs() < 1.0, "drift {drift}");

    // Too few beats: no stats.
    assert!(beat_offset_stats(&truth[..4], &truth[..4]).is_none());
}

#[test]
fn beat_metrics() {
    let truth: Vec<f64> = (0..16).map(|k| k as f64 * 0.5).collect();

    // Perfect detection: all metrics 1.0.
    assert!((beat_f_measure(&truth, &truth, BEAT_TOLERANCE_SECS) - 1.0).abs() < 1e-12);
    assert!((continuity_ratio(&truth, &truth) - 1.0).abs() < 1e-12);
    assert!((continuity_ratio_allowed_levels(&truth, &truth) - 1.0).abs() < 1e-12);

    // Small constant offset within tolerance: still perfect F.
    let offset: Vec<f64> = truth.iter().map(|&t| t + 0.03).collect();
    assert!((beat_f_measure(&offset, &truth, BEAT_TOLERANCE_SECS) - 1.0).abs() < 1e-12);

    // Offset beyond tolerance: F collapses.
    let far: Vec<f64> = truth.iter().map(|&t| t + 0.2).collect();
    assert!(beat_f_measure(&far, &truth, BEAT_TOLERANCE_SECS) < 0.2);

    // Half-tempo detection (every other truth beat): CMLt low, AMLt high.
    let half: Vec<f64> = truth.iter().step_by(2).copied().collect();
    assert!(continuity_ratio(&half, &truth) < 0.6);
    assert!(continuity_ratio_allowed_levels(&half, &truth) > 0.9);

    // Offbeat half-tempo (odd phase) is also an allowed level.
    let half_odd: Vec<f64> = truth.iter().skip(1).step_by(2).copied().collect();
    assert!(continuity_ratio_allowed_levels(&half_odd, &truth) > 0.9);

    // A gap in the detections breaks continuity for the beat after the gap.
    let mut gapped = truth.clone();
    gapped.remove(8);
    let cont = continuity_ratio(&gapped, &truth);
    assert!(
        cont < 0.95 && cont > 0.7,
        "gap should break continuity locally, got {}",
        cont
    );

    // Degenerate inputs.
    assert_eq!(beat_f_measure(&[], &truth, BEAT_TOLERANCE_SECS), 0.0);
    assert_eq!(beat_f_measure(&truth, &[], BEAT_TOLERANCE_SECS), 0.0);
    assert_eq!(continuity_ratio(&[], &truth), 0.0);
}
