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

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
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
const DEFAULT_TOLERANCE: f64 = 0.02;

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
fn score_file(
    id: &str,
    path: &Path,
    expected_bpm: f64,
    tolerance: f64,
    max_seconds: Option<f64>,
) -> Result<TrackResult, String> {
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
    })
}

fn print_metric(result: &TrackResult) {
    println!(
        "METRIC track=\"{}\" expected={:.1} detected={:.2} err_pct={} class={} \
         confidence={:.3} duration_secs={:.1} analysis_realtime_factor={:.1}",
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
    );
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
    }
}

fn print_summary(summary: &Summary) {
    let fmt_opt = |v: Option<f64>| {
        v.map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "n/a".to_string())
    };
    println!(
        "SUMMARY tracks={} scored={} skipped={} acc1={:.1}% acc2={:.1}% \
         median_err={}% mean_err={}% exact={} octave={} wrong={} failed={} tolerance={:.1}%",
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
    );
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
        match score_file(&track.id, &path, track.bpm, tolerance, max_seconds) {
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
        let result = score_file("self-test", path, expected, DEFAULT_TOLERANCE, None)
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
