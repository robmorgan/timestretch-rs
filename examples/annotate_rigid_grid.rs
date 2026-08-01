//! Semi-automatic ground-truth beat annotator for the BPM benchmark corpus.
//!
//! For every manifest track with a confirmed `bpm` (> 0), this fits a *rigid*
//! beat grid — constant tempo locked to the human-verified BPM — by maximizing
//! low-band onset energy at the grid points, then picks the downbeat phase
//! (of 4) by low-band accent. It writes `benchmarks/annotations/<id>.json` in
//! the schema the `bpm_accuracy` harness consumes:
//!
//! ```json
//! {"beats": [secs...], "downbeats": [secs...], "meta": {...}}
//! ```
//!
//! The method is deliberately independent of the production detector (no
//! tempogram, no Viterbi, no DP beat tracking): the only shared assumption is
//! the manifest BPM, which was verified with independent estimators. It is
//! valid for DAW-produced constant-tempo material only — do not use it on
//! rubato/live recordings.
//!
//! The `phase_lock` and `downbeat_margin` numbers in `meta` say how decisive
//! the fit was; low values mean "verify by ear before trusting". Use
//! `--clicks <dir>` to render 24 s click-overlay WAV excerpts (high tick =
//! annotated downbeat, low tick = other beats) for quick ear verification.
//!
//! Usage (from the repo root):
//!
//! ```text
//! cargo run --release --example annotate_rigid_grid -- [--clicks <dir>] [id ...]
//! ```
//!
//! With no ids, all manifest tracks with `bpm` > 0 are annotated.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const MANIFEST_PATH: &str = "benchmarks/manifest.toml";
const AUDIO_BASE: &str = "benchmarks/audio";
const ANNOTATIONS_DIR: &str = "benchmarks/annotations";

/// Onset-envelope hop as a fraction of the sample rate (~5 ms).
const HOP_SECS: f64 = 0.005;
/// Low-pass corner for the kick band, Hz (two cascaded biquads).
const KICK_BAND_HZ: f64 = 150.0;
/// BPM search half-width around the manifest value, as a fraction.
/// Covers label-vs-render discrepancies (e.g. a 124.0-labelled vinyl-era
/// master that actually plays at 124.2). A rigid true grid dominates the
/// score, so a generous window cannot pull the fit off-tempo.
const BPM_SEARCH_FRAC: f64 = 0.005;
/// Coarse search resolution.
const BPM_STEPS: usize = 201;
const PHASE_STEPS: usize = 256;
/// Active-region gate relative to the loudest 1 s low-band RMS window:
/// leading/trailing regions quieter than this carry no annotated beats.
const ACTIVE_GATE: f32 = 0.05;

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    track: Vec<TrackEntry>,
}

#[derive(Debug, Deserialize)]
struct TrackEntry {
    id: String,
    original: String,
    #[serde(default)]
    bpm: f64,
}

struct DecodedAudio {
    data: Vec<f32>,
    sample_rate: u32,
    channels: usize,
}

fn main() {
    let mut ids: Vec<String> = Vec::new();
    let mut clicks_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--clicks" {
            let dir = args.next().unwrap_or_else(|| {
                eprintln!("--clicks requires a directory argument");
                std::process::exit(2);
            });
            clicks_dir = Some(PathBuf::from(dir));
        } else {
            ids.push(arg);
        }
    }

    let manifest_text = std::fs::read_to_string(MANIFEST_PATH).unwrap_or_else(|e| {
        eprintln!("cannot read {MANIFEST_PATH}: {e} (run from the repo root)");
        std::process::exit(2);
    });
    let manifest: Manifest = toml::from_str(&manifest_text).unwrap_or_else(|e| {
        eprintln!("cannot parse {MANIFEST_PATH}: {e}");
        std::process::exit(2);
    });

    std::fs::create_dir_all(ANNOTATIONS_DIR).expect("create annotations dir");
    if let Some(dir) = &clicks_dir {
        std::fs::create_dir_all(dir).expect("create clicks dir");
    }

    let mut annotated = 0usize;
    let mut skipped = 0usize;
    for track in &manifest.track {
        if track.bpm <= 0.0 {
            continue;
        }
        if !ids.is_empty() && !ids.iter().any(|i| i == &track.id) {
            continue;
        }
        let path = Path::new(AUDIO_BASE).join(&track.original);
        if !path.exists() {
            println!("SKIP {}: missing audio {}", track.id, path.display());
            skipped += 1;
            continue;
        }
        match annotate(track, &path, clicks_dir.as_deref()) {
            Ok(()) => annotated += 1,
            Err(e) => {
                println!("FAIL {}: {e}", track.id);
                skipped += 1;
            }
        }
    }
    println!("annotated {annotated} track(s), skipped {skipped}");
}

fn annotate(track: &TrackEntry, path: &Path, clicks_dir: Option<&Path>) -> Result<(), String> {
    let audio = decode_audio(path)?;
    let mono = timestretch::downmix_to_mid(&audio.data, audio.channels);
    let sr = audio.sample_rate as f64;
    let duration = mono.len() as f64 / sr;

    // Kick-band onset envelope: 4th-order low-pass -> hop RMS -> half-wave
    // rectified log-energy difference.
    let low = lowpass4(&mono, sr, KICK_BAND_HZ);
    let hop = (sr * HOP_SECS).round().max(1.0) as usize;
    let env = hop_rms(&low, hop);
    let onset = onset_strength(&env);
    let frame_secs = hop as f64 / sr;

    // Active region from a slow (1 s) RMS of the kick band.
    let slow = moving_mean(&env, (1.0 / HOP_SECS) as usize);
    let peak = slow.iter().cloned().fold(0.0f32, f32::max);
    let gate = peak * ACTIVE_GATE;
    let first_active = slow.iter().position(|&v| v > gate).unwrap_or(0);
    let last_active = slow.len() - 1 - slow.iter().rev().position(|&v| v > gate).unwrap_or(0);
    let (active_start, active_end) = (
        first_active as f64 * frame_secs,
        last_active as f64 * frame_secs,
    );

    // Rigid-grid fit: tiny BPM range around truth x full phase circle,
    // coarse pass then a refined pass around the winner.
    let fit = |bpms: &[f64], phases: &[f64]| -> (f64, f64, f64) {
        let mut best = (track.bpm, 0.0, f64::MIN);
        for &bpm in bpms {
            let period = 60.0 / bpm;
            for &phase in phases {
                let score = grid_score(&onset, frame_secs, phase, period, duration);
                if score > best.2 {
                    best = (bpm, phase, score);
                }
            }
        }
        best
    };
    let base_period = 60.0 / track.bpm;
    let coarse_bpms: Vec<f64> = (0..BPM_STEPS)
        .map(|i| {
            track.bpm * (1.0 - BPM_SEARCH_FRAC)
                + track.bpm * 2.0 * BPM_SEARCH_FRAC * i as f64 / (BPM_STEPS - 1) as f64
        })
        .collect();
    let coarse_phases: Vec<f64> = (0..PHASE_STEPS)
        .map(|i| base_period * i as f64 / PHASE_STEPS as f64)
        .collect();
    let (bpm0, phase0, _) = fit(&coarse_bpms, &coarse_phases);
    let bpm_step = track.bpm * 2.0 * BPM_SEARCH_FRAC / (BPM_STEPS - 1) as f64;
    let phase_step = base_period / PHASE_STEPS as f64;
    let fine_bpms: Vec<f64> = (0..21)
        .map(|i| bpm0 + bpm_step * (i as f64 - 10.0) / 10.0)
        .collect();
    let fine_phases: Vec<f64> = (0..33)
        .map(|i| phase0 + phase_step * (i as f64 - 16.0) / 16.0)
        .collect();
    let (bpm, phase, score) = fit(&fine_bpms, &fine_phases);

    // Phase decisiveness: winner vs the best phase at least an eighth of a
    // period away (same BPM). ~1.0 means "no better competing phase".
    let period = 60.0 / bpm;
    let mut rival = f64::MIN;
    for i in 0..PHASE_STEPS {
        let p = period * i as f64 / PHASE_STEPS as f64;
        let dist = (p - phase.rem_euclid(period)).abs();
        let dist = dist.min(period - dist);
        if dist >= period / 8.0 {
            rival = rival.max(grid_score(&onset, frame_secs, p, period, duration));
        }
    }
    let phase_lock = if score > 0.0 {
        1.0 - rival / score
    } else {
        0.0
    };

    // Beat times over the active region.
    let mut beats: Vec<f64> = Vec::new();
    let mut t = phase.rem_euclid(period);
    while t < duration {
        if t >= active_start - period * 0.5 && t <= active_end + period * 0.5 {
            beats.push(t);
        }
        t += period;
    }
    if beats.len() < 16 {
        return Err(format!("only {} beats in the active region", beats.len()));
    }

    // Downbeat phase: kick-band accent per beat-index rotation (mod 4).
    let mut rotation_scores = [0.0f64; 4];
    for (i, &b) in beats.iter().enumerate() {
        rotation_scores[i % 4] += sample_env(&onset, frame_secs, b);
    }
    for (r, s) in rotation_scores.iter_mut().enumerate() {
        let n = (beats.len() + 3 - r) / 4;
        *s /= n.max(1) as f64;
    }
    let best_r = (0..4)
        .max_by(|&a, &b| rotation_scores[a].total_cmp(&rotation_scores[b]))
        .unwrap_or(0);
    let mut sorted = rotation_scores;
    sorted.sort_by(|a, b| b.total_cmp(a));
    let downbeat_margin = if sorted[0] > 0.0 {
        (sorted[0] - sorted[1]) / sorted[0]
    } else {
        0.0
    };
    let downbeats: Vec<f64> = beats
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == best_r)
        .map(|(_, &b)| b)
        .collect();

    let round4 = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| (x * 1e4).round() / 1e4).collect() };
    let json = serde_json::json!({
        "beats": round4(&beats),
        "downbeats": round4(&downbeats),
        "meta": {
            "generator": "annotate_rigid_grid (rigid grid at manifest BPM, kick-band phase fit)",
            "manifest_bpm": track.bpm,
            "fitted_bpm": (bpm * 1e4).round() / 1e4,
            "fitted_phase_secs": (phase.rem_euclid(period) * 1e4).round() / 1e4,
            "phase_lock": (phase_lock * 1e3).round() / 1e3,
            "downbeat_rotation_scores": rotation_scores.map(|s| (s * 1e3).round() / 1e3),
            "downbeat_margin": (downbeat_margin * 1e3).round() / 1e3,
            "active_region_secs": [(active_start * 10.0).round() / 10.0, (active_end * 10.0).round() / 10.0],
            "verified_by_ear": false,
        },
    });
    let out = Path::new(ANNOTATIONS_DIR).join(format!("{}.json", track.id));
    std::fs::write(&out, serde_json::to_string_pretty(&json).unwrap() + "\n")
        .map_err(|e| format!("write {}: {e}", out.display()))?;

    let flag = if phase_lock < 0.3 || downbeat_margin < 0.15 {
        "  [VERIFY BY EAR]"
    } else {
        ""
    };
    println!(
        "OK {}: {} beats, fitted {:.3} BPM (label {:.1}), phase_lock {:.2}, \
         downbeat rotation {} margin {:.2}{}",
        track.id,
        beats.len(),
        bpm,
        track.bpm,
        phase_lock,
        best_r,
        downbeat_margin,
        flag
    );

    if let Some(dir) = clicks_dir {
        write_click_preview(dir, &track.id, &mono, sr, &beats, best_r)?;
    }
    Ok(())
}

/// Mean onset strength sampled at every grid point (linear interpolation).
fn grid_score(onset: &[f32], frame_secs: f64, phase: f64, period: f64, duration: f64) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    let mut t = phase.rem_euclid(period);
    while t < duration {
        sum += sample_env(onset, frame_secs, t);
        n += 1;
        t += period;
    }
    if n == 0 { f64::MIN } else { sum / n as f64 }
}

fn sample_env(env: &[f32], frame_secs: f64, t: f64) -> f64 {
    let pos = t / frame_secs;
    let i = pos.floor() as usize;
    if i + 1 >= env.len() {
        return 0.0;
    }
    let frac = pos - i as f64;
    env[i] as f64 * (1.0 - frac) + env[i + 1] as f64 * frac
}

/// Two cascaded RBJ low-pass biquads (Q = 0.707) -> 4th-order response.
fn lowpass4(input: &[f32], sr: f64, corner_hz: f64) -> Vec<f32> {
    let w0 = 2.0 * std::f64::consts::PI * corner_hz / sr;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * 0.707);
    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let (b0, b1, b2, a1, a2) = (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        -2.0 * cos_w0 / a0,
        (1.0 - alpha) / a0,
    );
    let mut out = input.to_vec();
    for _ in 0..2 {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for v in out.iter_mut() {
            let x0 = *v as f64;
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *v = y0 as f32;
        }
    }
    out
}

fn hop_rms(input: &[f32], hop: usize) -> Vec<f32> {
    input
        .chunks(hop)
        .map(|c| {
            (c.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c.len() as f64).sqrt() as f32
        })
        .collect()
}

/// Half-wave rectified log-energy difference.
fn onset_strength(env: &[f32]) -> Vec<f32> {
    let eps = 1e-6f64;
    let mut out = vec![0.0f32; env.len()];
    for i in 1..env.len() {
        let d = ((env[i] as f64 + eps).ln() - (env[i - 1] as f64 + eps).ln()).max(0.0);
        out[i] = d as f32;
    }
    out
}

fn moving_mean(input: &[f32], window: usize) -> Vec<f32> {
    let w = window.max(1);
    let mut out = vec![0.0f32; input.len()];
    let mut sum = 0.0f64;
    for i in 0..input.len() {
        sum += input[i] as f64;
        if i >= w {
            sum -= input[i - w] as f64;
        }
        out[i] = (sum / w.min(i + 1) as f64) as f32;
    }
    out
}

/// 24 s excerpt around the first downbeat with click overlays: 1.5 kHz tick
/// on downbeats, 1 kHz on other beats.
fn write_click_preview(
    dir: &Path,
    id: &str,
    mono: &[f32],
    sr: f64,
    beats: &[f64],
    downbeat_rotation: usize,
) -> Result<(), String> {
    let start = (beats[downbeat_rotation] - 1.0).max(0.0);
    let len_secs = 24.0f64;
    let s0 = (start * sr) as usize;
    let s1 = ((start + len_secs) * sr).min(mono.len() as f64) as usize;
    let mut buf: Vec<f32> = mono[s0..s1].iter().map(|&v| v * 0.7).collect();
    for (i, &b) in beats.iter().enumerate() {
        if b < start || b > start + len_secs {
            continue;
        }
        let is_down = i % 4 == downbeat_rotation;
        let freq = if is_down { 1500.0 } else { 1000.0 };
        let click_len = (sr * 0.025) as usize;
        let at = ((b - start) * sr) as usize;
        for k in 0..click_len {
            let idx = at + k;
            if idx >= buf.len() {
                break;
            }
            let t = k as f64 / sr;
            let envl = (-t * 180.0).exp();
            buf[idx] += (0.5 * envl * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32;
        }
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let path = dir.join(format!("{id}_clicks.wav"));
    let mut writer =
        hound::WavWriter::create(&path, spec).map_err(|e| format!("{}: {e}", path.display()))?;
    for &v in &buf {
        let s = (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(s).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    println!("   clicks: {}", path.display());
    Ok(())
}

// --- decode (mirrors qa/bpm_accuracy.rs) ----------------------------------

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
            .map_err(|e| format!("WAV decode failed: {e}"))?;
            Ok(DecodedAudio {
                sample_rate: buffer.sample_rate,
                channels: buffer.channels.count(),
                data: buffer.data,
            })
        }
        "mp3" | "aiff" | "aif" => decode_with_symphonia(path),
        other => Err(format!("unsupported extension '{other}'")),
    }
}

fn decode_with_symphonia(path: &Path) -> Result<DecodedAudio, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
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
        .map_err(|e| format!("format probe failed: {e}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "no decodable audio track".to_string())?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder for codec: {e}"))?;

    let mut data: Vec<f32> = Vec::new();
    let mut sample_rate = 0u32;
    let mut channels = 0usize;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("packet read failed: {e}")),
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
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode failed: {e}")),
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
