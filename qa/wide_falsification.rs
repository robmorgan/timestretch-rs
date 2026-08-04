//! Stage 11 falsification harness: wide-range Master Tempo prototype
//! (ROADMAP Stage 11).
//!
//! The bet under test: a big-FFT identity-locked phase vocoder with
//! artifact-driven per-band phase resets can sound acceptable at ±30–100%
//! tempo. This harness batch-renders the listening matrix — per track and
//! tempo rate, one WAV per arm — plus an objective metric sidecar
//! (supporting evidence only; the owner listening pass is the gate).
//!
//! Arms:
//! - `varispeed` — Tape profile at constant rate (what ships today beyond
//!   the ±35% correction fade: pitch follows).
//! - `widepv_bare` — the shipped batch wide path (`stretch()` beyond ±20%
//!   routes to `stretch_wide_pv`: FFT 2048, identity locking, no resets,
//!   no band split).
//! - `widepv_resets` — the Stage 11 prototype: same PV driven in
//!   streaming mode with artifact-driven per-band phase resets (policy
//!   inherited from the deleted Stage-9 corrector,
//!   `git show d44bb7e^:src/engine/stages/pv_corrector.rs`).
//! - `widepv_1024` — the prototype at FFT 1024 (the 1024-vs-2048
//!   empirical settle).
//! - `widepv_lowfree` — the Stage-2 experiment at wide ratios: low band
//!   (<120 Hz) follows tempo, high band through the prototype. A/B vs
//!   `widepv_resets` is the wide-ratio low-band verdict the exit criteria
//!   ask for.
//! - `widepv_cohblend` — the prototype with the phase-gradient coherence
//!   blend held at full strength at wide ratios (the shipped taper
//!   zeroes it approaching 2.5x, leaving big slowdowns with no vertical
//!   coherence — the down-side "roboty" candidate cause from the first
//!   listening round).
//! - `widepv_hop8` — the prototype at 87.5% overlap (hop = FFT/8):
//!   denser synthesis grid for expansions, 2x compute.
//! - `rubberband` — external reference via the `rubberband` CLI (plus a
//!   `rubberband_fine` R3 render when the installed version supports
//!   `--fine`).
//!
//! Tempo rates default to +30/-30/+50/-50/+100/-75%. -100% tempo is a
//! stop (rate 0, time ratio undefined), so the down-side edge cannot be
//! rendered at all; instead rate 0.25 (-75%, time ratio 4x) probes how
//! the corrector degrades as rate approaches zero (PV ratio grows toward
//! spectral freeze) — the "edge characterized, no gate" point the exit
//! criteria ask for, alongside +100% on the up side.
//!
//! Run the self-checks (no corpus or CLI needed):
//! `cargo test --features qa-harnesses --release --test wide_falsification -- --nocapture`
//!
//! Render the listening matrix (uses corpus tracks when present, the
//! synthetic bass fixture always):
//! `cargo test --features qa-harnesses --release --test wide_falsification -- --ignored --nocapture`
//! or `./scripts/wide_falsification.sh`, which preflights the
//! `rubberband` CLI and pretty-prints the summary.
//!
//! WAVs land in `target/wide_falsification/<track>/<rate_tag>/<arm>.wav`
//! (gitignored; bpm-corpus renders must never leave this machine), with
//! loudness-matched copies under `norm/` for unbiased listening and one
//! `summary.csv` row per render.

use std::path::{Path, PathBuf};
use std::process::Command;

use timestretch::analysis::comparison;
use timestretch::analysis::preanalysis::{analyze_for_dj, downmix_to_mid};
use timestretch::core::crossover::LinkwitzRiley8;
use timestretch::core::resample::resample_sinc_default;
use timestretch::io::wav::read_wav_file;
use timestretch::{
    Channels, PhaseLockingMode, PreAnalysisArtifact, StretchParams, WindowType, measure_loudness,
};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// Each harness compiles the shared adapter separately, so arms another
// harness uses read as dead code here.
#[allow(dead_code)]
#[path = "ab/mod.rs"]
mod ab;

use ab::{Arm, render_with_rate_schedule};

/// Tempo rates under test (time ratio is the reciprocal). 0.25 is the
/// down-side edge probe: -100% is a stop, so -75% (time ratio 4x) is the
/// deepest rendered point.
const DEFAULT_RATES: [f64; 6] = [1.30, 0.70, 1.50, 0.50, 2.00, 0.25];

/// Prototype FFT sizes: 2048 matches the shipped `stretch_wide_pv`
/// constants; 1024 is the empirical-settle candidate. Hop is FFT/4 (75%
/// overlap, the Hann COLA point).
const WIDE_FFT: usize = 2048;
const WIDE_FFT_SMALL: usize = 1024;

/// Rigid sub-bass locking region inside the PV, matching the shipped
/// `stretch_wide_pv` (offline.rs).
const SUB_BASS_CUTOFF_HZ: f32 = 100.0;

/// Low/high split for the `widepv_lowfree` arm. Mirrors
/// `KEYLOCK_CROSSOVER_HZ` in `src/engine/stages/band_split.rs` (which is
/// `pub(crate)` and unreachable from a test crate).
const CROSSOVER_HZ: f64 = 120.0;

/// Artifact onset strength above which the low bands (<500 Hz) re-lock
/// too; upper bands always re-lock on a transient. Inherited from the
/// Stage-9 corrector's `ONSET_LOW_BAND_RESET_STRENGTH`.
const ONSET_LOW_BAND_RESET_STRENGTH: f32 = 0.45;

/// A low band only re-locks when its per-onset flux is a real fraction of
/// the onset's strongest band (the full-spectrum prototype adds the
/// sub-100 Hz band the live corrector never sees; `onset_band_flux`
/// gates which bands actually moved).
const LOW_BAND_FLUX_FRACTION: f32 = 0.25;

/// Varispeed arm callback size (matches the other engine harnesses).
const CALLBACK_FRAMES: usize = 256;

const SAMPLE_RATE: u32 = 44_100;

fn output_dir() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));
    target_dir.join("wide_falsification")
}

fn rate_tag(rate: f64) -> String {
    format!("{:+.0}pct", (rate - 1.0) * 100.0)
}

// --- Rendering arms ---------------------------------------------------------

/// One onset's scheduled reset: (position in padded frames, band mask).
type ResetSchedule = Vec<(usize, [bool; 4])>;

/// PV configuration for one prototype arm.
#[derive(Clone, Copy)]
struct WidePvConfig {
    fft: usize,
    hop: usize,
    /// Hold the phase-gradient coherence blend at full strength at wide
    /// ratios (the shipped taper zeroes it approaching 2.5x — the
    /// down-side "roboty" candidate cause).
    coherence_blend: bool,
}

impl WidePvConfig {
    fn new(fft: usize) -> Self {
        Self {
            fft,
            hop: fft / 4,
            coherence_blend: false,
        }
    }

    fn with_coherence_blend(mut self) -> Self {
        self.coherence_blend = true;
        self
    }

    fn with_hop(mut self, hop: usize) -> Self {
        self.hop = hop;
        self
    }
}

/// Builds the per-band reset schedule from an artifact's onsets, in the
/// padded coordinates of [`wide_pv_render_mono`]. Policy inherited from
/// the Stage-9 corrector's `begin_block`: mids/highs always re-lock, the
/// low bands only on strong onsets whose flux actually lives there.
fn reset_schedule(
    artifact: &PreAnalysisArtifact,
    excerpt_frames: usize,
    start_pad: usize,
) -> ResetSchedule {
    let mut schedule = Vec::new();
    for (i, &pos) in artifact.transient_onsets.iter().enumerate() {
        if pos >= excerpt_frames {
            continue;
        }
        let strength = artifact.transient_strengths.get(i).copied().unwrap_or(1.0);
        let flux = artifact.onset_band_flux.get(i).copied().unwrap_or([1.0; 4]);
        let peak = flux.iter().fold(0.0f32, |a, &b| a.max(b)).max(1e-6);
        let mut mask = [false, false, true, true];
        if strength >= ONSET_LOW_BAND_RESET_STRENGTH {
            mask[0] = flux[0] >= LOW_BAND_FLUX_FRACTION * peak;
            mask[1] = flux[1] >= LOW_BAND_FLUX_FRACTION * peak;
        }
        schedule.push((pos + start_pad, mask));
    }
    schedule
}

/// The Stage 11 prototype renderer: streaming big-FFT identity-locked PV
/// with optional artifact-driven per-band phase resets.
///
/// Padding replicates the batch `PhaseVocoder::process` policy EXACTLY
/// (graduated start mirror, tapered end mirror), and the output trims
/// `round(start_pad * ratio)` — the same alignment math as the batch
/// path. This is load-bearing beyond alignment: identity locking
/// accumulates phase from the first analysis frame, so a different pad
/// length seeds different sub-bass phase state and the streaming and
/// batch renders would diverge by a constant rotation on bass-heavy
/// material.
///
/// Returns the rendered mono signal (exactly `round(frames * ratio)`
/// samples) and the number of reset events fired.
fn wide_pv_render_mono(
    mono: &[f32],
    sample_rate: u32,
    ratio: f64,
    config: WidePvConfig,
    artifact: Option<&PreAnalysisArtifact>,
) -> (Vec<f32>, u64) {
    use timestretch::stretch::phase_vocoder::PhaseVocoder;

    let WidePvConfig {
        fft: fft_size, hop, ..
    } = config;
    let frames = mono.len();
    let expected = (frames as f64 * ratio).round() as usize;
    if frames < fft_size {
        // Matches the shipped short-input fallback's spirit: resample to
        // length, pitch follows.
        return (resample_sinc_default(mono, expected.max(1)), 0);
    }

    let ratio_dist = (ratio - 1.0).abs();
    let start_pad_mult = if ratio_dist > 0.3 {
        4
    } else if ratio_dist > 0.15 {
        6
    } else {
        8
    };
    let end_pad_mult = if ratio > 1.1 { 10 } else { 8 };
    let start_pad = (hop * start_pad_mult).min(frames);
    let end_pad = (hop * end_pad_mult).min(frames);
    let mut padded = Vec::with_capacity(frames + start_pad + end_pad);
    for i in 0..start_pad {
        padded.push(mono[start_pad - 1 - i]);
    }
    padded.extend_from_slice(mono);
    for i in 0..end_pad {
        let t = (i + 1) as f32 / end_pad as f32;
        let fade = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
        padded.push(mono[frames - 1 - i] * fade);
    }

    let mut pv = PhaseVocoder::with_options(
        fft_size,
        hop,
        ratio,
        sample_rate,
        SUB_BASS_CUTOFF_HZ,
        WindowType::Hann,
        PhaseLockingMode::Identity,
    );
    if config.coherence_blend {
        pv.set_wide_ratio_coherence_blend(true);
    }
    pv.reserve_streaming_capacity(fft_size + hop, ratio.max(1.0) + 0.5);

    let schedule = artifact
        .map(|a| reset_schedule(a, frames, start_pad))
        .unwrap_or_default();
    let mut next_reset = 0usize;
    let mut fired = 0u64;

    let mut window: Vec<f32> = Vec::with_capacity(fft_size + hop);
    let mut chunk: Vec<f32> = Vec::with_capacity(fft_size * 4);
    let mut out: Vec<f32> = Vec::with_capacity(expected + fft_size * 4);
    let mut fed = 0usize;
    while fed < padded.len() {
        let take = hop.min(padded.len() - fed);
        window.extend_from_slice(&padded[fed..fed + take]);
        fed += take;
        if window.len() < fft_size {
            continue;
        }

        // Fire onsets whose position entered the ingested span, each
        // exactly once; masks OR together when several land in one hop.
        let mut mask = [false; 4];
        let mut fire = false;
        while next_reset < schedule.len() && schedule[next_reset].0 <= fed {
            for (m, &b) in mask.iter_mut().zip(schedule[next_reset].1.iter()) {
                *m |= b;
            }
            fire = true;
            next_reset += 1;
        }
        if fire {
            pv.reset_phase_state_bands(mask, sample_rate);
            fired += 1;
        }

        pv.process_streaming_into(&window[..fft_size], &mut chunk)
            .expect("wide pv streaming render");
        out.extend_from_slice(&chunk);

        let remaining = window.len() - hop;
        window.copy_within(hop.., 0);
        window.truncate(remaining);
    }

    let trim = ((start_pad as f64) * ratio).round() as usize;
    out.drain(..trim.min(out.len()));
    out.resize(expected, 0.0);
    (out, fired)
}

/// `widepv_lowfree`: low band (<120 Hz) follows tempo (resampled to
/// length — plain varispeed sound), high band through the prototype.
fn lowfree_render_mono(
    mono: &[f32],
    sample_rate: u32,
    ratio: f64,
    config: WidePvConfig,
    artifact: Option<&PreAnalysisArtifact>,
) -> Vec<f32> {
    let expected = (mono.len() as f64 * ratio).round() as usize;
    let mut xover = LinkwitzRiley8::new(CROSSOVER_HZ, sample_rate);
    let mut low = vec![0.0f32; mono.len()];
    let mut high = vec![0.0f32; mono.len()];
    xover.process(mono, &mut low, &mut high);

    let low_stretched = resample_sinc_default(&low, expected.max(1));
    let (mut out, _) = wide_pv_render_mono(&high, sample_rate, ratio, config, artifact);
    for (o, &l) in out.iter_mut().zip(low_stretched.iter()) {
        *o += l;
    }
    out
}

fn deinterleave(input: &[f32], channels: usize, ch: usize) -> Vec<f32> {
    input.iter().skip(ch).step_by(channels).copied().collect()
}

fn interleave(per_channel: &[Vec<f32>]) -> Vec<f32> {
    let channels = per_channel.len();
    let frames = per_channel.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = vec![0.0f32; frames * channels];
    for (ch, data) in per_channel.iter().enumerate() {
        for (i, &s) in data.iter().take(frames).enumerate() {
            out[i * channels + ch] = s;
        }
    }
    out
}

/// Runs a per-channel mono renderer over an interleaved buffer.
fn render_per_channel(
    input: &[f32],
    channels: usize,
    render: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let per: Vec<Vec<f32>> = (0..channels)
        .map(|ch| render(&deinterleave(input, channels, ch)))
        .collect();
    interleave(&per)
}

// --- Metrics ----------------------------------------------------------------

fn integrated_lufs(interleaved: &[f32], channels: usize, sample_rate: u32) -> Option<f64> {
    measure_loudness(interleaved, channels, sample_rate).map(|m| m.integrated_lufs)
}

/// Fraction of total RMS energy below the keylock crossover.
fn low_band_fraction(mono: &[f32], sample_rate: u32) -> f64 {
    let mut xover = LinkwitzRiley8::new(CROSSOVER_HZ, sample_rate);
    let (mut low_energy, mut total_energy) = (0.0f64, 0.0f64);
    for &s in mono {
        let (low, _high) = xover.process_sample(s);
        low_energy += (low as f64) * (low as f64);
        total_energy += (s as f64) * (s as f64);
    }
    if total_energy <= 0.0 {
        return 0.0;
    }
    (low_energy / total_energy).sqrt()
}

/// Sample-to-sample jumps exceeding a threshold derived from the source's
/// own slope distribution, per million samples (the robust click detector
/// from `examples/stream_quality_bench.rs`).
fn clicks_per_million(source: &[f32], output: &[f32], warmup: usize) -> f64 {
    let max_in_diff = source
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    let threshold = (max_in_diff * 1.5).max(0.05);
    if output.len() <= warmup + 1 {
        return 0.0;
    }
    let region = &output[warmup..];
    let clicks = region
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > threshold)
        .count();
    clicks as f64 * 1.0e6 / region.len() as f64
}

/// Best normalized cross-correlation between `a` and `b` over lags in
/// `[-max_lag, max_lag]`, scanning a window of `a` starting at `offset`.
/// Returns `(lag, correlation)` — positive lag means `b` is late.
fn best_lag_xcorr(
    a: &[f32],
    b: &[f32],
    offset: usize,
    window: usize,
    max_lag: usize,
) -> (i64, f64) {
    let n = window
        .min(a.len().saturating_sub(offset))
        .min(b.len().saturating_sub(offset + max_lag));
    if n < 1024 || offset < max_lag {
        return (0, 0.0);
    }
    let a_win = &a[offset..offset + n];
    let a_norm = a_win
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        .sqrt();
    let mut best = (0i64, f64::MIN);
    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let b_start = (offset as i64 + lag) as usize;
        let b_win = &b[b_start..b_start + n];
        let dot: f64 = a_win
            .iter()
            .zip(b_win.iter())
            .map(|(&x, &y)| (x as f64) * (y as f64))
            .sum();
        let b_norm = b_win
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            .sqrt();
        let corr = if a_norm > 0.0 && b_norm > 0.0 {
            dot / (a_norm * b_norm)
        } else {
            0.0
        };
        if corr > best.1 {
            best = (lag, corr);
        }
    }
    best
}

// --- Fixtures and I/O -------------------------------------------------------

/// Bass-heavy synthetic fixture (same recipe as the Stage 2 listening
/// fixture in `qa/engine_keylock.rs`): 55 Hz sub with 220/880 Hz partials
/// and decaying 60 Hz kicks at 120 BPM. The only material where the
/// zero-crossing cents metric is valid — real mixes have no single
/// dominant tone.
fn bass_fixture(seconds: usize) -> Vec<f32> {
    let len = SAMPLE_RATE as usize * seconds;
    let mut input = vec![0.0f32; len];
    for (i, s) in input.iter_mut().enumerate() {
        let t = i as f64 / SAMPLE_RATE as f64;
        *s += 0.45 * (2.0 * std::f64::consts::PI * 55.0 * t).sin() as f32;
        *s += 0.15 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() as f32;
        *s += 0.1 * (2.0 * std::f64::consts::PI * 880.0 * t).sin() as f32;
    }
    let beat = SAMPLE_RATE as usize / 2;
    for start in (0..len).step_by(beat) {
        for k in 0..2_000.min(len - start) {
            let t = k as f64 / SAMPLE_RATE as f64;
            let env = (-t * 18.0).exp();
            input[start + k] += (0.8 * env * (2.0 * std::f64::consts::PI * 60.0 * t).sin()) as f32;
        }
    }
    input
}

fn write_wav(path: &Path, interleaved: &[f32], channels: usize, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for &s in interleaved {
        writer
            .write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

/// Decodes an MP3 to interleaved f32 via symphonia (same decoder as
/// `qa/rubberband_reference_gate.rs`).
fn decode_mp3(path: &Path) -> (Vec<f32>, u32, usize) {
    let file = std::fs::File::open(path).expect("open corpus mp3");
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .expect("probe corpus mp3");
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("audio track")
        .clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .expect("mp3 decoder");
    let mut samples = Vec::new();
    let mut sample_rate = 44_100;
    let mut channels = 2;
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track.id {
            continue;
        }
        if let Ok(decoded) = decoder.decode(&packet) {
            let spec = *decoded.spec();
            sample_rate = spec.rate;
            channels = spec.channels.count();
            let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buf.copy_interleaved_ref(decoded);
            samples.extend_from_slice(buf.samples());
        }
    }
    (samples, sample_rate, channels)
}

fn load_track(path: &Path) -> (Vec<f32>, u32, usize) {
    if path.extension().map(|e| e == "mp3").unwrap_or(false) {
        decode_mp3(path)
    } else {
        let buffer = read_wav_file(path.to_string_lossy().as_ref()).expect("read corpus wav");
        let channels = match buffer.channels {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        };
        (buffer.data, buffer.sample_rate, channels)
    }
}

/// Corpus tracks for the matrix: env override
/// `TIMESTRETCH_WIDE_TRACKS="tag=path,tag=path"`, else the bass-heavy
/// defaults that exist locally. bpm-corpus is commercial material — its
/// renders stay in gitignored `target/` and must never be published;
/// `saucers` is CC-licensed (public-corpus) if an example render ever
/// needs sharing.
fn corpus_tracks() -> Vec<(String, PathBuf)> {
    if let Ok(spec) = std::env::var("TIMESTRETCH_WIDE_TRACKS") {
        return spec
            .split(',')
            .filter_map(|entry| {
                let (tag, path) = entry.split_once('=')?;
                Some((tag.trim().to_string(), PathBuf::from(path.trim())))
            })
            .collect();
    }
    [
        (
            "msbwy",
            "benchmarks/audio/bpm-corpus/12247392_Music Sounds Better With You_(Original Mix).wav",
        ),
        (
            "hot_stuff",
            "benchmarks/audio/bpm-corpus/14220825_Hot Stuff_(Original Mix).wav",
        ),
        (
            "somebody",
            "benchmarks/audio/bpm-corpus/15650709_Somebody To Love_(Extended Mix).wav",
        ),
        (
            "saucers",
            "benchmarks/audio/public-corpus/01-Interplanetary_Criminal-Saucers.mp3",
        ),
    ]
    .into_iter()
    .map(|(tag, path)| (tag.to_string(), PathBuf::from(path)))
    .filter(|(_, path)| path.exists())
    .collect()
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn rates_under_test() -> Vec<f64> {
    std::env::var("TIMESTRETCH_WIDE_RATES")
        .ok()
        .map(|spec| {
            spec.split_whitespace()
                .filter_map(|r| r.parse().ok())
                .collect()
        })
        .filter(|v: &Vec<f64>| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_RATES.to_vec())
}

// --- The listening matrix ---------------------------------------------------

struct TrackMaterial {
    tag: String,
    interleaved: Vec<f32>,
    channels: usize,
    sample_rate: u32,
}

/// Loads the excerpt for one track: `TIMESTRETCH_WIDE_START_SECONDS` into
/// the file (clamped to keep the window in range),
/// `TIMESTRETCH_WIDE_MAX_SECONDS` long.
fn excerpt(path: &Path, tag: &str) -> TrackMaterial {
    let (interleaved, sample_rate, channels) = load_track(path);
    let frames = interleaved.len() / channels;
    let window = (env_f64("TIMESTRETCH_WIDE_MAX_SECONDS", 20.0) * sample_rate as f64) as usize;
    let start = (env_f64("TIMESTRETCH_WIDE_START_SECONDS", 60.0) * sample_rate as f64) as usize;
    let start = start.min(frames.saturating_sub(window));
    let end = (start + window).min(frames);
    TrackMaterial {
        tag: tag.to_string(),
        interleaved: interleaved[start * channels..end * channels].to_vec(),
        channels,
        sample_rate,
    }
}

fn rubberband_available() -> bool {
    Command::new("rubberband").arg("--version").output().is_ok()
}

/// Renders an external reference via the `rubberband` CLI. Returns false
/// (and skips the arm) when the CLI or the flag is unavailable.
fn render_rubberband(source: &Path, dest: &Path, ratio: f64, fine: bool) -> bool {
    let mut cmd = Command::new("rubberband");
    cmd.arg("-q");
    if fine {
        cmd.arg("--fine");
    }
    cmd.arg("--time")
        .arg(format!("{ratio}"))
        .arg(source)
        .arg(dest);
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

#[test]
#[ignore = "renders Stage 11 falsification listening material; run explicitly"]
fn falsification_render_wide_listening_matrix() {
    let out_root = output_dir();
    std::fs::create_dir_all(&out_root).expect("create output dir");
    let rates = rates_under_test();
    let rubberband = rubberband_available();
    if rubberband {
        let version = Command::new("rubberband")
            .arg("--version")
            .output()
            .map(|o| {
                String::from_utf8_lossy(if o.stderr.is_empty() {
                    &o.stdout
                } else {
                    &o.stderr
                })
                .trim()
                .to_string()
            })
            .unwrap_or_default();
        std::fs::write(
            out_root.join("versions.txt"),
            format!("rubberband {version}\n"),
        )
        .expect("write versions");
    } else {
        println!("NOTE: `rubberband` CLI not installed — reference arm skipped");
    }

    let mut materials: Vec<TrackMaterial> = vec![TrackMaterial {
        tag: "synthetic_bass".to_string(),
        interleaved: bass_fixture(12),
        channels: 1,
        sample_rate: SAMPLE_RATE,
    }];
    for (tag, path) in corpus_tracks() {
        materials.push(excerpt(&path, &tag));
    }

    let mut csv = String::from(
        "track,rate,arm,frames,lufs,lufs_delta_source,low_band_ratio,clicks_per_million,\
         rb_spectral,rb_perceptual,rb_lufs_diff,rb_lag_samples\n",
    );

    for material in &materials {
        let TrackMaterial {
            tag,
            interleaved,
            channels,
            sample_rate,
        } = material;
        let (channels, sample_rate) = (*channels, *sample_rate);
        let track_dir = out_root.join(tag);
        std::fs::create_dir_all(&track_dir).expect("create track dir");
        let source_wav = track_dir.join("source.wav");
        write_wav(&source_wav, interleaved, channels, sample_rate);

        let mono = downmix_to_mid(interleaved, channels);
        let artifact = analyze_for_dj(&mono, sample_rate);
        let src_lufs = integrated_lufs(interleaved, channels, sample_rate);
        let src_low = low_band_fraction(&mono, sample_rate);
        println!(
            "{tag}: {:.1}s, {} onsets, source LUFS {}",
            mono.len() as f64 / sample_rate as f64,
            artifact.transient_onsets.len(),
            src_lufs.map_or("n/a".into(), |l| format!("{l:.1}")),
        );

        for &rate in &rates {
            let ratio = 1.0 / rate;
            let rate_dir = track_dir.join(rate_tag(rate));
            std::fs::create_dir_all(&rate_dir).expect("create rate dir");

            let mut renders: Vec<(&str, Vec<f32>)> = Vec::new();

            let varispeed = render_with_rate_schedule(
                Arm::Tape,
                interleaved,
                channels,
                sample_rate,
                CALLBACK_FRAMES,
                &|_| rate,
            );
            renders.push(("varispeed", varispeed.output));

            let params = StretchParams::new(ratio)
                .with_sample_rate(sample_rate)
                .with_channels(channels as u32);
            renders.push((
                "widepv_bare",
                timestretch::stretch(interleaved, &params).expect("shipped wide stretch"),
            ));

            renders.push((
                "widepv_resets",
                render_per_channel(interleaved, channels, |ch| {
                    wide_pv_render_mono(
                        ch,
                        sample_rate,
                        ratio,
                        WidePvConfig::new(WIDE_FFT),
                        Some(&artifact),
                    )
                    .0
                }),
            ));
            renders.push((
                "widepv_1024",
                render_per_channel(interleaved, channels, |ch| {
                    wide_pv_render_mono(
                        ch,
                        sample_rate,
                        ratio,
                        WidePvConfig::new(WIDE_FFT_SMALL),
                        Some(&artifact),
                    )
                    .0
                }),
            ));
            renders.push((
                "widepv_lowfree",
                render_per_channel(interleaved, channels, |ch| {
                    lowfree_render_mono(
                        ch,
                        sample_rate,
                        ratio,
                        WidePvConfig::new(WIDE_FFT),
                        Some(&artifact),
                    )
                }),
            ));
            renders.push((
                "widepv_cohblend",
                render_per_channel(interleaved, channels, |ch| {
                    wide_pv_render_mono(
                        ch,
                        sample_rate,
                        ratio,
                        WidePvConfig::new(WIDE_FFT).with_coherence_blend(),
                        Some(&artifact),
                    )
                    .0
                }),
            ));
            renders.push((
                "widepv_hop8",
                render_per_channel(interleaved, channels, |ch| {
                    wide_pv_render_mono(
                        ch,
                        sample_rate,
                        ratio,
                        WidePvConfig::new(WIDE_FFT).with_hop(WIDE_FFT / 8),
                        Some(&artifact),
                    )
                    .0
                }),
            ));

            let mut rb_mono: Option<Vec<f32>> = None;
            if rubberband {
                let rb_wav = rate_dir.join("rubberband.wav");
                if render_rubberband(&source_wav, &rb_wav, ratio, false) {
                    let rb = read_wav_file(rb_wav.to_string_lossy().as_ref())
                        .expect("read rubberband render");
                    let rb_channels = match rb.channels {
                        Channels::Mono => 1,
                        Channels::Stereo => 2,
                    };
                    rb_mono = Some(downmix_to_mid(&rb.data, rb_channels));
                }
                // R3's finer engine is the honest competitive bar where
                // the installed version has it.
                let fine_wav = rate_dir.join("rubberband_fine.wav");
                if !render_rubberband(&source_wav, &fine_wav, ratio, true) {
                    let _ = std::fs::remove_file(&fine_wav);
                }
            }

            for (arm, output) in &renders {
                write_wav(
                    &rate_dir.join(format!("{arm}.wav")),
                    output,
                    channels,
                    sample_rate,
                );

                let out_mono = downmix_to_mid(output, channels);
                let frames = output.len() / channels;
                let lufs = integrated_lufs(output, channels, sample_rate);
                let delta = match (lufs, src_lufs) {
                    (Some(l), Some(s)) => Some(l - s),
                    _ => None,
                };
                if let Some(d) = delta {
                    if d.abs() > 1.5 {
                        println!(
                            "WARN {tag}/{}/{arm}: level off by {d:+.1} LUFS",
                            rate_tag(rate)
                        );
                    }
                    // Loudness-matched copy for unbiased listening; the
                    // unmatched primary render stays the artifact of
                    // record (level loss is a finding).
                    let norm_dir = rate_dir.join("norm");
                    std::fs::create_dir_all(&norm_dir).expect("create norm dir");
                    let gain = 10f32.powf((-14.0 - lufs.unwrap_or(-14.0)) as f32 / 20.0);
                    let normed: Vec<f32> = output.iter().map(|&s| s * gain).collect();
                    write_wav(
                        &norm_dir.join(format!("{arm}.wav")),
                        &normed,
                        channels,
                        sample_rate,
                    );
                }
                let low_ratio = if src_low > 0.0 {
                    low_band_fraction(&out_mono, sample_rate) / src_low
                } else {
                    0.0
                };
                let clicks = clicks_per_million(&mono, &out_mono, (sample_rate / 2) as usize);

                let (rb_cols, rb_lag) = match &rb_mono {
                    Some(rb) => {
                        let compare_len = out_mono.len().min(rb.len());
                        let report = comparison::generate_quality_report(
                            &out_mono[..compare_len],
                            &rb[..compare_len],
                            sample_rate,
                            2048,
                            512,
                        );
                        let (lag, _corr) = best_lag_xcorr(
                            &out_mono,
                            rb,
                            sample_rate as usize,
                            2 * sample_rate as usize,
                            WIDE_FFT / 2,
                        );
                        (
                            format!(
                                "{:.3},{:.3},{:+.2}",
                                report.spectral_similarity,
                                report.perceptual_spectral_similarity,
                                report.lufs_difference
                            ),
                            format!("{lag}"),
                        )
                    }
                    None => (",,".to_string(), String::new()),
                };

                csv.push_str(&format!(
                    "{tag},{},{arm},{frames},{},{},{low_ratio:.3},{clicks:.1},{rb_cols},{rb_lag}\n",
                    rate_tag(rate),
                    lufs.map_or(String::new(), |l| format!("{l:.2}")),
                    delta.map_or(String::new(), |d| format!("{d:+.2}")),
                ));
            }
            println!("rendered {tag} @ {}", rate_tag(rate));
        }
    }

    let summary = out_root.join("summary.csv");
    std::fs::write(&summary, csv).expect("write summary");
    println!("listening matrix in {}", out_root.display());
    println!("summary: {}", summary.display());
}

// --- Harness self-checks (trust the harness before trusting renders) --------

#[test]
fn wide_pv_identity_round_trip() {
    use timestretch::stretch::phase_vocoder::PhaseVocoder;

    let input = bass_fixture(4);
    for fft in [WIDE_FFT, WIDE_FFT_SMALL] {
        let (out, fired) =
            wide_pv_render_mono(&input, SAMPLE_RATE, 1.0, WidePvConfig::new(fft), None);
        assert_eq!(out.len(), input.len(), "identity length (fft {fft})");
        assert_eq!(fired, 0, "no artifact, no resets");
        let hop = fft / 4;

        // The harness's streaming driver must reproduce the shipped batch
        // PV (the reference implementation) — that is the correctness
        // claim. Fidelity to the SOURCE is a property of the FFT size
        // itself and belongs to the experiment: 2048 is near-transparent
        // at unity, while 1024 visibly rotates sub-bass phase on this
        // 55 Hz-heavy fixture — early evidence for the 1024-vs-2048
        // settle.
        let mut batch_pv = PhaseVocoder::with_options(
            fft,
            hop,
            1.0,
            SAMPLE_RATE,
            SUB_BASS_CUTOFF_HZ,
            WindowType::Hann,
            PhaseLockingMode::Identity,
        );
        let mut batch = batch_pv.process(&input).expect("batch render");
        batch.resize(input.len(), 0.0);
        let (b_lag, b_corr) = best_lag_xcorr(
            &batch,
            &out,
            SAMPLE_RATE as usize,
            2 * SAMPLE_RATE as usize,
            2 * hop,
        );
        let (s_lag, s_corr) = best_lag_xcorr(
            &input,
            &out,
            SAMPLE_RATE as usize,
            2 * SAMPLE_RATE as usize,
            2 * hop,
        );
        println!(
            "identity fft {fft}: vs batch lag {b_lag} xcorr {b_corr:.4}, \
             vs source lag {s_lag} xcorr {s_corr:.4}"
        );
        assert!(
            b_lag.unsigned_abs() as usize <= hop,
            "streaming driver misaligned vs batch by {b_lag} samples (fft {fft})"
        );
        assert!(
            b_corr > 0.98,
            "streaming driver diverged from batch: {b_corr:.4} (fft {fft})"
        );
        if fft == WIDE_FFT {
            assert!(
                s_lag.unsigned_abs() as usize <= hop,
                "identity render misaligned by {s_lag} samples"
            );
            assert!(s_corr > 0.98, "identity render decorrelated: {s_corr:.4}");
            let in_lufs = integrated_lufs(&input, 1, SAMPLE_RATE).expect("source lufs");
            let out_lufs = integrated_lufs(&out, 1, SAMPLE_RATE).expect("render lufs");
            assert!(
                (in_lufs - out_lufs).abs() < 0.5,
                "identity level off: {:+.2} LUFS",
                out_lufs - in_lufs
            );
        }
    }
}

#[test]
fn wide_renders_have_exact_length() {
    let input = bass_fixture(4);
    let frames = input.len();
    for rate in DEFAULT_RATES {
        let ratio = 1.0 / rate;
        let expected = (frames as f64 * ratio).round() as usize;
        for (label, config) in [
            ("widepv", WidePvConfig::new(WIDE_FFT)),
            (
                "cohblend",
                WidePvConfig::new(WIDE_FFT).with_coherence_blend(),
            ),
            ("hop8", WidePvConfig::new(WIDE_FFT).with_hop(WIDE_FFT / 8)),
        ] {
            let (out, _) = wide_pv_render_mono(&input, SAMPLE_RATE, ratio, config, None);
            assert_eq!(out.len(), expected, "{label} length at rate {rate}");
            // The trailing edge must be real content, not the resize
            // backfill: the end mirror pad guarantees the streaming tail
            // covers it.
            let tail = &out[expected.saturating_sub(2_048)..];
            let tail_rms = (tail.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
                / tail.len() as f64)
                .sqrt();
            assert!(tail_rms > 0.01, "{label} tail is silence at rate {rate}");
        }
        let low = lowfree_render_mono(
            &input,
            SAMPLE_RATE,
            ratio,
            WidePvConfig::new(WIDE_FFT),
            None,
        );
        assert_eq!(low.len(), expected, "lowfree length at rate {rate}");
    }
}

#[test]
fn coherence_blend_engages_only_at_wide_expansion() {
    let input = bass_fixture(4);
    let base = WidePvConfig::new(WIDE_FFT);
    let blend = base.with_coherence_blend();
    let rel_diff = |ratio: f64| {
        let (a, _) = wide_pv_render_mono(&input, SAMPLE_RATE, ratio, base, None);
        let (b, _) = wide_pv_render_mono(&input, SAMPLE_RATE, ratio, blend, None);
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum();
        let reference: f64 = a.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (diff / reference.max(1e-12)).sqrt()
    };

    // At unity the shipped taper is already at full strength, so the knob
    // must be a no-op; at 2x expansion the shipped taper is zero, so the
    // knob must change the render.
    let at_unity = rel_diff(1.0);
    let at_expansion = rel_diff(2.0);
    println!("cohblend rel diff: unity {at_unity:.6}, 2x expansion {at_expansion:.6}");
    assert!(
        at_unity < 1e-6,
        "cohblend altered the unity render: {at_unity}"
    );
    assert!(
        at_expansion > 1e-3,
        "cohblend had no effect at 2x expansion: {at_expansion}"
    );
}

#[test]
fn band_split_resums_to_allpass() {
    let input = bass_fixture(4);
    let mut xover = LinkwitzRiley8::new(CROSSOVER_HZ, SAMPLE_RATE);
    let mut low = vec![0.0f32; input.len()];
    let mut high = vec![0.0f32; input.len()];
    xover.process(&input, &mut low, &mut high);
    let resum: Vec<f32> = low.iter().zip(high.iter()).map(|(&l, &h)| l + h).collect();

    let rms = |x: &[f32]| {
        (x.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / x.len() as f64).sqrt()
    };
    let ratio = rms(&resum) / rms(&input);
    assert!(
        (ratio - 1.0).abs() < 0.03,
        "LR8 re-sum level off: {ratio:.4}"
    );
    let report = comparison::generate_quality_report(&resum, &input, SAMPLE_RATE, 2048, 512);
    assert!(
        report.spectral_similarity > 0.95,
        "LR8 re-sum spectrum diverged: {:.3}",
        report.spectral_similarity
    );
}

#[test]
fn artifact_resets_fire_and_change_the_render() {
    let input = bass_fixture(6);
    let artifact = analyze_for_dj(&input, SAMPLE_RATE);
    assert!(
        artifact.transient_onsets.len() >= 4,
        "kick fixture should yield onsets, got {}",
        artifact.transient_onsets.len()
    );

    let ratio = 1.0 / 1.5;
    let (with_resets, fired) = wide_pv_render_mono(
        &input,
        SAMPLE_RATE,
        ratio,
        WidePvConfig::new(WIDE_FFT),
        Some(&artifact),
    );
    let (without, _) = wide_pv_render_mono(
        &input,
        SAMPLE_RATE,
        ratio,
        WidePvConfig::new(WIDE_FFT),
        None,
    );
    println!(
        "resets fired: {fired} ({} onsets in artifact)",
        artifact.transient_onsets.len()
    );
    assert!(fired > 0, "no resets fired on the kick fixture");
    assert!(
        fired as usize <= artifact.transient_onsets.len(),
        "each onset fires at most once"
    );

    let diff_energy: f64 = with_resets
        .iter()
        .zip(without.iter())
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let ref_energy: f64 = without.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rel = (diff_energy / ref_energy.max(1e-12)).sqrt();
    println!("reset arm relative difference: {rel:.4}");
    assert!(rel > 1e-3, "resets had no effect on the render");
}
