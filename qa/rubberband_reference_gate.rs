//! External-reference quality gate (ROADMAP Stage 7): the pull engine's
//! keylock chain vs a Rubber Band CLI render of the same public-corpus
//! track, REQUIRED in CI.
//!
//! Unlike `rubberband_comparison.rs` (a manual harness fed by env vars),
//! this test is self-contained: it decodes a pinned public-corpus track
//! (see `scripts/fetch_public_corpus.sh`), produces the external reference
//! by invoking the `rubberband` CLI, renders the same stretch through the
//! new engine, and gates on spectral similarity and level integrity.
//!
//! Skips (with a message) when the corpus or the `rubberband` binary is
//! missing so local `--all-targets` runs stay green; CI exports
//! `TIMESTRETCH_REQUIRE_RUBBERBAND=1` to turn either absence into a hard
//! failure.

use std::path::{Path, PathBuf};
use std::process::Command;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use timestretch::analysis::comparison;
use timestretch::io::wav::{read_wav_file, write_wav_file_float};
use timestretch::stretch::hybrid::HybridStretcher;
use timestretch::{AudioBuffer, EdmPreset, StretchParams};

/// Public-corpus fixture: Valya Kan – Memories [YARN020], CC-BY 4.0,
/// verified 125.000 BPM (the corpus's cleanest steady-tempo entry).
const CORPUS_FILE: &str = "benchmarks/audio/public-corpus/02-Valya_Kan-Memories.mp3";
/// Comparison window, in seconds, taken from this offset into the track
/// (past the beatless intro, inside the full-mix section).
const START_SECS: f64 = 60.0;
const WINDOW_SECS: f64 = 25.0;
/// DJ beatmatch stretches under gate: 125 -> 118 BPM (slow down) and
/// 125 -> 132 BPM (speed up); both inside the keylock's SOLA band.
const TARGET_BPMS: [f64; 2] = [118.0, 132.0];
const SOURCE_BPM: f64 = 125.0;

const REQUIRE_ENV: &str = "TIMESTRETCH_REQUIRE_RUBBERBAND";
const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;

fn require() -> bool {
    std::env::var(REQUIRE_ENV)
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn output_dir() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));
    target_dir.join("rubberband_reference_gate")
}

/// Decodes an MP3 to interleaved f32 via symphonia.
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

/// Renders through the new pull engine's keylock chain at constant rate
/// `1 / ratio` and trims the pipeline-latency prefix (same driver as
/// `rubberband_comparison.rs`).
fn render_new_engine(input: &[f32], sample_rate: u32, ratio: f64) -> Vec<f32> {
    use timestretch::engine::{Engine, EngineConfig, EngineProfile};
    let rate = 1.0 / ratio;
    let handles = Engine::build(EngineConfig {
        sample_rate,
        channels: 1,
        profile: EngineProfile::Keylock,
        initial_tempo_rate: rate,
        ..EngineConfig::default()
    })
    .expect("engine builds");
    let (controller, mut processor, mut source) =
        (handles.controller, handles.processor, handles.source);
    source.set_track_position(0);
    let latency = processor.pipeline_latency_frames();

    let mut feed = 0usize;
    let mut finished = false;
    let mut out = vec![0.0f32; 256];
    let mut collected: Vec<f32> = Vec::with_capacity((input.len() as f64 * ratio) as usize + 4096);
    loop {
        while feed < input.len() && source.occupied_frames() < 8_192 {
            let end = (feed + 8_192).min(input.len());
            feed += source.push(&input[feed..end]);
        }
        if feed >= input.len() && !finished {
            finished = source.finish();
        }
        let underruns_before = controller.underrun_frames();
        processor.process(&mut out);
        if controller.underrun_frames() > underruns_before {
            let shortfall = (controller.underrun_frames() - underruns_before) as usize;
            collected.extend_from_slice(&out[..out.len() - shortfall.min(out.len())]);
            break;
        }
        collected.extend_from_slice(&out);
        if collected.len() > input.len() * 6 + 100_000 {
            panic!("new-engine render did not terminate");
        }
    }
    collected.drain(..latency.min(collected.len()));
    collected
}

#[test]
fn keylock_meets_external_rubberband_reference() {
    let corpus = Path::new(CORPUS_FILE);
    if !corpus.exists() {
        if require() {
            panic!(
                "{REQUIRE_ENV}=1 but the public corpus is missing — run \
                 scripts/fetch_public_corpus.sh"
            );
        }
        println!("Skipping: public corpus not fetched (scripts/fetch_public_corpus.sh)");
        return;
    }
    let rubberband_ok = Command::new("rubberband").arg("--version").output().is_ok();
    if !rubberband_ok {
        if require() {
            panic!("{REQUIRE_ENV}=1 but the `rubberband` CLI is not installed");
        }
        println!("Skipping: `rubberband` CLI not installed");
        return;
    }

    // Decode, downmix, and trim the comparison window.
    let (interleaved, sample_rate, channels) = decode_mp3(corpus);
    let mono = timestretch::downmix_to_mid(&interleaved, channels);
    let start = (START_SECS * sample_rate as f64) as usize;
    let len = (WINDOW_SECS * sample_rate as f64) as usize;
    assert!(
        mono.len() >= start + len,
        "corpus track shorter than the comparison window"
    );
    let window = &mono[start..start + len];

    let out_dir = output_dir();
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let original_wav = out_dir.join("original.wav");
    let original_buffer = AudioBuffer::from_mono(window.to_vec(), sample_rate);
    write_wav_file_float(original_wav.to_string_lossy().as_ref(), &original_buffer)
        .expect("write original wav");

    for target_bpm in TARGET_BPMS {
        let ratio = SOURCE_BPM / target_bpm;
        let reference_wav = out_dir.join(format!("reference_{target_bpm:.0}.wav"));
        let status = Command::new("rubberband")
            .arg("-q")
            .arg("--time")
            .arg(format!("{ratio}"))
            .arg(&original_wav)
            .arg(&reference_wav)
            .status()
            .expect("run rubberband");
        assert!(status.success(), "rubberband CLI failed for {target_bpm}");

        let reference = read_wav_file(reference_wav.to_string_lossy().as_ref())
            .expect("read rubberband reference")
            .mix_to_mono();
        let stretched = render_new_engine(window, sample_rate, ratio);

        let compare_len = stretched.len().min(reference.data.len());
        assert!(compare_len >= FFT_SIZE, "comparison window too short");
        let report = comparison::generate_quality_report(
            &stretched[..compare_len],
            &reference.data[..compare_len],
            sample_rate,
            FFT_SIZE,
            HOP_SIZE,
        );

        // Captured-hybrid-baseline comparison (ROADMAP Stage 8): the frozen
        // batch engine rendered against the same external reference. The
        // new engine must score at least as well (small tolerance for
        // metric noise). Retires with the hybrid at Stage 9.
        let hybrid_params = StretchParams::new(ratio)
            .with_sample_rate(sample_rate)
            .with_channels(1)
            .with_preset(EdmPreset::DjBeatmatch);
        let hybrid_out = HybridStretcher::new(hybrid_params)
            .process(window)
            .expect("hybrid render");
        let hybrid_len = hybrid_out.len().min(reference.data.len());
        let hybrid_report = comparison::generate_quality_report(
            &hybrid_out[..hybrid_len],
            &reference.data[..hybrid_len],
            sample_rate,
            FFT_SIZE,
            HOP_SIZE,
        );
        println!(
            "rubberband gate: 125->{target_bpm} BPM (ratio {ratio:.4}): spectral={:.3} \
             perceptual={:.3} lufs_diff={:+.2} flux={:.3} | hybrid baseline: spectral={:.3} \
             perceptual={:.3}",
            report.spectral_similarity,
            report.perceptual_spectral_similarity,
            report.lufs_difference,
            report.spectral_flux_similarity,
            hybrid_report.spectral_similarity,
            hybrid_report.perceptual_spectral_similarity
        );
        assert!(
            report.spectral_similarity >= hybrid_report.spectral_similarity - 0.02,
            "batch quality fell below the hybrid baseline: {:.3} vs {:.3}",
            report.spectral_similarity,
            hybrid_report.spectral_similarity
        );

        // Gates. Thresholds calibrated 2026-07 against rubberband 4.0
        // (macOS) with margin for CLI version variance on CI runners; a
        // regression in the keylock chain (HF loss, level sag, splice
        // artifacts) moves these by far more than version noise does.
        assert!(
            report.spectral_similarity >= 0.85,
            "spectral similarity vs external reference regressed: {:.3}",
            report.spectral_similarity
        );
        assert!(
            report.perceptual_spectral_similarity >= 0.85,
            "perceptual similarity vs external reference regressed: {:.3}",
            report.perceptual_spectral_similarity
        );
        assert!(
            report.lufs_difference.abs() <= 1.5,
            "level vs external reference off by {:.2} LUFS",
            report.lufs_difference
        );
    }
}
