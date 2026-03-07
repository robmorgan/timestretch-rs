use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use timestretch::analysis::comparison;
use timestretch::io::wav::{read_wav_file, write_wav_file_float};
use timestretch::{AudioBuffer, EdmPreset, StretchParams};

const ORIG_WAV_ENV: &str = "TIMESTRETCH_RUBBERBAND_ORIGINAL_WAV";
const REF_WAV_ENV: &str = "TIMESTRETCH_RUBBERBAND_REFERENCE_WAV";
const RATIO_ENV: &str = "TIMESTRETCH_RUBBERBAND_RATIO";
const SOURCE_BPM_ENV: &str = "TIMESTRETCH_RUBBERBAND_SOURCE_BPM";
const TARGET_BPM_ENV: &str = "TIMESTRETCH_RUBBERBAND_TARGET_BPM";
const MAX_SECONDS_ENV: &str = "TIMESTRETCH_RUBBERBAND_MAX_SECONDS";
const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;
const DEFAULT_MAX_SECONDS: f64 = 20.0;

fn output_dir() -> PathBuf {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));
    target_dir.join("rubberband_benchmark")
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok()?.parse::<f64>().ok()
}

fn trim_to_seconds(buffer: &AudioBuffer, seconds: f64) -> AudioBuffer {
    if seconds <= 0.0 {
        return buffer.clone();
    }
    let frames = ((seconds * buffer.sample_rate as f64).round() as usize).min(buffer.num_frames());
    buffer.slice(0, frames)
}

fn resolve_ratio(original: &AudioBuffer, reference: &AudioBuffer) -> Option<f64> {
    if let Some(ratio) = env_f64(RATIO_ENV) {
        return Some(ratio);
    }
    if let (Some(source_bpm), Some(target_bpm)) = (env_f64(SOURCE_BPM_ENV), env_f64(TARGET_BPM_ENV))
    {
        if target_bpm > 0.0 {
            return Some(source_bpm / target_bpm);
        }
    }
    if original.num_frames() > 0 {
        return Some(reference.num_frames() as f64 / original.num_frames() as f64);
    }
    None
}

#[test]
fn benchmark_against_external_rubberband_render() {
    let Some(original_path) = env_path(ORIG_WAV_ENV) else {
        println!(
            "Skipping Rubber Band comparison: {} is not set",
            ORIG_WAV_ENV
        );
        return;
    };
    let Some(reference_path) = env_path(REF_WAV_ENV) else {
        println!(
            "Skipping Rubber Band comparison: {} is not set",
            REF_WAV_ENV
        );
        return;
    };
    if !original_path.exists() {
        println!(
            "Skipping Rubber Band comparison: original file not found ({})",
            original_path.display()
        );
        return;
    }
    if !reference_path.exists() {
        println!(
            "Skipping Rubber Band comparison: reference file not found ({})",
            reference_path.display()
        );
        return;
    }

    let mut original = read_wav_file(original_path.to_string_lossy().as_ref())
        .expect("failed to read TIMESTRETCH_RUBBERBAND_ORIGINAL_WAV");
    let mut reference = read_wav_file(reference_path.to_string_lossy().as_ref())
        .expect("failed to read TIMESTRETCH_RUBBERBAND_REFERENCE_WAV");

    if reference.sample_rate != original.sample_rate {
        reference = reference.resample(original.sample_rate);
    }

    let Some(ratio) = resolve_ratio(&original, &reference) else {
        println!("Skipping Rubber Band comparison: failed to resolve a stretch ratio");
        return;
    };
    assert!(
        ratio.is_finite() && ratio > 0.01 && ratio < 100.0,
        "invalid stretch ratio resolved from env/reference lengths: {}",
        ratio
    );

    let max_seconds = env_f64(MAX_SECONDS_ENV).unwrap_or(DEFAULT_MAX_SECONDS);
    original = trim_to_seconds(&original, max_seconds);
    let mut reference_mono =
        trim_to_seconds(&reference, max_seconds * ratio.max(0.01)).mix_to_mono();
    let original_mono = original.mix_to_mono();

    let params = StretchParams::new(ratio)
        .with_sample_rate(original_mono.sample_rate)
        .with_channels(1)
        .with_preset(EdmPreset::DjBeatmatch);

    let stretched = timestretch::stretch(&original_mono.data, &params).expect("stretch failed");
    assert!(!stretched.is_empty(), "stretched output must not be empty");

    let compare_len = stretched.len().min(reference_mono.data.len());
    assert!(
        compare_len >= FFT_SIZE,
        "comparison window too short ({} samples), need at least {}",
        compare_len,
        FFT_SIZE
    );

    let test = &stretched[..compare_len];
    reference_mono.data.truncate(compare_len);
    let reference = &reference_mono.data;

    let report = comparison::generate_quality_report(
        test,
        reference,
        original_mono.sample_rate,
        FFT_SIZE,
        HOP_SIZE,
    );

    let out_dir = output_dir();
    fs::create_dir_all(&out_dir).expect("failed to create rubberband benchmark output directory");

    let output_wav = out_dir.join("timestretch_vs_rubberband_output.wav");
    let output_buffer = AudioBuffer::from_mono(test.to_vec(), original_mono.sample_rate);
    write_wav_file_float(output_wav.to_string_lossy().as_ref(), &output_buffer)
        .expect("failed to write benchmark output WAV");

    let csv_path = out_dir.join("rubberband_comparison_report.csv");
    let mut csv = BufWriter::new(File::create(&csv_path).expect("failed to create CSV report"));
    writeln!(
        csv,
        "scenario,ratio,sample_rate,spectral_similarity,perceptual_spectral_similarity,cross_correlation,transient_within_10ms,transient_total,lufs_difference,spectral_flux_similarity,overall_grade"
    )
    .expect("failed to write CSV header");
    writeln!(
        csv,
        "external_rubberband_render,{:.8},{},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{}",
        ratio,
        original_mono.sample_rate,
        report.spectral_similarity,
        report.perceptual_spectral_similarity,
        report.cross_correlation,
        report.onset_timing.within_10ms,
        report.onset_timing.total_onsets,
        report.lufs_difference,
        report.spectral_flux_similarity,
        report.overall_grade
    )
    .expect("failed to write CSV row");
    csv.flush().expect("failed to flush CSV report");

    assert!(
        report.spectral_similarity.is_finite()
            && report.perceptual_spectral_similarity.is_finite()
            && report.cross_correlation.is_finite(),
        "quality metrics must be finite; got spectral={:.6}, perceptual={:.6}, xcorr={:.6}",
        report.spectral_similarity,
        report.perceptual_spectral_similarity,
        report.cross_correlation
    );

    println!(
        "rubberband comparison: ratio={:.6}, spectral={:.3}, perceptual={:.3}, xcorr={:.3}, grade={}",
        ratio,
        report.spectral_similarity,
        report.perceptual_spectral_similarity,
        report.cross_correlation,
        report.overall_grade
    );
    println!("wrote: {}", csv_path.display());
}
