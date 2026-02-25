//! Reference audio quality benchmark.
//!
//! Compares library time-stretch output against professionally-stretched
//! reference audio files. Skips gracefully when audio files are not present.

use serde::{Deserialize, Serialize};
use std::path::Path;

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    track: Vec<Track>,
}

#[derive(Debug, Deserialize)]
struct Track {
    id: String,
    description: String,
    original: String,
    bpm: f64,
    #[serde(default)]
    reference: Vec<Reference>,
}

#[derive(Debug, Deserialize)]
struct Reference {
    file: String,
    target_bpm: f64,
    software: String,
    algorithm: String,
}

// ---------------------------------------------------------------------------
// Report types (JSON output)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Report {
    tracks: Vec<TrackReport>,
    summary: Summary,
}

#[derive(Debug, Serialize)]
struct TrackReport {
    id: String,
    description: String,
    bpm: f64,
    references: Vec<ReferenceReport>,
}

#[derive(Debug, Serialize)]
struct ReferenceReport {
    software: String,
    algorithm: String,
    target_bpm: f64,
    ratio: f64,
    presets: Vec<PresetReport>,
}

#[derive(Debug, Serialize)]
struct PresetReport {
    preset: String,
    length_diff_samples: isize,
    length_diff_pct: f64,
    rms_diff_db: f64,
    spectral_similarity: f64,
    band_similarity: BandSimilarityReport,
    transient_match_rate: f64,
    transient_matched: usize,
    transient_total: usize,
    cross_correlation_peak: f64,
    cross_correlation_offset: isize,
}

#[derive(Debug, Serialize)]
struct BandSimilarityReport {
    sub_bass: f64,
    low: f64,
    mid: f64,
    high: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    tracks_tested: usize,
    references_tested: usize,
    skipped: usize,
    average_spectral_similarity: f64,
    best_preset_per_track: Vec<BestPreset>,
}

#[derive(Debug, Serialize)]
struct BestPreset {
    track_id: String,
    preset: String,
    spectral_similarity: f64,
}

// ---------------------------------------------------------------------------
// Preset list
// ---------------------------------------------------------------------------

const ALL_PRESETS: &[(timestretch::EdmPreset, &str)] = &[
    (timestretch::EdmPreset::DjBeatmatch, "DjBeatmatch"),
    (timestretch::EdmPreset::HouseLoop, "HouseLoop"),
    (timestretch::EdmPreset::Halftime, "Halftime"),
    (timestretch::EdmPreset::Ambient, "Ambient"),
    (timestretch::EdmPreset::VocalChop, "VocalChop"),
];

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[test]
fn reference_quality_benchmark() {
    let manifest_path = Path::new("benchmarks/manifest.toml");
    if !manifest_path.exists() {
        println!("benchmarks/manifest.toml not found, skipping reference quality benchmark");
        return;
    }

    let manifest_str = std::fs::read_to_string(manifest_path).expect("Failed to read manifest");
    let manifest: Manifest = toml::from_str(&manifest_str).expect("Failed to parse manifest");

    if manifest.track.is_empty() {
        println!("No tracks in manifest, skipping reference quality benchmark");
        return;
    }

    let audio_base = Path::new("benchmarks/audio");
    let output_dir = audio_base.join("output");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let mut report = Report {
        tracks: Vec::new(),
        summary: Summary {
            tracks_tested: 0,
            references_tested: 0,
            skipped: 0,
            average_spectral_similarity: 0.0,
            best_preset_per_track: Vec::new(),
        },
    };

    let mut all_spectral_sims: Vec<f64> = Vec::new();

    println!("\n=== Reference Audio Quality Report ===\n");

    for track in &manifest.track {
        let original_path = audio_base.join(&track.original);
        if !original_path.exists() {
            println!(
                "Skipping track '{}': original file not found ({})",
                track.id,
                original_path.display()
            );
            report.summary.skipped += 1;
            continue;
        }

        let original =
            timestretch::io::wav::read_wav_file(original_path.to_str().expect("Invalid path"))
                .expect("Failed to read original WAV");

        println!(
            "Track: {} ({}, {} BPM, {:?})",
            track.id, track.description, track.bpm, original.channels
        );

        let mut track_report = TrackReport {
            id: track.id.clone(),
            description: track.description.clone(),
            bpm: track.bpm,
            references: Vec::new(),
        };

        let mut best_preset_name = String::new();
        let mut best_spectral = 0.0f64;

        for reference in &track.reference {
            let ref_path = audio_base.join(&reference.file);
            if !ref_path.exists() {
                println!(
                    "  Skipping reference: {} not found ({})",
                    reference.software,
                    ref_path.display()
                );
                report.summary.skipped += 1;
                continue;
            }

            let ref_audio =
                timestretch::io::wav::read_wav_file(ref_path.to_str().expect("Invalid path"))
                    .expect("Failed to read reference WAV");

            let ratio = track.bpm / reference.target_bpm;
            println!(
                "  vs. {} ({}) -> {} BPM (ratio {:.3})\n",
                reference.software, reference.algorithm, reference.target_bpm, ratio
            );

            let mut ref_report = ReferenceReport {
                software: reference.software.clone(),
                algorithm: reference.algorithm.clone(),
                target_bpm: reference.target_bpm,
                ratio,
                presets: Vec::new(),
            };

            for &(preset, preset_name) in ALL_PRESETS {
                let params = timestretch::StretchParams::new(ratio)
                    .with_sample_rate(original.sample_rate)
                    .with_channels(original.channels.count() as u32)
                    .with_preset(preset);

                let output = timestretch::stretch(&original.data, &params).expect("Stretch failed");

                // Write output WAV
                let out_filename = format!("{}_{}.wav", track.id, preset_name);
                let out_path = output_dir.join(&out_filename);
                let out_buf = timestretch::AudioBuffer::new(
                    output.clone(),
                    original.sample_rate,
                    original.channels,
                );
                timestretch::io::wav::write_wav_file_float(
                    out_path.to_str().expect("Invalid path"),
                    &out_buf,
                )
                .expect("Failed to write output WAV");

                // --- Compute metrics ---

                // Length accuracy
                let length_diff = output.len() as isize - ref_audio.data.len() as isize;
                let length_diff_pct = if !ref_audio.data.is_empty() {
                    (length_diff as f64 / ref_audio.data.len() as f64) * 100.0
                } else {
                    0.0
                };

                // RMS difference in dB
                let output_rms = rms(&output);
                let ref_rms = rms(&ref_audio.data);
                let rms_diff_db = if ref_rms > 1e-10 && output_rms > 1e-10 {
                    20.0 * (output_rms / ref_rms).log10()
                } else {
                    0.0
                };

                // Spectral similarity
                let spec_sim = timestretch::analysis::comparison::spectral_similarity(
                    &output,
                    &ref_audio.data,
                    2048,
                    512,
                );

                // Band spectral similarity
                let band_sim = timestretch::analysis::comparison::band_spectral_similarity(
                    &output,
                    &ref_audio.data,
                    2048,
                    512,
                    original.sample_rate,
                );

                // Transient match
                let transient_result = timestretch::analysis::comparison::transient_match_score(
                    &ref_audio.data,
                    &output,
                    original.sample_rate,
                    10.0,
                );

                // Cross-correlation (on a windowed segment to limit compute)
                let max_corr_samples = (original.sample_rate as usize * 10)
                    .min(output.len())
                    .min(ref_audio.data.len());
                let xcorr = timestretch::analysis::comparison::cross_correlation(
                    &output[..max_corr_samples],
                    &ref_audio.data[..max_corr_samples],
                );

                // Track best preset
                all_spectral_sims.push(spec_sim);
                if spec_sim > best_spectral {
                    best_spectral = spec_sim;
                    best_preset_name = preset_name.to_string();
                }

                // Console output
                println!("    Preset: {}", preset_name);
                println!(
                    "      Length accuracy:       {:+} samples ({:+.2}%)",
                    length_diff, length_diff_pct
                );
                println!("      RMS difference:        {:+.1} dB", rms_diff_db);
                println!("      Spectral similarity:   {:.3}", spec_sim);
                println!(
                    "        Sub-bass: {:.3}  Low: {:.3}  Mid: {:.3}  High: {:.3}",
                    band_sim.sub_bass, band_sim.low, band_sim.mid, band_sim.high
                );
                println!(
                    "      Transient match rate:  {:.1}% ({}/{} matched)",
                    transient_result.match_rate * 100.0,
                    transient_result.matched,
                    transient_result.total_reference
                );
                println!(
                    "      Cross-correlation:     {:.3} (offset: {:+} samples)\n",
                    xcorr.peak_value, xcorr.peak_offset
                );

                ref_report.presets.push(PresetReport {
                    preset: preset_name.to_string(),
                    length_diff_samples: length_diff,
                    length_diff_pct,
                    rms_diff_db,
                    spectral_similarity: spec_sim,
                    band_similarity: BandSimilarityReport {
                        sub_bass: band_sim.sub_bass,
                        low: band_sim.low,
                        mid: band_sim.mid,
                        high: band_sim.high,
                    },
                    transient_match_rate: transient_result.match_rate,
                    transient_matched: transient_result.matched,
                    transient_total: transient_result.total_reference,
                    cross_correlation_peak: xcorr.peak_value,
                    cross_correlation_offset: xcorr.peak_offset,
                });
            }

            report.summary.references_tested += 1;
            track_report.references.push(ref_report);
        }

        if !best_preset_name.is_empty() {
            report.summary.best_preset_per_track.push(BestPreset {
                track_id: track.id.clone(),
                preset: best_preset_name.clone(),
                spectral_similarity: best_spectral,
            });
        }

        report.summary.tracks_tested += 1;
        report.tracks.push(track_report);
        println!();
    }

    // Summary
    let avg_spectral = if !all_spectral_sims.is_empty() {
        all_spectral_sims.iter().sum::<f64>() / all_spectral_sims.len() as f64
    } else {
        0.0
    };
    report.summary.average_spectral_similarity = avg_spectral;

    println!("=== Summary ===");
    println!("Best preset per track:");
    for bp in &report.summary.best_preset_per_track {
        println!(
            "  {}: {} (spectral: {:.3})",
            bp.track_id, bp.preset, bp.spectral_similarity
        );
    }
    println!("Average spectral similarity: {:.3}", avg_spectral);
    println!(
        "Tracks tested: {}, References tested: {}, Skipped: {}",
        report.summary.tracks_tested, report.summary.references_tested, report.summary.skipped
    );

    // Write JSON report
    let json = serde_json::to_string_pretty(&report).expect("Failed to serialize report");
    let report_path = output_dir.join("report.json");
    std::fs::write(&report_path, &json).expect("Failed to write report JSON");
    println!("\nJSON report written to: {}\n", report_path.display());
}

/// Computes RMS of a signal.
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}
