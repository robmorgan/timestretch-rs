//! Reference audio quality benchmark.
//!
//! Compares library time-stretch output against professionally-stretched
//! reference audio files.
//!
//! By default, this benchmark skips when the corpus is unavailable so regular
//! CI can run without copyrighted assets. Set
//! `TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1` to require a fully configured
//! corpus and fail on any missing file/checksum.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// Environment variable that enables strict corpus validation.
const STRICT_ENV_VAR: &str = "TIMESTRETCH_STRICT_REFERENCE_BENCHMARK";
/// Optional environment variable to limit analyzed audio duration per file.
const MAX_SECONDS_ENV_VAR: &str = "TIMESTRETCH_REFERENCE_MAX_SECONDS";

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
    #[serde(default)]
    original_sha256: Option<String>,
    bpm: f64,
    #[serde(default)]
    reference: Vec<Reference>,
}

#[derive(Debug, Deserialize)]
struct Reference {
    file: String,
    #[serde(default)]
    file_sha256: Option<String>,
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
    perceptual_spectral_similarity: f64,
    band_similarity: BandSimilarityReport,
    bark_band_similarity: BarkBandSimilarityReport,
    transient_match_rate: f64,
    transient_matched: usize,
    transient_total: usize,
    onset_timing: OnsetTimingReport,
    cross_correlation_peak: f64,
    cross_correlation_offset: isize,
    lufs_difference: f64,
    spectral_flux_similarity: f64,
    overall_grade: String,
}

#[derive(Debug, Serialize)]
struct BandSimilarityReport {
    sub_bass: f64,
    low: f64,
    mid: f64,
    high: f64,
}

#[derive(Debug, Serialize)]
struct BarkBandSimilarityReport {
    sub_bass: f64,
    bass: f64,
    low_mid: f64,
    mid: f64,
    upper_mid: f64,
    presence: f64,
    brilliance: f64,
    air: f64,
    overall: f64,
}

#[derive(Debug, Serialize)]
struct OnsetTimingReport {
    mean_error_ms: f64,
    median_error_ms: f64,
    std_dev_ms: f64,
    max_error_ms: f64,
    within_5ms: usize,
    within_10ms: usize,
    within_20ms: usize,
    total_onsets: usize,
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
    let strict = strict_benchmark_mode();
    let max_seconds = benchmark_max_seconds();
    let manifest_path = Path::new("benchmarks/manifest.toml");
    if !manifest_path.exists() {
        if strict {
            panic!("benchmarks/manifest.toml not found in strict mode");
        }
        println!(
            "benchmarks/manifest.toml not found, skipping reference quality benchmark (strict mode disabled)"
        );
        return;
    }

    let manifest_str = std::fs::read_to_string(manifest_path).expect("Failed to read manifest");
    let manifest: Manifest = toml::from_str(&manifest_str).expect("Failed to parse manifest");

    if manifest.track.is_empty() {
        if strict {
            panic!("No tracks in manifest in strict mode");
        }
        println!(
            "No tracks in manifest, skipping reference quality benchmark (strict mode disabled)"
        );
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

    println!(
        "\n=== Reference Audio Quality Report ==={}",
        if strict { " (strict mode)" } else { "" }
    );
    if let Some(secs) = max_seconds {
        println!(
            "Using benchmark window: first {:.1}s per original/reference file",
            secs
        );
    }
    println!();

    for track in &manifest.track {
        if track.reference.is_empty() {
            if strict {
                panic!(
                    "Track '{}' has no references configured in strict mode",
                    track.id
                );
            }
            println!("Skipping track '{}': no references configured", track.id);
            report.summary.skipped += 1;
            continue;
        }

        let original_path = match resolve_audio_path(audio_base, &track.original) {
            Ok(path) => path,
            Err(msg) => {
                if strict {
                    panic!("Invalid original path for track '{}': {}", track.id, msg);
                }
                println!("Skipping track '{}': {}", track.id, msg);
                report.summary.skipped += 1;
                continue;
            }
        };
        if !original_path.exists() {
            if strict {
                panic!(
                    "Track '{}' original file not found in strict mode ({})",
                    track.id,
                    original_path.display()
                );
            }
            println!(
                "Skipping track '{}': original file not found ({})",
                track.id,
                original_path.display()
            );
            report.summary.skipped += 1;
            continue;
        }
        validate_sha256(
            &original_path,
            track.original_sha256.as_deref(),
            &format!("track '{}' original", track.id),
            strict,
        )
        .unwrap_or_else(|msg| panic!("{}", msg));

        let original =
            timestretch::io::wav::read_wav_file(original_path.to_str().expect("Invalid path"))
                .expect("Failed to read original WAV");
        let original_data = maybe_trim_interleaved(
            &original.data,
            original.sample_rate,
            original.channels.count(),
            max_seconds,
        );

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
            let ref_path = match resolve_audio_path(audio_base, &reference.file) {
                Ok(path) => path,
                Err(msg) => {
                    if strict {
                        panic!(
                            "Invalid reference path for track '{}' ({}): {}",
                            track.id, reference.software, msg
                        );
                    }
                    println!("  Skipping reference: {} ({})", reference.software, msg);
                    report.summary.skipped += 1;
                    continue;
                }
            };
            if !ref_path.exists() {
                if strict {
                    panic!(
                        "Reference '{}' for track '{}' not found in strict mode ({})",
                        reference.software,
                        track.id,
                        ref_path.display()
                    );
                }
                println!(
                    "  Skipping reference: {} not found ({})",
                    reference.software,
                    ref_path.display()
                );
                report.summary.skipped += 1;
                continue;
            }
            validate_sha256(
                &ref_path,
                reference.file_sha256.as_deref(),
                &format!("track '{}' reference '{}'", track.id, reference.software),
                strict,
            )
            .unwrap_or_else(|msg| panic!("{}", msg));

            let ref_audio =
                timestretch::io::wav::read_wav_file(ref_path.to_str().expect("Invalid path"))
                    .expect("Failed to read reference WAV");
            let ref_data = maybe_trim_interleaved(
                &ref_audio.data,
                ref_audio.sample_rate,
                ref_audio.channels.count(),
                max_seconds,
            );

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

                let output = timestretch::stretch(&original_data, &params).expect("Stretch failed");

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
                let length_diff = output.len() as isize - ref_data.len() as isize;
                let length_diff_pct = if !ref_data.is_empty() {
                    (length_diff as f64 / ref_data.len() as f64) * 100.0
                } else {
                    0.0
                };

                // RMS difference in dB
                let output_rms = rms(&output);
                let ref_rms = rms(&ref_data);
                let rms_diff_db = if ref_rms > 1e-10 && output_rms > 1e-10 {
                    20.0 * (output_rms / ref_rms).log10()
                } else {
                    0.0
                };

                // Spectral similarity (unweighted)
                let spec_sim = timestretch::analysis::comparison::spectral_similarity(
                    &output, &ref_data, 2048, 512,
                );

                // Perceptual spectral similarity (A-weighted)
                let perc_sim = timestretch::analysis::comparison::perceptual_spectral_similarity(
                    &output,
                    &ref_data,
                    2048,
                    512,
                    original.sample_rate,
                );

                // Band spectral similarity (EDM bands)
                let band_sim = timestretch::analysis::comparison::band_spectral_similarity(
                    &output,
                    &ref_data,
                    2048,
                    512,
                    original.sample_rate,
                );

                // Bark-scale band similarity
                let bark_sim = timestretch::analysis::comparison::bark_band_similarity(
                    &output,
                    &ref_data,
                    2048,
                    512,
                    original.sample_rate,
                );

                // Transient match
                let transient_result = timestretch::analysis::comparison::transient_match_score(
                    &ref_data,
                    &output,
                    original.sample_rate,
                    10.0,
                );

                // Onset timing analysis
                let onset_timing = timestretch::analysis::comparison::onset_timing_analysis(
                    &ref_data,
                    &output,
                    original.sample_rate,
                );

                // Cross-correlation (on a windowed segment to limit compute)
                let max_corr_samples = (original.sample_rate as usize * 10)
                    .min(output.len())
                    .min(ref_data.len());
                let xcorr = timestretch::analysis::comparison::cross_correlation(
                    &output[..max_corr_samples],
                    &ref_data[..max_corr_samples],
                );

                // LUFS loudness difference
                let lufs_diff = timestretch::analysis::comparison::lufs_difference(
                    &output,
                    &ref_data,
                    original.sample_rate,
                );

                // Spectral flux similarity
                let flux_sim = timestretch::analysis::comparison::spectral_flux_similarity(
                    &output, &ref_data, 2048, 512,
                );

                // Generate overall quality report for grade
                let quality_report = timestretch::analysis::comparison::generate_quality_report(
                    &output,
                    &ref_data,
                    original.sample_rate,
                    2048,
                    512,
                );

                // Track best preset
                all_spectral_sims.push(spec_sim);
                if spec_sim > best_spectral {
                    best_spectral = spec_sim;
                    best_preset_name = preset_name.to_string();
                }

                // Console output
                println!(
                    "    Preset: {} (Grade: {})",
                    preset_name, quality_report.overall_grade
                );
                println!(
                    "      Length accuracy:            {:+} samples ({:+.2}%)",
                    length_diff, length_diff_pct
                );
                println!("      RMS difference:            {:+.1} dB", rms_diff_db);
                println!("      LUFS difference:           {:+.2} dB", lufs_diff);
                println!("      Spectral similarity:       {:.3}", spec_sim);
                println!(
                    "      Perceptual similarity:     {:.3} (A-weighted)",
                    perc_sim
                );
                println!(
                    "        Sub-bass: {:.3}  Low: {:.3}  Mid: {:.3}  High: {:.3}",
                    band_sim.sub_bass, band_sim.low, band_sim.mid, band_sim.high
                );
                println!("      Bark bands (overall {:.3}):", bark_sim.overall);
                for i in 0..timestretch::analysis::comparison::BARK_BAND_COUNT {
                    println!(
                        "        {}: {:.3}",
                        timestretch::analysis::comparison::BARK_BAND_NAMES[i],
                        bark_sim.bands[i]
                    );
                }
                println!(
                    "      Transient match rate:      {:.1}% ({}/{} matched)",
                    transient_result.match_rate * 100.0,
                    transient_result.matched,
                    transient_result.total_reference
                );
                if onset_timing.total_onsets > 0 {
                    println!(
                        "      Onset timing:              mean={:.1}ms median={:.1}ms std={:.1}ms max={:.1}ms",
                        onset_timing.mean_error_ms,
                        onset_timing.median_error_ms,
                        onset_timing.std_dev_ms,
                        onset_timing.max_error_ms,
                    );
                    println!(
                        "        Within 5ms: {}/{}  10ms: {}/{}  20ms: {}/{}",
                        onset_timing.within_5ms,
                        onset_timing.total_onsets,
                        onset_timing.within_10ms,
                        onset_timing.total_onsets,
                        onset_timing.within_20ms,
                        onset_timing.total_onsets,
                    );
                }
                println!(
                    "      Cross-correlation:         {:.3} (offset: {:+} samples)",
                    xcorr.peak_value, xcorr.peak_offset
                );
                println!("      Spectral flux similarity:  {:.3}\n", flux_sim);

                ref_report.presets.push(PresetReport {
                    preset: preset_name.to_string(),
                    length_diff_samples: length_diff,
                    length_diff_pct,
                    rms_diff_db,
                    spectral_similarity: spec_sim,
                    perceptual_spectral_similarity: perc_sim,
                    band_similarity: BandSimilarityReport {
                        sub_bass: band_sim.sub_bass,
                        low: band_sim.low,
                        mid: band_sim.mid,
                        high: band_sim.high,
                    },
                    bark_band_similarity: BarkBandSimilarityReport {
                        sub_bass: bark_sim.bands[0],
                        bass: bark_sim.bands[1],
                        low_mid: bark_sim.bands[2],
                        mid: bark_sim.bands[3],
                        upper_mid: bark_sim.bands[4],
                        presence: bark_sim.bands[5],
                        brilliance: bark_sim.bands[6],
                        air: bark_sim.bands[7],
                        overall: bark_sim.overall,
                    },
                    transient_match_rate: transient_result.match_rate,
                    transient_matched: transient_result.matched,
                    transient_total: transient_result.total_reference,
                    onset_timing: OnsetTimingReport {
                        mean_error_ms: onset_timing.mean_error_ms,
                        median_error_ms: onset_timing.median_error_ms,
                        std_dev_ms: onset_timing.std_dev_ms,
                        max_error_ms: onset_timing.max_error_ms,
                        within_5ms: onset_timing.within_5ms,
                        within_10ms: onset_timing.within_10ms,
                        within_20ms: onset_timing.within_20ms,
                        total_onsets: onset_timing.total_onsets,
                    },
                    cross_correlation_peak: xcorr.peak_value,
                    cross_correlation_offset: xcorr.peak_offset,
                    lufs_difference: lufs_diff,
                    spectral_flux_similarity: flux_sim,
                    overall_grade: quality_report.overall_grade.to_string(),
                });
            }

            report.summary.references_tested += 1;
            track_report.references.push(ref_report);
        }

        if strict && track_report.references.is_empty() {
            panic!(
                "Track '{}' had zero valid references in strict mode",
                track.id
            );
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

    if strict {
        assert!(
            report.summary.tracks_tested > 0,
            "Strict mode requires at least one tested track"
        );
        assert!(
            report.summary.references_tested > 0,
            "Strict mode requires at least one tested reference"
        );
        assert_eq!(
            report.summary.skipped, 0,
            "Strict mode does not allow skipped tracks/references"
        );
    }

    // Write JSON report
    let json = serde_json::to_string_pretty(&report).expect("Failed to serialize report");
    let report_path = output_dir.join("report.json");
    std::fs::write(&report_path, &json).expect("Failed to write report JSON");
    println!("\nJSON report written to: {}\n", report_path.display());
}

fn strict_benchmark_mode() -> bool {
    let value = std::env::var(STRICT_ENV_VAR).unwrap_or_default();
    let normalized = value.trim().to_ascii_lowercase();
    !normalized.is_empty() && normalized != "0" && normalized != "false" && normalized != "no"
}

fn benchmark_max_seconds() -> Option<f64> {
    let value = std::env::var(MAX_SECONDS_ENV_VAR).ok()?;
    let parsed = value.trim().parse::<f64>().ok()?;
    (parsed.is_finite() && parsed > 0.0).then_some(parsed)
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

/// Computes RMS of a signal.
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}
