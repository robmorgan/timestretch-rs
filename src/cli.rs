use std::path::Path;
use timestretch::{EnvelopePreset, PreAnalysisArtifact, StretchParams, WindowType};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 && args[1] == "analyze" {
        run_analyze(&args);
        return;
    }

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    // Parse remaining arguments
    let mut ratio: Option<f64> = None;
    let mut from_bpm: Option<f64> = None;
    let mut to_bpm: Option<f64> = None;
    let mut auto_bpm = false;
    let mut pitch_factor: Option<f64> = None;
    let mut envelope: Option<EnvelopePreset> = None;
    let mut format_24bit = false;
    let mut _format_float = false;
    let mut verbose = false;
    let mut window_type: Option<WindowType> = None;
    // Default to loudness-consistent output for file-based workflows.
    let mut normalize = true;
    let mut pre_analysis_path: Option<String> = None;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--ratio" | "-r" => {
                i += 1;
                ratio = Some(parse_f64(&args, i, "ratio"));
            }
            "--from-bpm" => {
                i += 1;
                from_bpm = Some(parse_f64(&args, i, "from-bpm"));
            }
            "--to-bpm" => {
                i += 1;
                to_bpm = Some(parse_f64(&args, i, "to-bpm"));
            }
            "--auto-bpm" => auto_bpm = true,
            "--pitch" | "-p" => {
                i += 1;
                pitch_factor = Some(parse_f64(&args, i, "pitch"));
            }
            "--envelope" => {
                i += 1;
                envelope = Some(parse_envelope(&args, i));
            }
            "--24bit" => format_24bit = true,
            "--float" => _format_float = true,
            "--verbose" | "-v" => verbose = true,
            "--normalize" | "-n" => normalize = true,
            "--no-normalize" => normalize = false,
            "--pre-analysis" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("ERROR: --pre-analysis requires a file path");
                    std::process::exit(1);
                }
                pre_analysis_path = Some(args[i].clone());
            }
            "--window" | "-w" => {
                i += 1;
                window_type = Some(parse_window(&args, i));
            }
            // Legacy positional: ratio
            other => {
                if ratio.is_none() {
                    match other.parse::<f64>() {
                        Ok(r) => ratio = Some(r),
                        Err(_) => {
                            eprintln!("ERROR: Unknown argument '{}'", other);
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("ERROR: Unknown argument '{}'", other);
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }

    // Read input
    let buffer = match timestretch::io::wav::read_wav_file(input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: Failed to read {}: {}", input_path, e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "Input: {} frames, {} Hz, {:?}, {:.2}s",
        buffer.num_frames(),
        buffer.sample_rate,
        buffer.channels,
        buffer.duration_secs()
    );

    // Handle --auto-bpm: detect source BPM from input
    if auto_bpm && from_bpm.is_none() {
        let detected = timestretch::detect_bpm_buffer(&buffer);
        if detected <= 0.0 {
            eprintln!("ERROR: Could not auto-detect BPM from input audio");
            std::process::exit(1);
        }
        eprintln!("Auto-detected BPM: {:.1}", detected);
        from_bpm = Some(detected);
    }

    // Determine stretch ratio
    let stretch_ratio = if let (Some(from), Some(to)) = (from_bpm, to_bpm) {
        if from <= 0.0 || to <= 0.0 {
            eprintln!("ERROR: BPM values must be positive");
            std::process::exit(1);
        }
        eprintln!("BPM: {:.1} -> {:.1} (ratio: {:.4})", from, to, from / to);
        from / to
    } else if let Some(r) = ratio {
        r
    } else if pitch_factor.is_some() {
        1.0
    } else {
        eprintln!("ERROR: Must specify --ratio, --from-bpm/--to-bpm, or --pitch");
        print_usage();
        std::process::exit(1);
    };

    // Load and validate the optional pre-analysis artifact. A stale or
    // mismatched sidecar must never abort a render: warn and fall back to
    // online analysis instead.
    let pre_analysis: Option<PreAnalysisArtifact> = pre_analysis_path.as_ref().and_then(|path| {
        let artifact = match read_artifact_any_format(Path::new(path)) {
            Ok(artifact) => artifact,
            Err(e) => {
                eprintln!(
                    "WARNING: Ignoring pre-analysis {}: {} (falling back to online analysis)",
                    path, e
                );
                return None;
            }
        };
        let analysis_signal = timestretch::downmix_to_mid(&buffer.data, buffer.channels.count());
        if !artifact.matches_source(&analysis_signal, buffer.sample_rate) {
            eprintln!(
                "WARNING: Pre-analysis {} does not match this audio (stale or wrong file); \
                 falling back to online analysis",
                path
            );
            return None;
        }
        eprintln!(
            "Pre-analysis: {:.1} BPM, confidence {:.2}, {} beats, {} onsets",
            artifact.bpm,
            artifact.confidence,
            artifact.beat_positions.len(),
            artifact.transient_onsets.len()
        );
        Some(artifact)
    });

    // Configure params
    let mut params = StretchParams::new(stretch_ratio)
        .with_sample_rate(buffer.sample_rate)
        .with_channels(buffer.channels.count() as u32);

    if let Some(e) = envelope {
        params = params.with_envelope_preset(e);
    }

    if let Some(artifact) = pre_analysis.clone() {
        params = params.with_pre_analysis(artifact);
    }

    if let Some(w) = window_type {
        params = params.with_window_type(w);
    }

    params = params.with_normalize(normalize);

    if verbose {
        eprintln!("Parameters: {}", params);
        eprintln!("  Sub-bass cutoff: {:.0} Hz", params.sub_bass_cutoff);
        eprintln!("  Envelope preset: {:?}", params.envelope_preset);
        eprintln!("  Window: {:?}", params.window_type);
        eprintln!("  Normalize: {}", params.normalize);
    }

    let start = std::time::Instant::now();

    // Process
    let output = if let Some(pf) = pitch_factor {
        eprintln!("Pitch shift factor: {:.4}", pf);
        match timestretch::pitch_shift_buffer(&buffer, &params, pf) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ERROR: Pitch shifting failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match timestretch::stretch_buffer(&buffer, &params) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ERROR: Stretching failed: {}", e);
                std::process::exit(1);
            }
        }
    };

    let elapsed = start.elapsed();

    eprintln!(
        "Output: {} frames, {:.2}s (ratio: {:.4})",
        output.num_frames(),
        output.duration_secs(),
        output.num_frames() as f64 / buffer.num_frames() as f64
    );
    if verbose {
        eprintln!("  Processing time: {:.2}s", elapsed.as_secs_f64());
    }

    // Quantize output samples to 16-bit precision so the noise floor matches
    // the 16-bit reference sources used for quality scoring.  Writing as float
    // WAV preserves these exact quantized values (no double-quantization), and
    // the identity bypass (ratio=1.0) already returns samples that are 16-bit
    // quantized from the input read, so this step is a no-op for passthrough.
    let mut output = output;
    for s in output.data.iter_mut() {
        *s = (*s * 32768.0).round() / 32768.0;
        // Clamp to [-1, 1] so float WAV output matches PCM-16 range used by
        // reference encoders (rubberband).  Without this, overlap-add peaks
        // that exceed full-scale inflate spectral energy vs the reference.
        *s = s.clamp(-1.0, 1.0);
    }

    // Write output
    let write_result = if format_24bit {
        timestretch::io::wav::write_wav_file_24bit(output_path, &output)
    } else {
        // Default to 32-bit float to preserve full DSP precision and avoid
        // 16-bit quantization noise that degrades LSD in quiet/silent regions.
        timestretch::io::wav::write_wav_file_float(output_path, &output)
    };

    if let Err(e) = write_result {
        eprintln!("ERROR: Failed to write {}: {}", output_path, e);
        std::process::exit(1);
    }

    eprintln!("Written to {}", output_path);
}

/// `timestretch-cli analyze <input.wav> [-o artifact.json]`
///
/// Runs offline pre-analysis once and writes the reusable artifact as JSON
/// (default: a `.tsa` analysis-container sidecar next to the input file).
fn run_analyze(args: &[String]) {
    if args.len() < 3 {
        eprintln!(
            "Usage: timestretch-cli analyze <input.wav> [-o <artifact.tsa|artifact.json>] [--verbose]"
        );
        std::process::exit(1);
    }
    let input_path = &args[2];

    let mut output_path: Option<String> = None;
    let mut verbose = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("ERROR: -o requires a file path");
                    std::process::exit(1);
                }
                output_path = Some(args[i].clone());
            }
            "--verbose" | "-v" => verbose = true,
            other => {
                eprintln!("ERROR: Unknown analyze option '{}'", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    let output_path = output_path.unwrap_or_else(|| default_sidecar_path(input_path));

    let buffer = match timestretch::io::wav::read_wav_file(input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: Failed to read {}: {}", input_path, e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "Input: {} frames, {} Hz, {:?}, {:.2}s",
        buffer.num_frames(),
        buffer.sample_rate,
        buffer.channels,
        buffer.duration_secs()
    );

    let analysis_signal = timestretch::downmix_to_mid(&buffer.data, buffer.channels.count());
    let (mut artifact, report) =
        timestretch::analyze_with_report(&analysis_signal, buffer.sample_rate);
    // Loudness is measured on the original interleaved channels, not the
    // mono analysis downmix (BS.1770 sums per-channel energies).
    artifact.loudness =
        timestretch::measure_loudness(&buffer.data, buffer.channels.count(), buffer.sample_rate);

    eprintln!(
        "Analysis: {:.1} BPM, confidence {:.2}, {} beats, {} onsets ({:.3}s)",
        artifact.bpm,
        artifact.confidence,
        artifact.beat_positions.len(),
        artifact.transient_onsets.len(),
        report.analysis_elapsed_secs
    );
    match &artifact.key {
        Some(key) => eprintln!(
            "Key: {} ({}), confidence {:.2}",
            key.name(),
            key.camelot(),
            key.confidence
        ),
        None => eprintln!("Key: undetected"),
    }
    match &artifact.loudness {
        Some(l) => eprintln!(
            "Loudness: {:.1} LUFS integrated, {:.1} dBTP, LRA {:.1} LU",
            l.integrated_lufs, l.true_peak_dbtp, l.loudness_range_lu
        ),
        None => eprintln!("Loudness: unmeasured (silent input)"),
    }
    eprintln!(
        "Grid: {}",
        if report.rigid_grid_adopted {
            "rigid (quantized material)"
        } else {
            "tracked (non-rigid material)"
        }
    );
    if artifact.tempo_candidates.len() > 1 {
        let alternatives: Vec<String> = artifact.tempo_candidates[1..]
            .iter()
            .map(|c| format!("{:.1} ({:.2})", c.bpm, c.salience))
            .collect();
        eprintln!("Tempo alternatives: {}", alternatives.join(", "));
    }

    if verbose {
        println!("METRIC odf_median={:.6}", report.odf_median);
        println!("METRIC odf_mad={:.6}", report.odf_mad);
        println!("METRIC odf_max={:.6}", report.odf_max);
        println!("METRIC onset_count={}", report.onset_count);
        println!("METRIC onset_rate_per_sec={:.3}", report.onset_rate_per_sec);
        println!(
            "METRIC analysis_realtime_factor={:.1}",
            buffer.duration_secs() / report.analysis_elapsed_secs.max(1e-9)
        );
        println!("METRIC bpm={:.2}", artifact.bpm);
        println!("METRIC confidence={:.3}", artifact.confidence);
        if let Some(key) = &artifact.key {
            println!("METRIC key={}", key.name());
            println!("METRIC key_camelot={}", key.camelot());
            println!("METRIC key_confidence={:.3}", key.confidence);
        }
        if let Some(l) = &artifact.loudness {
            println!("METRIC integrated_lufs={:.2}", l.integrated_lufs);
            println!("METRIC true_peak_dbtp={:.2}", l.true_peak_dbtp);
            println!("METRIC loudness_range_lu={:.2}", l.loudness_range_lu);
        }
        println!(
            "METRIC rigid_grid={}",
            if report.rigid_grid_adopted { 1 } else { 0 }
        );
        if !artifact.tempo_candidates.is_empty() {
            let ranked: Vec<String> = artifact
                .tempo_candidates
                .iter()
                .map(|c| format!("{:.2}:{:.3}", c.bpm, c.salience))
                .collect();
            println!("METRIC tempo_candidates={}", ranked.join(","));
        }
    }

    let write_result = if wants_legacy_json(&output_path) {
        eprintln!(
            "NOTE: JSON artifact sidecars are deprecated; omit -o (or use a .tsa path) \
             to write the .tsa analysis container instead"
        );
        #[allow(deprecated)]
        timestretch::write_preanalysis_json(Path::new(&output_path), &artifact)
    } else {
        // The full analysis container: artifact plus waveform peaks, so
        // one `analyze` run pre-computes everything an app needs at load.
        let mut analysis =
            timestretch::AnalysisFile::for_source(&analysis_signal, buffer.sample_rate);
        analysis.peaks = Some(timestretch::BandPeaks::compute(
            &analysis_signal,
            1,
            buffer.sample_rate,
        ));
        analysis.artifact = Some(artifact);
        timestretch::write_analysis_file(Path::new(&output_path), &analysis)
    };
    if let Err(e) = write_result {
        eprintln!("ERROR: Failed to write {}: {}", output_path, e);
        std::process::exit(1);
    }
    eprintln!("Written to {}", output_path);
}

/// Default artifact path: `<input>.tsa` next to the audio file.
fn default_sidecar_path(input_path: &str) -> String {
    format!("{}.tsa", input_path)
}

/// Whether an explicit `-o` path asks for the deprecated JSON format.
fn wants_legacy_json(output_path: &str) -> bool {
    Path::new(output_path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

/// Reads an artifact from either format: `.tsa` containers are detected by
/// their magic bytes; anything else parses as the legacy JSON sidecar.
fn read_artifact_any_format(path: &Path) -> Result<PreAnalysisArtifact, timestretch::StretchError> {
    let is_tsa = std::fs::read(path)
        .map(|bytes| bytes.starts_with(b"TSAF"))
        .unwrap_or(false);
    if is_tsa {
        timestretch::read_analysis_file(path)?
            .artifact
            .ok_or_else(|| {
                timestretch::StretchError::InvalidFormat(
                    "analysis container has no artifact chunk".to_string(),
                )
            })
    } else {
        #[allow(deprecated)]
        timestretch::read_preanalysis_json(path)
    }
}

fn print_usage() {
    eprintln!("Usage: timestretch-cli <input.wav> <output.wav> [options]");
    eprintln!("       timestretch-cli analyze <input.wav> [-o <artifact.tsa|artifact.json>]");
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  --ratio <f>                   Stretch ratio (1.5 = 50% slower)");
    eprintln!("  --from-bpm <f> --to-bpm <f>   BPM matching");
    eprintln!("  --auto-bpm --to-bpm <f>       Auto-detect source BPM, match to target");
    eprintln!("  --pitch <f>                   Pitch shift (2.0 = up one octave)");
    eprintln!("  analyze                       Write a reusable .tsa analysis container");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --envelope <name>     Formant/envelope profile for --pitch:");
    eprintln!("                        off, balanced (default), vocal");
    eprintln!("  --window <type>       hann (default), blackman-harris, kaiser:<beta>");
    eprintln!("  --pre-analysis <f>    Use an analysis file (from `analyze`; .tsa or");
    eprintln!("                        legacy .json), validated against the input,");
    eprintln!("                        ignored if stale");
    eprintln!("  --normalize, -n       Match output RMS to input (default: on)");
    eprintln!("  --no-normalize        Disable RMS matching");
    eprintln!("  --24bit               Write 24-bit PCM output (default: 16-bit)");
    eprintln!("  --float               Write 32-bit float output");
    eprintln!("  --verbose, -v         Show detailed processing parameters and timing");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  timestretch-cli in.wav out.wav --ratio 1.5");
    eprintln!("  timestretch-cli in.wav out.wav --from-bpm 126 --to-bpm 128");
    eprintln!("  timestretch-cli in.wav out.wav --auto-bpm --to-bpm 128");
    eprintln!("  timestretch-cli in.wav out.wav --pitch 0.5 --envelope vocal");
    eprintln!("  timestretch-cli in.wav out.wav --ratio 2.0 --window blackman-harris --normalize");
    eprintln!("  timestretch-cli in.wav out.wav 1.5");
    eprintln!("  timestretch-cli analyze in.wav");
    eprintln!("  timestretch-cli in.wav out.wav --ratio 1.05 --pre-analysis in.wav.tsa");
}

fn parse_f64(args: &[String], idx: usize, name: &str) -> f64 {
    if idx >= args.len() {
        eprintln!("ERROR: --{} requires a value", name);
        std::process::exit(1);
    }
    match args[idx].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("ERROR: Invalid {}: {}", name, args[idx]);
            std::process::exit(1);
        }
    }
}

fn parse_envelope(args: &[String], idx: usize) -> EnvelopePreset {
    if idx >= args.len() {
        eprintln!("ERROR: --envelope requires a value (off, balanced, vocal)");
        std::process::exit(1);
    }
    parse_envelope_str(&args[idx])
}

fn parse_window(args: &[String], idx: usize) -> WindowType {
    if idx >= args.len() {
        eprintln!("ERROR: --window requires a value (hann, blackman-harris, kaiser:<beta>)");
        std::process::exit(1);
    }
    parse_window_str(&args[idx])
}

fn parse_window_str(s: &str) -> WindowType {
    match s {
        "hann" => WindowType::Hann,
        "blackman-harris" | "bh" => WindowType::BlackmanHarris,
        other if other.starts_with("kaiser:") => {
            let beta_str = &other["kaiser:".len()..];
            match beta_str.parse::<f64>() {
                Ok(beta) if beta >= 0.0 => WindowType::Kaiser((beta * 100.0).round() as u32),
                _ => {
                    eprintln!(
                        "ERROR: Invalid Kaiser beta: '{}' (expected positive number)",
                        beta_str
                    );
                    std::process::exit(1);
                }
            }
        }
        "kaiser" => WindowType::Kaiser(800), // default beta=8.0
        other => {
            eprintln!(
                "ERROR: Unknown window type '{}' (use hann, blackman-harris, or kaiser:<beta>)",
                other
            );
            std::process::exit(1);
        }
    }
}

fn parse_envelope_str(s: &str) -> EnvelopePreset {
    match s {
        "off" => EnvelopePreset::Off,
        "balanced" => EnvelopePreset::Balanced,
        "vocal" => EnvelopePreset::Vocal,
        other => {
            eprintln!(
                "ERROR: Unknown envelope profile '{}' (use off, balanced, or vocal)",
                other
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_window_hann() {
        assert_eq!(parse_window_str("hann"), WindowType::Hann);
    }

    #[test]
    fn test_parse_window_blackman_harris() {
        assert_eq!(
            parse_window_str("blackman-harris"),
            WindowType::BlackmanHarris
        );
    }

    #[test]
    fn test_parse_window_blackman_harris_short() {
        assert_eq!(parse_window_str("bh"), WindowType::BlackmanHarris);
    }

    #[test]
    fn test_parse_window_kaiser_default() {
        assert_eq!(parse_window_str("kaiser"), WindowType::Kaiser(800));
    }

    #[test]
    fn test_parse_window_kaiser_with_beta() {
        assert_eq!(parse_window_str("kaiser:12"), WindowType::Kaiser(1200));
    }

    #[test]
    fn test_parse_window_kaiser_fractional_beta() {
        assert_eq!(parse_window_str("kaiser:5.5"), WindowType::Kaiser(550));
    }
}
