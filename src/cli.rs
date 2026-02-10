use timestretch::{EdmPreset, StretchParams};

fn main() {
    let args: Vec<String> = std::env::args().collect();

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
    let mut preset: Option<EdmPreset> = None;
    let mut format_24bit = false;
    let mut format_float = false;
    let mut verbose = false;

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
            "--preset" => {
                i += 1;
                preset = Some(parse_preset(&args, i));
            }
            "--24bit" => format_24bit = true,
            "--float" => format_float = true,
            "--verbose" | "-v" => verbose = true,
            // Legacy positional: ratio [preset]
            other => {
                if ratio.is_none() {
                    match other.parse::<f64>() {
                        Ok(r) => ratio = Some(r),
                        Err(_) => {
                            // Try as preset name
                            preset = Some(parse_preset_str(other));
                            i += 1;
                            continue;
                        }
                    }
                } else if preset.is_none() {
                    preset = Some(parse_preset_str(other));
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
        if preset.is_none() {
            preset = Some(EdmPreset::DjBeatmatch);
        }
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

    // Configure params
    let mut params = StretchParams::new(stretch_ratio)
        .with_sample_rate(buffer.sample_rate)
        .with_channels(buffer.channels.count() as u32);

    if let Some(p) = preset {
        params = params.with_preset(p);
    }

    if verbose {
        eprintln!("Parameters: {}", params);
        eprintln!(
            "  Transient sensitivity: {:.2}",
            params.transient_sensitivity
        );
        eprintln!("  Sub-bass cutoff: {:.0} Hz", params.sub_bass_cutoff);
        eprintln!("  WSOLA segment: {} samples", params.wsola_segment_size);
        eprintln!("  WSOLA search: {} samples", params.wsola_search_range);
        eprintln!("  Beat-aware: {}", params.beat_aware);
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
        let input_duration = buffer.duration_secs();
        let processing_secs = elapsed.as_secs_f64();
        let realtime_factor = if processing_secs > 0.0 {
            input_duration / processing_secs
        } else {
            f64::INFINITY
        };
        eprintln!(
            "Processing time: {:.3}s ({:.1}x realtime)",
            processing_secs, realtime_factor
        );
    }

    // Write output
    let write_result = if format_float {
        timestretch::io::wav::write_wav_file_float(output_path, &output)
    } else if format_24bit {
        timestretch::io::wav::write_wav_file_24bit(output_path, &output)
    } else {
        timestretch::io::wav::write_wav_file_16bit(output_path, &output)
    };

    if let Err(e) = write_result {
        eprintln!("ERROR: Failed to write {}: {}", output_path, e);
        std::process::exit(1);
    }

    eprintln!("Written to {}", output_path);
}

fn print_usage() {
    eprintln!("Usage: timestretch-cli <input.wav> <output.wav> [options]");
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  --ratio <f>                   Stretch ratio (1.5 = 50% slower)");
    eprintln!("  --from-bpm <f> --to-bpm <f>   BPM matching (auto-selects DJ preset)");
    eprintln!("  --auto-bpm --to-bpm <f>       Auto-detect source BPM, match to target");
    eprintln!("  --pitch <f>                   Pitch shift (2.0 = up one octave)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --preset <name>   dj, house, halftime, ambient, vocal");
    eprintln!("  --24bit           Write 24-bit PCM output (default: 16-bit)");
    eprintln!("  --float           Write 32-bit float output");
    eprintln!("  --verbose, -v     Show detailed processing parameters and timing");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  timestretch-cli in.wav out.wav --ratio 1.5");
    eprintln!("  timestretch-cli in.wav out.wav --from-bpm 126 --to-bpm 128");
    eprintln!("  timestretch-cli in.wav out.wav --auto-bpm --to-bpm 128");
    eprintln!("  timestretch-cli in.wav out.wav --pitch 0.5 --preset vocal");
    eprintln!("  timestretch-cli in.wav out.wav 1.5 house");
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

fn parse_preset(args: &[String], idx: usize) -> EdmPreset {
    if idx >= args.len() {
        eprintln!("ERROR: --preset requires a value");
        std::process::exit(1);
    }
    parse_preset_str(&args[idx])
}

fn parse_preset_str(s: &str) -> EdmPreset {
    match s {
        "dj" => EdmPreset::DjBeatmatch,
        "house" => EdmPreset::HouseLoop,
        "halftime" => EdmPreset::Halftime,
        "ambient" => EdmPreset::Ambient,
        "vocal" => EdmPreset::VocalChop,
        other => {
            eprintln!("WARNING: Unknown preset '{}', using HouseLoop", other);
            EdmPreset::HouseLoop
        }
    }
}
