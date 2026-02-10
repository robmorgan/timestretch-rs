use timestretch::{EdmPreset, StretchParams};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: timestretch-cli <input.wav> <output.wav> <ratio> [preset]");
        eprintln!("  ratio: stretch factor (e.g., 1.5 for 50% slower, 0.5 for 2x faster)");
        eprintln!("  preset: dj, house, halftime, ambient, vocal (optional)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let ratio: f64 = match args[3].parse() {
        Ok(r) => r,
        Err(_) => {
            eprintln!("ERROR: Invalid stretch ratio: {}", args[3]);
            std::process::exit(1);
        }
    };

    let preset = if args.len() > 4 {
        match args[4].as_str() {
            "dj" => Some(EdmPreset::DjBeatmatch),
            "house" => Some(EdmPreset::HouseLoop),
            "halftime" => Some(EdmPreset::Halftime),
            "ambient" => Some(EdmPreset::Ambient),
            "vocal" => Some(EdmPreset::VocalChop),
            other => {
                eprintln!("WARNING: Unknown preset '{}', using default", other);
                None
            }
        }
    } else {
        None
    };

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

    // Configure params
    let mut params = StretchParams::new(ratio)
        .with_sample_rate(buffer.sample_rate)
        .with_channels(buffer.channels.count() as u32);

    if let Some(p) = preset {
        params = params.with_preset(p);
    }

    // Stretch
    let output = match timestretch::stretch_buffer(&buffer, &params) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ERROR: Stretching failed: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "Output: {} frames, {:.2}s (ratio: {:.3})",
        output.num_frames(),
        output.duration_secs(),
        output.num_frames() as f64 / buffer.num_frames() as f64
    );

    // Write output
    if let Err(e) = timestretch::io::wav::write_wav_file_16bit(output_path, &output) {
        eprintln!("ERROR: Failed to write {}: {}", output_path, e);
        std::process::exit(1);
    }

    eprintln!("Written to {}", output_path);
}
