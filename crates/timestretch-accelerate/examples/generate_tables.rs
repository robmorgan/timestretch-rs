//! Generate the core's immutable window tables using the Accelerate adapter.
//! Run with `--check` to compare every byte, or `--write` to save generated tables.
#![forbid(unsafe_code)]

use std::{env, error::Error, path::PathBuf, process::ExitCode};

const USAGE: &str = "Usage: generate_tables [--check | --write] [--output DIRECTORY]

  --check             Compare all six tables byte for byte without writing (default)
  --write             Create or replace the six .bin files in the output directory
  --output DIRECTORY  Use this directory instead of timestretch-core's table directory
  --help              Show this help

Generation and comparison require macOS and Apple Accelerate.";

#[derive(Clone, Copy)]
enum Mode {
    Check,
    Write,
}

struct Options {
    mode: Mode,
    output: PathBuf,
}

fn options() -> Result<Option<Options>, Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let mut mode = None;
    let mut output = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--help" | "-h") => {
                println!("{USAGE}");
                return Ok(None);
            }
            Some("--check" | "--write") => {
                if mode.is_some() {
                    return Err("specify --check or --write only once".into());
                }
                mode = Some(if arg == "--write" {
                    Mode::Write
                } else {
                    Mode::Check
                });
            }
            Some("--output") => {
                if output.is_some() {
                    return Err("specify --output only once".into());
                }
                let directory = args.next().ok_or("--output requires a directory")?;
                if directory.to_string_lossy().starts_with("--") {
                    return Err("--output requires a directory, not an option".into());
                }
                output = Some(PathBuf::from(directory));
            }
            _ => {
                return Err(format!("unknown argument: {}\n{USAGE}", arg.to_string_lossy()).into());
            }
        }
    }
    Ok(Some(Options {
        mode: mode.unwrap_or(Mode::Check),
        output: output.unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../timestretch-core/src/processor/tables")
        }),
    }))
}

#[cfg(target_os = "macos")]
fn generate() -> Vec<(String, Vec<f32>)> {
    use std::f32::consts::{FRAC_PI_2, TAU};
    use timestretch_accelerate::sin_cos;

    let mut tables = Vec::with_capacity(6);
    for n in [512, 1024, 2048, 4096, 8192] {
        // Round the angular step to f32 before multiplying by each index.
        // Reordering the division or using f64 changes the table's float bits.
        let step = TAU / (n - 1) as f32;
        let angles: Vec<_> = (0..n).map(|i| i as f32 * step).collect();
        let mut sine = vec![0.0; n];
        let mut cosine = vec![0.0; n];
        sin_cos(&angles, &mut sine, &mut cosine);
        tables.push((format!("cosine-{n}.bin"), cosine));
    }

    let n = 16384;
    let step = FRAC_PI_2 / (n - 1) as f32;
    let angles: Vec<_> = (0..n).map(|i| i as f32 * step).collect();
    let mut sine = vec![0.0; n];
    let mut cosine = vec![0.0; n];
    sin_cos(&angles, &mut sine, &mut cosine);
    for value in &mut sine {
        // Two rounded f32 multiplications; powf and scalar sin differ in rounding.
        *value = (*value * *value) * *value;
    }
    tables.push(("flank.bin".into(), sine));
    tables
}

#[cfg(target_os = "macos")]
fn run(options: Options) -> Result<(), Box<dyn Error>> {
    use std::fs;

    let tables = generate();
    if matches!(options.mode, Mode::Write) {
        fs::create_dir_all(&options.output)
            .map_err(|e| format!("{}: {e}", options.output.display()))?;
    }
    let mut mismatches = 0;
    let mut values = 0;
    for (name, table) in &tables {
        values += table.len();
        let path = options.output.join(name);
        let bytes: Vec<_> = table.iter().flat_map(|value| value.to_le_bytes()).collect();
        match options.mode {
            Mode::Check => {
                let stored = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                if stored == bytes {
                    println!("{name}: exact match ({} floats)", table.len());
                } else {
                    mismatches += 1;
                    if stored.len() != bytes.len() {
                        eprintln!(
                            "{name}: length mismatch: stored {} bytes, generated {} bytes",
                            stored.len(),
                            bytes.len()
                        );
                    } else {
                        let index =
                            stored.iter().zip(&bytes).position(|(a, b)| a != b).unwrap() / 4;
                        let start = index * 4;
                        let stored_bits = u32::from_le_bytes(stored[start..start + 4].try_into()?);
                        let generated_bits = table[index].to_bits();
                        eprintln!(
                            "{name}: first mismatch at float {index}: stored 0x{stored_bits:08x}, generated 0x{generated_bits:08x}"
                        );
                    }
                }
            }
            Mode::Write => {
                fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
                println!("{}: wrote {} floats", path.display(), table.len());
            }
        }
    }
    if mismatches != 0 {
        return Err(format!("{mismatches} table(s) differ; no files were changed").into());
    }
    match options.mode {
        Mode::Check => println!(
            "All {} tables match exactly ({values} floats).",
            tables.len()
        ),
        Mode::Write => println!("Wrote {} tables ({values} floats).", tables.len()),
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run(options: Options) -> Result<(), Box<dyn Error>> {
    let _ = (options.mode, options.output);
    Err("table generation and comparison require macOS and Apple Accelerate".into())
}

fn main() -> ExitCode {
    let result = options().and_then(|options| options.map_or(Ok(()), run));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
