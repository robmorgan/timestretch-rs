//! File I/O and argument handling are outside the session processing path.
use std::{error::Error, path::Path};
use timestretch::{
    Backend, Parameters, PitchRatio, RatePolicy, RatioConstraint, SessionConfig, TempoRate,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const USAGE: &str = "timestretch-cli <input.wav> <output.wav> [--tempo RATE] [--pitch RATIO | --semitones N] [--linked] [--backend auto|portable|accelerate] [--sample-rate HZ] [--native-only] [--preserve-duration] [--normalize] [--report FILE.json]\ntimestretch-cli analyze <input.wav> [-o FILE.tsa]";
#[derive(Default)]
struct Options {
    config: SessionConfig,
    normalize: bool,
    report: Option<String>,
    rate: Option<u32>,
}
fn parse(args: &[String]) -> Result<Options> {
    let mut options = Options::default();
    let mut i = 0;
    let mut linked = false;
    while i < args.len() {
        let key = args[i].as_str();
        let value = |i: &mut usize| -> Result<&str> {
            *i += 1;
            args.get(*i)
                .map(String::as_str)
                .ok_or_else(|| format!("{key} needs a value").into())
        };
        match key {
            "--tempo" => options.config.parameters.tempo = TempoRate::new(value(&mut i)?.parse()?)?,
            "--pitch" => options.config.parameters.pitch = PitchRatio::new(value(&mut i)?.parse()?)?,
            "--semitones" => options.config.parameters.pitch = PitchRatio::from_semitones(value(&mut i)?.parse()?)?,
            "--linked" => linked = true,
            "--backend" => options.config.backend = match value(&mut i)? {
                "auto" => Backend::Auto, "portable" => Backend::Portable, "accelerate" => Backend::Accelerate,
                other => return Err(format!("unknown backend: {other}").into()),
            },
            "--sample-rate" => options.rate = Some(value(&mut i)?.parse()?),
            "--native-only" => options.config.rate_policy = RatePolicy::NativeOnly,
            "--preserve-duration" => options.config.parameters.constraint = RatioConstraint::PreserveDuration,
            "--normalize" => options.normalize = true,
            "--report" => options.report = Some(value(&mut i)?.to_owned()),
            "--ratio" | "--profile" | "--window" | "--envelope" | "--pre-analysis" | "--fft-size" =>
                return Err(format!("{key} was removed in 0.16; use --tempo (source/output time), --pitch and --backend; analysis is independent").into()),
            _ => return Err(format!("unknown option: {key}").into()),
        }
        i += 1;
    }
    if linked {
        options.config.parameters = Parameters {
            pitch: PitchRatio::new(options.config.parameters.tempo.get())?,
            ..options.config.parameters
        };
    }
    options.config.parameters.validate()?;
    Ok(options)
}
fn run(args: &[String]) -> Result<()> {
    if args.is_empty() || args[0] == "--help" {
        println!("{USAGE}");
        return Ok(());
    }
    if args[0] == "analyze" {
        let input = args.get(1).ok_or(USAGE)?;
        let output = match &args[2..] {
            [] => format!("{input}.tsa"),
            [flag, path] if flag == "-o" => path.clone(),
            _ => return Err(USAGE.into()),
        };
        let audio = timestretch::io::read_wav_file(input)?;
        let mid = timestretch::downmix_to_mid(&audio.data, audio.channels.count());
        let mut artifact = timestretch::analyze(&mid, audio.sample_rate);
        artifact.loudness =
            timestretch::measure_loudness(&audio.data, audio.channels.count(), audio.sample_rate);
        let mut file = timestretch::AnalysisFile::for_source(&mid, audio.sample_rate);
        file.peaks = Some(timestretch::BandPeaks::compute(&mid, 1, audio.sample_rate));
        file.artifact = Some(artifact);
        timestretch::write_analysis_file(Path::new(&output), &file)?;
        println!("{output}");
        return Ok(());
    }
    if args.len() < 2 {
        return Err(USAGE.into());
    }
    let mut options = parse(&args[2..])?;
    let source = timestretch::io::read_wav_file(&args[0])?;
    options.config.source_sample_rate = source.sample_rate;
    options.config.output_sample_rate = options.rate.unwrap_or(source.sample_rate);
    options.config.channels = source.channels.count();
    let output = timestretch::render(&source.data, options.config)?;
    let r = output.report;
    let mut audio =
        timestretch::AudioBuffer::try_new(output.samples, r.output_sample_rate, source.channels)?;
    let gain = if options.normalize {
        let target = source.rms();
        let rms = audio.rms();
        if rms > 0.0 { target / rms } else { 1.0 }
    } else {
        1.0
    };
    for sample in &mut audio.data {
        *sample *= gain;
    }
    timestretch::io::write_wav_file_float(&args[1], &audio)?;
    let report = serde_json::json!({"backend":format!("{:?}",r.backend),"source_rate":r.source_sample_rate,
        "native_rate":r.timing.native_sample_rate,"output_rate":r.output_sample_rate,"channels":r.channels,
        "source_frames":r.source_frames,"output_frames":r.output_frames,
        "requested_tempo":r.parameters.requested.tempo.get(),"achieved_tempo":r.parameters.tempo(),
        "achieved_stretch":r.parameters.achieved_stretch,"achieved_pitch":r.parameters.achieved_pitch,
        "normalization_gain":gain,"source_lookahead_frames":r.timing.source_lookahead_frames,
        "control_quantum_frames":r.timing.control_quantum_frames});
    if let Some(path) = options.report {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(e) = run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }
    #[test]
    fn options_are_validated_and_normalization_is_explicit() {
        let o = parse(&args(&["--tempo", "2", "--linked"])).unwrap();
        assert!(!o.normalize);
        assert_eq!(o.config.parameters.pitch.get(), 2.0);
        assert!(parse(&args(&["--tempo", "NaN"])).is_err());
        assert!(parse(&args(&["--profile", "wide"])).is_err());
        assert!(parse(&args(&["--tempo"])).is_err());
        assert!(parse(&args(&["--tempo", "0.03125"])).is_ok());
        assert!(parse(&args(&["--tempo", "0.0625", "--pitch", "2"])).is_ok());
        assert!(parse(&args(&["--tempo", "0.03125", "--pitch", "2"])).is_err());
        assert!(parse(&args(&["--tempo", "0.03125", "--linked"])).is_err());
    }
}
