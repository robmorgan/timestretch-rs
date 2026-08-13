//! One arm of an A/B listening comparison: renders excerpts through THIS
//! build's `stretch()` at the given rates. Driven by `scripts/ab.sh`,
//! which renders the other arm from a baseline git ref (or Rubber Band),
//! level-matches, and blinds the set with a sealed key.
//!
//! Usage:
//!   cargo run --release --example ab_render -- <arm_name> <out_dir> \
//!       <rates> <wav_path[:start_secs]>...
//!
//! `rates` is comma-separated tempo rates (e.g. `0.92,1.08`). Excerpts
//! are 20 s from `start_secs` (default 0). Output:
//! `<out_dir>/<track_stem>/<rate_tag>/{<arm_name>,source}.wav`, 32-bit
//! float, unmatched (the assembly step in ab.sh level-matches across
//! arms so every arm gets identical treatment).

use std::path::{Path, PathBuf};

use timestretch::io::{read_wav_file, write_wav_file_float};
use timestretch::{AudioBuffer, StretchParams};

const EXCERPT_SECS: f64 = 20.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        eprintln!("usage: ab_render <arm_name> <out_dir> <rate,rate,...> <wav[:start_secs]>...");
        std::process::exit(2);
    }
    let arm = &args[0];
    let out_base = PathBuf::from(&args[1]);
    let rates: Vec<f64> = args[2]
        .split(',')
        .map(|r| r.trim().parse::<f64>().expect("rate"))
        .collect();

    for spec in &args[3..] {
        let (path_str, start_secs) = match spec.rsplit_once(':') {
            Some((p, s)) if s.parse::<f64>().is_ok() => (p, s.parse::<f64>().unwrap()),
            _ => (spec.as_str(), 0.0),
        };
        let path = Path::new(path_str);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("track")
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .take(24)
            .collect::<String>();
        let buf = read_wav_file(path.to_str().unwrap()).expect("read wav");
        let ch = buf.channels.count();
        let sr = buf.sample_rate;
        let start = ((start_secs * sr as f64) as usize * ch).min(buf.data.len());
        let len = ((EXCERPT_SECS * sr as f64) as usize * ch).min(buf.data.len() - start);
        let excerpt = &buf.data[start..start + len];

        for &rate in &rates {
            let pct = (rate - 1.0) * 100.0;
            let rate_tag = format!("{}{:.0}pct", if pct >= 0.0 { "+" } else { "-" }, pct.abs());
            let dir = out_base.join(&stem).join(&rate_tag);
            std::fs::create_dir_all(&dir).expect("mkdir");
            let params = StretchParams::new(1.0 / rate)
                .with_sample_rate(sr)
                .with_channels(ch as u32);
            let out = timestretch::stretch(excerpt, &params).expect("stretch");
            let out_buf = AudioBuffer::new(out, sr, buf.channels);
            write_wav_file_float(dir.join(format!("{arm}.wav")).to_str().unwrap(), &out_buf)
                .expect("write");
            let src_buf = AudioBuffer::new(excerpt.to_vec(), sr, buf.channels);
            write_wav_file_float(dir.join("source.wav").to_str().unwrap(), &src_buf)
                .expect("write");
            println!("rendered {stem}/{rate_tag} [{arm}]");
        }
    }
}
