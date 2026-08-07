//! ROADMAP Stage 16 audition-set renderer: prepares the blind-listening
//! material for (a) the tonal-HF granulation verdict and (b) the
//! corrected-range re-audition of the Stage 7 "SOLA owns the corrected
//! range" decision on the post-Stage-13 (phase-fixed) vocoder.
//!
//! Per corpus excerpt and DJ rate, renders four arms:
//!   - `ours_sola.wav`  — the shipped keylock path (`stretch()`, which is
//!     the live graph for DJ ratios by construction)
//!   - `pv512.wav` / `pv1024.wav` — the re-audition prototype: the SAME
//!     two-band topology as the keylock chain (LinkwitzRiley8 at 120 Hz,
//!     low band pitch-follows via band-limited resample), but with the
//!     high band corrected by a fixed small/medium-FFT identity-locked
//!     phase vocoder instead of SOLA
//!   - `rubberband.wav` — Rubber Band R3 (`rubberband-r3` or
//!     `rubberband --fine`) reference, skipped with a warning if the CLI
//!     is absent
//!
//! All arms are RMS-matched to the source excerpt. Output:
//! `target/stage16_audition/<track>/<rate_tag>/<arm>.wav` plus a
//! `summary.txt` of levels. Usage:
//!   cargo run --release --example stage16_audition [corpus_dir]

use std::path::{Path, PathBuf};
use std::process::Command;

use timestretch::core::crossover::LinkwitzRiley8;
use timestretch::core::resample::resample_sinc_default;
use timestretch::core::window::WindowType;
use timestretch::io::{read_wav_file, write_wav_file_16bit};
use timestretch::stretch::{PhaseLockingMode, PhaseVocoder};
use timestretch::{AudioBuffer, Channels, StretchParams};

const CROSSOVER_HZ: f64 = 120.0;
const EXCERPT_SECS: f64 = 20.0;
/// DJ rates for the audition (tempo rate; stretch ratio is 1/rate).
const RATES: [f64; 4] = [0.92, 0.96, 1.04, 1.08];

/// (tag, corpus file, excerpt start secs — chosen for sustained tonal
/// content over a beat: strings, vocals, pads).
const TRACKS: [(&str, &str, f64); 3] = [
    ("hot_stuff", "14220825_Hot Stuff_(Original Mix).wav", 60.0),
    (
        "msbwy",
        "12247392_Music Sounds Better With You_(Original Mix).wav",
        90.0,
    ),
    (
        "cold_heart",
        "15836669_Cold Heart_(PNAU Extended Mix).wav",
        45.0,
    ),
];

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&s| s as f64 * s as f64).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

fn match_rms(x: &mut [f32], target: f64) {
    let current = rms(x);
    if current > 1e-9 {
        let g = (target / current) as f32;
        for s in x.iter_mut() {
            *s *= g;
        }
    }
}

/// The re-audition prototype: keylock's two-band topology with a fixed
/// phase vocoder as the high-band corrector. Constant-rate offline
/// equivalence: the high band is PV-stretched at the ratio (pitch
/// preserved), the low band is resampled to length (pitch follows tempo
/// — the deck contract), and both sum at equal length.
fn render_pv_prototype(input: &[f32], sample_rate: u32, ratio: f64, fft: usize) -> Vec<f32> {
    let mut split = LinkwitzRiley8::new(CROSSOVER_HZ, sample_rate);
    let mut low = vec![0.0f32; input.len()];
    let mut high = vec![0.0f32; input.len()];
    split.process(input, &mut low, &mut high);

    let out_len = (input.len() as f64 * ratio).round() as usize;
    let low_out = resample_sinc_default(&low, out_len);

    let mut pv = PhaseVocoder::with_options(
        fft,
        fft / 8,
        ratio,
        sample_rate,
        100.0,
        WindowType::Hann,
        PhaseLockingMode::Identity,
    );
    let mut high_out = pv.process(&high).expect("pv render");
    high_out.resize(out_len, 0.0);

    low_out
        .iter()
        .zip(high_out.iter())
        .map(|(&l, &h)| l + h)
        .collect()
}

fn render_rubberband(src: &Path, out: &Path, ratio: f64) -> bool {
    let cli = ["rubberband-r3", "rubberband"]
        .iter()
        .find(|c| Command::new(c).arg("--version").output().is_ok());
    let Some(cli) = cli else {
        return false;
    };
    let mut cmd = Command::new(cli);
    if *cli == "rubberband" {
        cmd.arg("--fine");
    }
    cmd.arg("--time")
        .arg(format!("{ratio}"))
        .arg(src)
        .arg(out)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    let corpus = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmarks/audio/bpm-corpus".to_string());
    let out_base = PathBuf::from("target/stage16_audition");
    let mut summary = String::new();

    for (tag, file, start_secs) in TRACKS {
        let path = Path::new(&corpus).join(file);
        let Ok(buf) = read_wav_file(path.to_str().unwrap()) else {
            eprintln!("skipping {tag}: cannot read {}", path.display());
            continue;
        };
        let ch = buf.channels.count();
        let sr = buf.sample_rate;
        let start = (start_secs * sr as f64) as usize * ch;
        let len = (EXCERPT_SECS * sr as f64) as usize * ch;
        if start + len > buf.data.len() {
            eprintln!("skipping {tag}: excerpt out of range");
            continue;
        }
        let excerpt = &buf.data[start..start + len];
        let source_rms = rms(excerpt);

        for rate in RATES {
            let ratio = 1.0 / rate;
            let pct = (rate - 1.0) * 100.0;
            let rate_tag = format!("{}{:.0}pct", if pct >= 0.0 { "+" } else { "-" }, pct.abs());
            let dir = out_base.join(tag).join(&rate_tag);
            std::fs::create_dir_all(&dir).expect("mkdir");

            // Source excerpt (for the rubberband CLI and as the anchor).
            let src_wav = dir.join("source.wav");
            let src_buf = AudioBuffer::new(excerpt.to_vec(), sr, buf.channels);
            write_wav_file_16bit(src_wav.to_str().unwrap(), &src_buf).expect("write");

            // Arm 1: shipped keylock path.
            let params = StretchParams::new(ratio)
                .with_sample_rate(sr)
                .with_channels(ch as u32);
            let mut ours = timestretch::stretch(excerpt, &params).expect("stretch");
            match_rms(&mut ours, source_rms);
            let ours_buf = AudioBuffer::new(ours, sr, buf.channels);
            write_wav_file_16bit(dir.join("ours_sola.wav").to_str().unwrap(), &ours_buf)
                .expect("write");

            // Arms 2+3: the PV-behind-the-split prototypes (mono mid for
            // the corrected path comparison; stereo nuance is not what
            // this audition decides).
            let mid = timestretch::downmix_to_mid(excerpt, ch);
            for fft in [512usize, 1024] {
                let mut rendered = render_pv_prototype(&mid, sr, ratio, fft);
                match_rms(&mut rendered, rms(&mid));
                let pv_buf = AudioBuffer::new(rendered, sr, Channels::Mono);
                write_wav_file_16bit(dir.join(format!("pv{fft}.wav")).to_str().unwrap(), &pv_buf)
                    .expect("write");
            }

            // Arm 4: Rubber Band reference.
            let rb_ok = render_rubberband(&src_wav, &dir.join("rubberband.wav"), ratio);
            if !rb_ok {
                eprintln!("  rubberband CLI unavailable/failed for {tag}/{rate_tag}");
            }

            summary.push_str(&format!(
                "{tag}/{rate_tag}: source_rms {source_rms:.4} rb={}\n",
                if rb_ok { "ok" } else { "MISSING" }
            ));
            println!("rendered {tag}/{rate_tag}");
        }
    }
    std::fs::write(out_base.join("summary.txt"), summary).expect("summary");
    println!("audition set in {}", out_base.display());
}
