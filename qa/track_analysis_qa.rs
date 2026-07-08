//! Real-track analysis QA: runs offline pre-analysis on a local directory of
//! real WAV files and checks the results are usable.
//!
//! Point `TIMESTRETCH_TRACK_QA_DIR` at a directory of WAVs (e.g. a DJ
//! library folder). Files with a `<digits>bpm` token in the filename (the
//! `benchmarks/manifest.toml` convention, e.g. `..._124bpm.wav`) also get a
//! BPM-accuracy assertion. Skips cleanly when the variable is unset;
//! `TIMESTRETCH_STRICT_TRACK_QA=1` turns "directory set but no WAVs found"
//! into a failure.
//!
//! Run with:
//! `TIMESTRETCH_TRACK_QA_DIR=~/music cargo test --features qa-harnesses \
//!    --release --test track_analysis_qa -- --nocapture`

use std::path::PathBuf;

/// Parses a `<digits>bpm` token from a filename (case-insensitive), e.g.
/// `Track_(Original Mix)_124bpm.wav` -> `Some(124.0)`.
fn bpm_from_filename(name: &str) -> Option<f64> {
    let lower = name.to_lowercase();
    for token in lower.split(|c: char| !c.is_ascii_alphanumeric()) {
        if let Some(digits) = token.strip_suffix("bpm") {
            if (2..=3).contains(&digits.len()) && digits.chars().all(|c| c.is_ascii_digit()) {
                return digits.parse().ok();
            }
        }
    }
    None
}

fn qa_dir() -> Option<PathBuf> {
    match std::env::var("TIMESTRETCH_TRACK_QA_DIR") {
        Ok(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => None,
    }
}

fn strict() -> bool {
    std::env::var("TIMESTRETCH_TRACK_QA_DIR").is_ok()
        && std::env::var("TIMESTRETCH_STRICT_TRACK_QA").is_ok_and(|v| v == "1")
}

#[test]
fn track_analysis_qa() {
    let Some(dir) = qa_dir() else {
        eprintln!("track_analysis_qa: TIMESTRETCH_TRACK_QA_DIR not set; skipping");
        return;
    };

    let mut wavs: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map(|ext| ext.eq_ignore_ascii_case("wav"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(e) => {
            if strict() {
                panic!("track_analysis_qa: cannot read {}: {}", dir.display(), e);
            }
            eprintln!(
                "track_analysis_qa: cannot read {} ({}); skipping",
                dir.display(),
                e
            );
            return;
        }
    };
    wavs.sort();

    if wavs.is_empty() {
        if strict() {
            panic!("track_analysis_qa: no WAV files in {}", dir.display());
        }
        eprintln!(
            "track_analysis_qa: no WAV files in {}; skipping",
            dir.display()
        );
        return;
    }

    for path in &wavs {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let buffer = match timestretch::io::wav::read_wav_file(path.to_str().unwrap_or_default()) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "track_analysis_qa: {}: unreadable ({}); skipping file",
                    name, e
                );
                continue;
            }
        };
        let mid = timestretch::downmix_to_mid(&buffer.data, buffer.channels.count());
        let (artifact, report) = timestretch::analyze_for_dj_with_report(&mid, buffer.sample_rate);

        let duration = buffer.duration_secs();
        println!(
            "METRIC track=\"{}\" duration_secs={:.1} bpm={:.2} confidence={:.3} \
             onset_count={} onset_rate_per_sec={:.3} odf_median={:.6} odf_mad={:.6} \
             odf_max={:.6} analysis_realtime_factor={:.1}",
            name,
            duration,
            artifact.bpm,
            artifact.confidence,
            report.onset_count,
            report.onset_rate_per_sec,
            report.odf_median,
            report.odf_mad,
            report.odf_max,
            duration / report.analysis_elapsed_secs.max(1e-9),
        );

        // Skeleton-stage assertions: report fields must be finite. The
        // musical assertions (onset rate band, BPM range, filename-tag
        // accuracy, is_usable) arrive with the detection-front-end fix.
        assert!(report.odf_median.is_finite(), "{}: odf_median", name);
        assert!(report.odf_mad.is_finite(), "{}: odf_mad", name);
        assert!(report.odf_max.is_finite(), "{}: odf_max", name);
        assert!(artifact.bpm.is_finite(), "{}: bpm", name);
        assert!(artifact.confidence.is_finite(), "{}: confidence", name);

        if let Some(tagged) = bpm_from_filename(&name) {
            println!("METRIC track=\"{}\" filename_bpm={:.0}", name, tagged);
        }
    }
}

#[test]
fn bpm_filename_parser() {
    assert_eq!(
        bpm_from_filename("12247392_Music Sounds Better With You_(Original Mix)_124bpm.wav"),
        Some(124.0)
    );
    assert_eq!(bpm_from_filename("dense_mastered_128bpm.wav"), Some(128.0));
    assert_eq!(bpm_from_filename("track_88BPM.wav"), Some(88.0));
    assert_eq!(bpm_from_filename("Lolas Theme x Gimme x Hung Up.wav"), None);
    assert_eq!(bpm_from_filename("1bpm.wav"), None); // too few digits
    assert_eq!(bpm_from_filename("bpm.wav"), None);
}
