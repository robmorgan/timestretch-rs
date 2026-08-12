//! Stage 12 robustness harness: deterministic seeded-random adversarial
//! input against every parsing/validation surface of the public API.
//!
//! cargo-fuzz stand-in (no nightly/cargo-fuzz requirement): a simple
//! xorshift64* generator drives thousands of structured-random inputs per
//! target — truncations at every byte length, random byte flips, random
//! garbage, huge declared sizes, hostile parameters, degenerate audio.
//!
//! Contract under test (documented in `src/lib.rs`, "Robustness: the
//! no-panic contract"): every public entry point returns `Err` (or `None`
//! on validated loads) for invalid input — it never panics, hangs, or
//! allocates unboundedly. `Ok` results for mangled-but-structurally-valid
//! input are fine; when audio comes back for nominal-range input it must
//! be finite.
//!
//! Every crash this harness has found is fixed and pinned by a minimized
//! `regression_*` unit test next to the fix:
//! - `src/io/tsa.rs`: PEAK bucket-count multiply overflow
//!   (`regression_peak_bucket_count_mul_overflow_rejected`).
//! - `src/engine/offline.rs`: divide-by-zero on `channels == 0`, feed
//!   hang on interleave mismatch, unvalidated direct-call ratios
//!   (`regression_zero_channels_errs_not_panics`,
//!   `regression_interleave_mismatch_errs_not_hangs`,
//!   `regression_non_finite_and_out_of_range_ratio_errs`).
//!
//! Run: `cargo test --features qa-harnesses --test robustness`

use std::path::PathBuf;

use timestretch::engine::offline::stretch_offline;
use timestretch::engine::{Engine, EngineConfig, EngineProfile};
use timestretch::io::wav::{read_wav, write_wav_16bit, write_wav_24bit, write_wav_float};
use timestretch::{
    AnalysisFile, AudioBuffer, BandPeaks, PREANALYSIS_VERSION, PreAnalysisArtifact, StretchParams,
    hash_samples, pitch_shift, stretch, stretch_into,
};

const SR: u32 = 44_100;

/// Campaign widening: XORs an optional `TIMESTRETCH_FUZZ_SEED` (decimal
/// u64, set by the scheduled CI campaign to its run id) into the fixed
/// per-test seed. Unset, every run is byte-deterministic from the
/// constants; set, each campaign is an independent — but reproducible,
/// the workflow logs the value — random exploration.
fn campaign_seed(base: u64) -> u64 {
    match std::env::var("TIMESTRETCH_FUZZ_SEED") {
        Ok(v) => base ^ v.trim().parse::<u64>().unwrap_or(0),
        Err(_) => base,
    }
}

/// xorshift64* — deterministic, dependency-free. Seeds are fixed constants
/// so every run fuzzes the identical corpus.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.byte()).collect()
    }
}

/// A small but fully-populated `.tsa` container plus its identity.
fn small_container() -> (Vec<u8>, u32, usize, u64) {
    let mono: Vec<f32> = (0..2_048)
        .map(|i| {
            let t = i as f64 / SR as f64;
            (0.6 * (std::f64::consts::TAU * 60.0 * t).sin()
                + 0.2 * (std::f64::consts::TAU * 5_000.0 * t).sin()) as f32
        })
        .collect();
    let mut af = AnalysisFile::for_source(&mono, SR);
    af.artifact = Some(test_artifact(mono.len(), hash_samples(&mono)));
    af.peaks = Some(BandPeaks::compute(&mono, 1, SR));
    let bytes = af.to_bytes();
    (bytes, SR, mono.len(), hash_samples(&mono))
}

fn test_artifact(source_len: usize, content_hash: u64) -> PreAnalysisArtifact {
    PreAnalysisArtifact {
        version: PREANALYSIS_VERSION,
        sample_rate: SR,
        bpm: 126.0,
        confidence: 0.9,
        beat_positions: vec![0, 21_000, 42_000],
        beat_positions_fractional: vec![0.0, 21_000.0, 42_000.0],
        downbeat_beat_indices: vec![0],
        transient_onsets: vec![0, 21_000, 42_000],
        transient_strengths: vec![1.0, 0.8, 0.9],
        onset_band_flux: vec![[1.0, 0.5, 0.2, 0.1]; 3],
        analysis_hop_size: 512,
        source_len_samples: source_len,
        content_hash,
        ..Default::default()
    }
}

/// Both `.tsa` decode paths on the same bytes; touches every field of a
/// successful decode so lazily-wrong data cannot hide.
fn exercise_tsa(bytes: &[u8], sr: u32, len: usize, hash: u64) {
    for parsed in [
        AnalysisFile::from_bytes(bytes).ok(),
        AnalysisFile::from_bytes_validated(bytes, sr, len, hash),
    ]
    .into_iter()
    .flatten()
    {
        let _ = parsed.sample_rate;
        let _ = parsed.source_len_samples;
        if let Some(a) = &parsed.artifact {
            let _ = a.beat_positions_fractional.len();
            let _ = a.strength_at(usize::MAX); // must never index-panic
            let _ = a.is_usable(SR, 0.5);
        }
        if let Some(p) = &parsed.peaks {
            let _ = p.level(0).num_buckets();
        }
    }
}

#[test]
fn tsa_truncation_at_every_length() {
    let (bytes, sr, len, hash) = small_container();
    // Every strict prefix: no panic; a prefix ending exactly on a chunk
    // boundary is a valid shorter container, anything else must Err.
    for cut in 0..bytes.len() {
        exercise_tsa(&bytes[..cut], sr, len, hash);
    }
    // The full container must round-trip.
    let full = AnalysisFile::from_bytes_validated(&bytes, sr, len, hash)
        .expect("untouched container must load");
    assert!(full.artifact.is_some() && full.peaks.is_some());
}

#[test]
fn tsa_random_byte_flips_never_panic() {
    let (bytes, sr, len, hash) = small_container();
    let mut rng = Rng::new(campaign_seed(0x7541_F1E5));
    for _ in 0..3_000 {
        let mut mutated = bytes.clone();
        for _ in 0..1 + rng.below(8) {
            let idx = rng.below(mutated.len());
            match rng.below(3) {
                0 => mutated[idx] ^= 1 << rng.below(8), // single bit
                1 => mutated[idx] = rng.byte(),         // random byte
                _ => mutated[idx] = 0xFF,               // saturate
            }
        }
        exercise_tsa(&mutated, sr, len, hash);
    }
}

#[test]
fn tsa_random_garbage_never_panics() {
    let mut rng = Rng::new(campaign_seed(0x7541_6A2B));
    for _ in 0..3_000 {
        let len = rng.below(600);
        let mut garbage = rng.bytes(len);
        // Half the corpus gets a valid magic + version prefix so the
        // parser reaches the chunk loop instead of bailing at the header.
        if rng.below(2) == 0 && garbage.len() >= 8 {
            garbage[0..4].copy_from_slice(b"TSAF");
            garbage[4..8].copy_from_slice(&1u32.to_le_bytes());
        }
        exercise_tsa(&garbage, SR, 2_048, 1);
    }
}

#[test]
fn tsa_huge_declared_sizes_rejected() {
    let (bytes, sr, len, hash) = small_container();
    let mut rng = Rng::new(campaign_seed(0x7541_D00D));
    // Chunk payload lengths and PEAK bucket counts stamped with huge
    // values at random plausible offsets: must Err (or skip), never panic
    // and never allocate anywhere near the declared size.
    let huge = [
        u64::MAX,
        u64::MAX / 2,
        u64::MAX / 6 + 1,
        u32::MAX as u64,
        1 << 48,
    ];
    for _ in 0..2_000 {
        let mut mutated = bytes.clone();
        let off = 28 + rng.below(mutated.len() - 36); // anywhere past the file header
        let value = huge[rng.below(huge.len())];
        mutated[off..off + 8].copy_from_slice(&value.to_le_bytes());
        exercise_tsa(&mutated, sr, len, hash);
    }
    // Deterministic worst case: the first chunk header's payload_len.
    let mut mutated = bytes;
    mutated[36..44].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(AnalysisFile::from_bytes(&mutated).is_err());
}

/// Unique-per-test temp file for the path-based deprecated JSON API.
fn temp_json_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "timestretch_robustness_{}_{tag}.json",
        std::process::id()
    ))
}

#[allow(deprecated)]
fn exercise_preanalysis_json(path: &PathBuf, contents: &[u8]) {
    std::fs::write(path, contents).expect("temp write");
    if let Ok(artifact) = timestretch::read_preanalysis_json(path) {
        // Whatever parsed must be safely usable.
        let _ = artifact.is_usable(SR, 0.5);
        let _ = artifact.strength_at(usize::MAX);
        let _ = artifact.matches_identity(SR, 0, 0);
        let _ = artifact.resample_to(48_000);
    }
}

#[test]
#[allow(deprecated)]
fn preanalysis_json_adversarial_never_panics() {
    let path = temp_json_path("mutations");
    let valid_path = temp_json_path("valid");
    timestretch::write_preanalysis_json(&valid_path, &test_artifact(2_048, 1234))
        .expect("serialize valid artifact");
    let valid = std::fs::read(&valid_path).expect("read back");

    // Truncation at every length.
    for cut in 0..valid.len() {
        exercise_preanalysis_json(&path, &valid[..cut]);
    }

    // Random byte flips (printable and raw).
    let mut rng = Rng::new(campaign_seed(0x1502_7350));
    for _ in 0..1_500 {
        let mut mutated = valid.clone();
        for _ in 0..1 + rng.below(6) {
            let idx = rng.below(mutated.len());
            mutated[idx] = if rng.below(2) == 0 {
                rng.byte()
            } else {
                b' ' + (rng.byte() % 95) // printable ASCII
            };
        }
        exercise_preanalysis_json(&path, &mutated);
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&valid_path);
}

#[test]
#[allow(deprecated)]
fn preanalysis_json_structured_hostility_never_panics() {
    let path = temp_json_path("structured");
    let mut cases: Vec<String> = vec![
        // Type confusion.
        r#"{"sample_rate":"44100","bpm":"fast","beat_positions":{}}"#.into(),
        r#"{"sample_rate":[],"bpm":[],"confidence":{},"beat_positions":0}"#.into(),
        r#"{"version":-1,"sample_rate":-44100,"bpm":128.0,"downbeat_offset_samples":-5,"confidence":0.5}"#.into(),
        r#"{"sample_rate":44100,"bpm":128.0,"downbeat_offset_samples":0,"confidence":0.5,"transient_strengths":[[1.0]],"onset_band_flux":[1.0]}"#.into(),
        r#"{"sample_rate":44100,"bpm":128.0,"downbeat_offset_samples":0,"confidence":0.5,"key":{"root":700,"mode":"Dorian","confidence":0.1}}"#.into(),
        // Huge / boundary numbers.
        r#"{"sample_rate":4294967296,"bpm":1e999,"downbeat_offset_samples":18446744073709551616,"confidence":1e39}"#.into(),
        r#"{"sample_rate":44100,"bpm":-1e308,"downbeat_offset_samples":0,"confidence":-1e38,"beat_positions":[18446744073709551615],"source_len_samples":18446744073709551615}"#.into(),
        format!(
            r#"{{"sample_rate":44100,"bpm":128.0,"downbeat_offset_samples":0,"confidence":0.5,"content_hash":{}}}"#,
            u64::MAX
        ),
        // Non-JSON float literals.
        r#"{"sample_rate":44100,"bpm":NaN,"downbeat_offset_samples":0,"confidence":0.5}"#.into(),
        r#"{"sample_rate":44100,"bpm":Infinity,"downbeat_offset_samples":0,"confidence":-Infinity}"#.into(),
        // Wrong top-level types, truncated tokens, empty.
        "[]".into(),
        "null".into(),
        "42".into(),
        "\"artifact\"".into(),
        "{".into(),
        r#"{"sample_rate""#.into(),
        String::new(),
        // Duplicate keys, unknown keys, unicode.
        r#"{"bpm":1,"bpm":2,"bpm":3,"sample_rate":44100,"downbeat_offset_samples":0,"confidence":0.5}"#.into(),
        "{\"sample_rate\":44100,\"bpm\":128.0,\"downbeat_offset_samples\":0,\"confidence\":0.5,\"\u{1F5FF}\":\"\u{0000}\"}".into(),
    ];
    // Deep nesting: serde_json's recursion limit must turn these into
    // errors, not stack overflows.
    cases.push("[".repeat(60_000));
    cases.push(format!("{}1{}", "[".repeat(60_000), "]".repeat(60_000)));
    cases.push(format!(
        "{}\"x\"{}",
        "{\"a\":".repeat(30_000),
        "}".repeat(30_000)
    ));
    cases.push(format!(
        r#"{{"beat_positions":{}1{}}}"#,
        "[".repeat(50_000),
        "]".repeat(50_000)
    ));
    // Very long arrays and strings (bounded allocation is fine; a hang or
    // panic is not).
    cases.push(format!(
        r#"{{"sample_rate":44100,"bpm":128.0,"downbeat_offset_samples":0,"confidence":0.5,"beat_positions":[{}]}}"#,
        vec!["1"; 100_000].join(",")
    ));

    for case in &cases {
        exercise_preanalysis_json(&path, case.as_bytes());
    }
    let _ = std::fs::remove_file(&path);
}

/// Valid seed WAVs across the supported encodings.
fn seed_wavs() -> Vec<Vec<u8>> {
    let mono: Vec<f32> = (0..128)
        .map(|i| ((i as f32 * 0.13).sin() * 0.8).clamp(-1.0, 1.0))
        .collect();
    let stereo: Vec<f32> = (0..256)
        .map(|i| ((i as f32 * 0.07).cos() * 0.5).clamp(-1.0, 1.0))
        .collect();
    vec![
        write_wav_16bit(&AudioBuffer::from_mono(mono.clone(), SR)),
        write_wav_24bit(&AudioBuffer::from_stereo(stereo.clone(), 48_000)),
        write_wav_float(&AudioBuffer::from_stereo(stereo, 96_000)),
    ]
}

fn exercise_wav(bytes: &[u8]) {
    if let Ok(buffer) = read_wav(bytes) {
        let _ = buffer.num_frames();
        let _ = buffer.duration_secs();
        let _ = buffer.channel(usize::MAX); // must return empty, not panic
    }
}

#[test]
fn wav_truncation_at_every_length() {
    // Extends the hand-built corruption matrix in tests/wav_error_paths.rs
    // with exhaustive truncation of real encoder output.
    for wav in seed_wavs() {
        for cut in 0..wav.len() {
            exercise_wav(&wav[..cut]);
        }
        assert!(read_wav(&wav).is_ok(), "untouched WAV must parse");
    }
}

#[test]
fn wav_random_byte_flips_never_panic() {
    let seeds = seed_wavs();
    let mut rng = Rng::new(campaign_seed(0x3A4E_F11B));
    for _ in 0..3_000 {
        let mut wav = seeds[rng.below(seeds.len())].clone();
        for _ in 0..1 + rng.below(8) {
            let idx = rng.below(wav.len());
            wav[idx] = rng.byte();
        }
        exercise_wav(&wav);
    }
}

#[test]
fn wav_random_garbage_and_huge_chunks_never_panic() {
    let mut rng = Rng::new(campaign_seed(0x3A4E_600D));
    for _ in 0..2_000 {
        let len = 44 + rng.below(300);
        let mut garbage = rng.bytes(len);
        match rng.below(3) {
            0 => {} // raw garbage
            1 => {
                garbage[0..4].copy_from_slice(b"RIFF");
                garbage[8..12].copy_from_slice(b"WAVE");
            }
            _ => {
                // Valid header + one chunk with a hostile declared size.
                garbage[0..4].copy_from_slice(b"RIFF");
                garbage[8..12].copy_from_slice(b"WAVE");
                let id: &[u8; 4] = [b"fmt ", b"data", b"LIST", b"junk"][rng.below(4)];
                garbage[12..16].copy_from_slice(id);
                let size = [u32::MAX, u32::MAX - 1, 0, 1, 0x8000_0000][rng.below(5)];
                garbage[16..20].copy_from_slice(&size.to_le_bytes());
            }
        }
        exercise_wav(&garbage);
    }

    // Structured hostile fmt fields: any channel count, format code, bit
    // depth, and sample rate — Err or Ok, never a panic.
    for _ in 0..1_000 {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(rng.next_u64() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&(rng.next_u64() as u16).to_le_bytes()); // format code
        wav.extend_from_slice(&(rng.next_u64() as u16).to_le_bytes()); // channels
        wav.extend_from_slice(&(rng.next_u64() as u32).to_le_bytes()); // sample rate
        wav.extend_from_slice(&0u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&0u16.to_le_bytes()); // block align
        wav.extend_from_slice(&(rng.next_u64() as u16).to_le_bytes()); // bits
        wav.extend_from_slice(b"data");
        let data_len = rng.below(64);
        wav.extend_from_slice(&(data_len as u32).to_le_bytes());
        wav.extend_from_slice(&rng.bytes(data_len));
        exercise_wav(&wav);
    }
}

/// Degenerate audio generators for the batch API.
fn degenerate_inputs() -> Vec<(&'static str, Vec<f32>)> {
    let denormal = f32::MIN_POSITIVE / 2.0;
    vec![
        ("empty", vec![]),
        ("one-sample", vec![0.25]),
        ("three-samples", vec![0.1, -0.2, 0.3]),
        ("all-zero", vec![0.0; 256]),
        ("denormals", vec![denormal; 256]),
        (
            "alternating-full-scale",
            (0..256)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
        ("dc-offset", vec![0.9; 256]),
        (
            "mixed-denormal-normal",
            (0..512)
                .map(|i| {
                    if i % 3 == 0 {
                        denormal
                    } else {
                        (i as f32 * 0.1).sin() * 0.5
                    }
                })
                .collect(),
        ),
    ]
}

fn assert_finite(label: &str, out: &[f32]) {
    assert!(
        out.iter().all(|s| s.is_finite()),
        "{label}: output contains non-finite samples"
    );
}

#[test]
fn batch_api_boundary_matrix_never_panics() {
    // Ratios at and just outside the documented 0.01..=100.0 validation
    // range, sample rates at and just outside 8000..=192000, mono/stereo.
    let ratios = [0.0099, 0.01, 0.2, 0.5, 0.98, 1.0, 1.05, 2.0, 100.0, 100.01];
    let rates = [7_999u32, 8_000, 44_100, 192_000, 192_001];
    for (label, input) in degenerate_inputs() {
        for &ratio in &ratios {
            for &rate in &rates {
                for channels in [1u32, 2] {
                    let params = StretchParams::new(ratio)
                        .with_sample_rate(rate)
                        .with_channels(channels);
                    let case = format!("{label} r={ratio} sr={rate} ch={channels}");
                    if let Ok(out) = stretch(&input, &params) {
                        assert_finite(&case, &out);
                    }
                    let mut appended = vec![0.0f32; 3];
                    if let Ok(n) = stretch_into(&input, &params, &mut appended) {
                        assert_eq!(appended.len(), 3 + n, "{case}: stretch_into count");
                        assert_finite(&case, &appended);
                    }
                }
            }
        }
    }
}

#[test]
fn batch_api_rejects_non_finite_and_mismatched_input() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = vec![0.0f32; 64];
        input[13] = bad;
        let params = StretchParams::new(1.5).with_channels(1);
        assert!(stretch(&input, &params).is_err(), "{bad} must be rejected");
        assert!(pitch_shift(&input, &params, 1.2).is_err());
    }
    // Interleave mismatch: stereo params, odd sample count.
    let params = StretchParams::new(1.05).with_channels(2);
    assert!(stretch(&[0.0; 33], &params).is_err());
    // Non-finite ratios via the params struct.
    for ratio in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let params = StretchParams::new(ratio).with_channels(1);
        assert!(stretch(&[0.0; 64], &params).is_err(), "ratio {ratio}");
    }
    // Non-finite / out-of-range pitch factors.
    let params = StretchParams::new(1.0).with_channels(1);
    for factor in [f64::NAN, f64::INFINITY, 0.0, 0.0099, 100.01, -3.0] {
        assert!(
            pitch_shift(&[0.0; 64], &params, factor).is_err(),
            "pitch factor {factor}"
        );
    }
}

#[test]
fn batch_api_wide_pv_paths_with_real_windows_never_panic() {
    // Long enough to enter the true phase-vocoder wide path (>= one FFT
    // window) at ratios beyond the engine's 0.25..=4.0 rate range.
    let input: Vec<f32> = (0..6_144)
        .map(|i| (std::f32::consts::TAU * 220.0 * i as f32 / SR as f32).sin() * 0.5)
        .collect();
    for ratio in [0.01, 0.1, 10.0, 100.0] {
        let params = StretchParams::new(ratio)
            .with_sample_rate(SR)
            .with_channels(1);
        let out = stretch(&input, &params).unwrap_or_else(|e| panic!("ratio {ratio}: {e}"));
        assert_finite(&format!("wide ratio {ratio}"), &out);
        let expected = (input.len() as f64 * ratio).round() as usize;
        assert_eq!(out.len(), expected, "wide ratio {ratio} length");
    }
}

#[test]
fn batch_api_hostile_pre_analysis_never_panics() {
    // A caller-provided artifact is authoritative — even one whose
    // contents are internally inconsistent must not crash the render.
    let input: Vec<f32> = (0..4_096)
        .map(|i| (std::f32::consts::TAU * 220.0 * i as f32 / SR as f32).sin() * 0.5)
        .collect();
    let hostile: Vec<PreAnalysisArtifact> = vec![
        // Positions far beyond the input.
        PreAnalysisArtifact {
            beat_positions: vec![usize::MAX / 2, usize::MAX - 1],
            beat_positions_fractional: vec![1e18, 1e300],
            transient_onsets: vec![usize::MAX - 1],
            ..test_artifact(0, 0)
        },
        // Parallel arrays with mismatched lengths.
        PreAnalysisArtifact {
            transient_onsets: vec![0, 100, 200, 300, 400],
            transient_strengths: vec![1.0],
            onset_band_flux: vec![],
            downbeat_beat_indices: vec![999_999],
            ..test_artifact(0, 0)
        },
        // Zero/degenerate metadata.
        PreAnalysisArtifact {
            sample_rate: 0,
            bpm: 0.0,
            analysis_hop_size: 0,
            confidence: -1.0,
            ..test_artifact(0, 0)
        },
        // Extreme numeric metadata.
        PreAnalysisArtifact {
            bpm: 1e308,
            confidence: f32::MAX,
            downbeat_offset_samples: usize::MAX,
            ..test_artifact(0, 0)
        },
    ];
    for (idx, artifact) in hostile.into_iter().enumerate() {
        for ratio in [0.9, 1.1] {
            let params = StretchParams::new(ratio)
                .with_sample_rate(SR)
                .with_channels(1)
                .with_pre_analysis(artifact.clone());
            if let Ok(out) = stretch(&input, &params) {
                assert_finite(&format!("hostile artifact {idx} r={ratio}"), &out);
            }
        }
    }
}

#[test]
fn batch_api_seeded_random_params_never_panic() {
    let mut rng = Rng::new(campaign_seed(0xBA7C_4A11));
    for iter in 0..120 {
        // Log-uniform ratio across the full valid range, occasionally
        // stepping just outside it.
        let ratio = if rng.below(8) == 0 {
            [0.0, -5.0, 0.00999, 100.001, f64::MAX][rng.below(5)]
        } else {
            10f64.powf(rng.unit_f64() * 4.0 - 2.0) // 0.01..100
        };
        let rate = [8_000u32, 11_025, 22_050, 44_100, 48_000, 96_000, 192_000][rng.below(7)];
        let channels = 1 + rng.below(2) as u32;
        let len = [0usize, 1, 2, 3, 17, 64, 255, 512][rng.below(8)];
        let input: Vec<f32> = (0..len)
            .map(|_| ((rng.unit_f64() * 2.0 - 1.0) as f32) * 0.9)
            .collect();
        let params = StretchParams::new(ratio)
            .with_sample_rate(rate)
            .with_channels(channels);
        if let Ok(out) = stretch(&input, &params) {
            assert_finite(&format!("random iter {iter}"), &out);
        }
    }
}

#[test]
fn stretch_offline_direct_adversarial() {
    // stretch_offline is public in its own right: hostile channel counts
    // and interleave mismatches must Err (pinned as unit regressions in
    // src/engine/offline.rs; swept here across the matrix).
    for channels in [0usize, 9, usize::MAX] {
        assert!(
            stretch_offline(&[0.0; 24], channels, SR, 1.5, None).is_err(),
            "channels {channels} must be rejected"
        );
    }
    // channels 3..=8 are valid for the engine even though StretchParams
    // cannot express them: a whole-frame input must process (or Err), not
    // panic.
    for channels in [3usize, 8] {
        let input = vec![0.1f32; 240 * channels];
        if let Ok(out) = stretch_offline(&input, channels, SR, 1.1, None) {
            assert_finite(&format!("offline ch={channels}"), &out);
        }
    }
    for mismatched_len in [1usize, 3, 4_097] {
        assert!(stretch_offline(&vec![0.0; mismatched_len], 2, SR, 1.05, None).is_err());
    }
}

#[test]
fn engine_config_boundary_matrix() {
    // Invalid configurations must Err from Engine::build, never panic.
    let invalid = [
        EngineConfig {
            channels: 0,
            ..EngineConfig::default()
        },
        EngineConfig {
            channels: 9,
            ..EngineConfig::default()
        },
        EngineConfig {
            channels: usize::MAX,
            ..EngineConfig::default()
        },
        EngineConfig {
            sample_rate: 0,
            ..EngineConfig::default()
        },
        EngineConfig {
            sample_rate: 7_999,
            ..EngineConfig::default()
        },
        EngineConfig {
            sample_rate: 192_001,
            ..EngineConfig::default()
        },
        EngineConfig {
            sample_rate: u32::MAX,
            ..EngineConfig::default()
        },
        EngineConfig {
            max_block_frames: 0,
            ..EngineConfig::default()
        },
        EngineConfig {
            max_block_frames: 63,
            ..EngineConfig::default()
        },
        EngineConfig {
            max_block_frames: 8_193,
            ..EngineConfig::default()
        },
        EngineConfig {
            max_block_frames: usize::MAX,
            ..EngineConfig::default()
        },
        EngineConfig {
            source_capacity_frames: 0,
            ..EngineConfig::default()
        },
        EngineConfig {
            source_capacity_frames: 100,
            ..EngineConfig::default()
        },
        EngineConfig {
            initial_tempo_rate: f64::NAN,
            ..EngineConfig::default()
        },
        EngineConfig {
            initial_tempo_rate: f64::INFINITY,
            ..EngineConfig::default()
        },
    ];
    for (idx, config) in invalid.into_iter().enumerate() {
        assert!(
            Engine::build(config).is_err(),
            "invalid config {idx} must be rejected"
        );
    }

    // Exact boundaries must build, for every profile.
    for profile in [
        EngineProfile::Tape,
        EngineProfile::Keylock,
        EngineProfile::WideKeylock,
    ] {
        for sample_rate in [8_000u32, 192_000] {
            for channels in [1usize, 2, 8] {
                for max_block_frames in [64usize, 8_192] {
                    let config = EngineConfig {
                        sample_rate,
                        channels,
                        profile,
                        max_block_frames,
                        source_capacity_frames: 8_192 * 4 * 4 + 64,
                        ..EngineConfig::default()
                    };
                    assert!(
                        Engine::build(config).is_ok(),
                        "boundary config must build: {profile:?} sr={sample_rate} \
                         ch={channels} block={max_block_frames}"
                    );
                }
            }
        }
    }
}

#[test]
fn engine_process_with_extreme_configs_stays_finite() {
    let mut rng = Rng::new(campaign_seed(0xE9C1_EE7A));
    for (profile, channels, sample_rate) in [
        (EngineProfile::Tape, 8usize, 192_000u32),
        (EngineProfile::Keylock, 1, 8_000),
        (EngineProfile::Keylock, 8, 44_100),
        (EngineProfile::WideKeylock, 2, 8_000),
    ] {
        let handles = Engine::build(EngineConfig {
            sample_rate,
            channels,
            profile,
            ..EngineConfig::default()
        })
        .expect("valid config builds");
        let (controller, mut processor, mut source) =
            (handles.controller, handles.processor, handles.source);

        let chunk: Vec<f32> = (0..2_048 * channels)
            .map(|_| ((rng.unit_f64() * 2.0 - 1.0) as f32) * 0.8)
            .collect();
        let mut out = vec![0.0f32; 256 * channels];
        for _ in 0..48 {
            // Hostile control values are clamped by contract, not errors.
            controller.set_tempo_rate(match rng.below(6) {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => -10.0,
                3 => 1e300,
                4 => 0.0,
                _ => 0.8 + rng.unit_f64() * 0.6,
            });
            controller.set_keylock(rng.below(2) == 0);
            while source.free_frames() >= 2_048 {
                source.push(&chunk);
            }
            processor.process(&mut out);
            assert_finite(&format!("{profile:?} ch={channels} sr={sample_rate}"), &out);
        }
    }
}
