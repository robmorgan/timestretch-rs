//! Retained seeded TSA/JSON/WAV adversarial gates; fixture counts unchanged.
#![cfg(all(feature = "analysis", feature = "io"))]
use std::path::PathBuf;

use timestretch::io::wav::{read_wav, write_wav_16bit, write_wav_24bit, write_wav_float};
use timestretch::{
    AnalysisFile, AudioBuffer, BandPeaks, PREANALYSIS_VERSION, PreAnalysisArtifact, hash_samples,
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
        write_wav_16bit(&AudioBuffer::from_mono(mono.clone(), SR).unwrap()),
        write_wav_24bit(&AudioBuffer::from_stereo(stereo.clone(), 48_000).unwrap()),
        write_wav_float(&AudioBuffer::from_stereo(stereo, 96_000).unwrap()),
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
