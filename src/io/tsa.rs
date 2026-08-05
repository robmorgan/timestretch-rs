//! `.tsa` analysis container: one file per track holding every persisted
//! analysis product.
//!
//! The container consolidates what used to be two sidecars — the
//! pre-analysis artifact (`.tsanalysis.json`) and the waveform-peaks cache
//! (`.tspeaks`) — behind a single content-bound identity: sample rate,
//! mono-signal length, and FNV-1a content hash ([`hash_samples`]), so a
//! renamed or retagged file keeps its analysis.
//!
//! Two API layers:
//! - **Bytes** ([`AnalysisFile::to_bytes`], [`AnalysisFile::from_bytes`],
//!   [`AnalysisFile::from_bytes_validated`]) — no filesystem coupling, for
//!   consumers that store the blob elsewhere (e.g. a library database
//!   keyed by content hash).
//! - **Files** ([`read_analysis_file`], [`read_analysis_file_validated`],
//!   [`write_analysis_file`], [`analysis_file_path`]) — the sidecar
//!   convention (`<audio>.tsa`), thin wrappers over the bytes layer with
//!   an atomic (temp + rename) writer.
//!
//! Layout (little-endian throughout): a 28-byte file header — magic
//! `TSAF`, container version, `sample_rate` u32, `source_len_samples`
//! u64, `content_hash` u64 — followed by sequential chunks, each tagged
//! `[u8; 4]` + chunk version u32 + payload length u64. Unknown tags and
//! unknown versions of known tags are skipped (forward compatibility);
//! duplicate known chunks, truncation, and trailing bytes are structural
//! errors. Readers never panic on hostile input.

use std::path::{Path, PathBuf};

use crate::analysis::waveform::{
    BASE_BUCKETS_PER_SEC, BandPeaks, CROSSOVER_HIGH_HZ, CROSSOVER_LOW_HZ, NUM_BANDS, PeakLevel,
    base_num_buckets,
};
use crate::core::preanalysis::{PreAnalysisArtifact, hash_samples};
use crate::error::StretchError;

/// Current `.tsa` container format version.
pub const TSA_CONTAINER_VERSION: u32 = 1;

const MAGIC: [u8; 4] = *b"TSAF";
const FILE_HEADER_LEN: usize = 28;
const CHUNK_HEADER_LEN: usize = 16;
/// Pre-analysis artifact chunk: serde-JSON bytes of [`PreAnalysisArtifact`]
/// (the artifact's own schema versioning applies inside the payload).
const TAG_ARTIFACT: [u8; 4] = *b"ARTF";
/// Waveform-peaks chunk: analyzer parameters + the base pyramid level.
const TAG_PEAKS: [u8; 4] = *b"PEAK";
const ARTIFACT_CHUNK_VERSION: u32 = 1;
const PEAKS_CHUNK_VERSION: u32 = 1;
/// PEAK payload prefix: buckets/s, two crossovers (f64 bits), bucket count.
const PEAKS_PARAMS_LEN: usize = 32;

/// Consolidated per-track analysis: one content-bound identity, optional
/// payload chunks. Missing chunks are simply absent analysis, not errors.
#[derive(Clone)]
pub struct AnalysisFile {
    /// Sample rate the analysis ran at.
    pub sample_rate: u32,
    /// Length in frames of the mono analysis signal.
    pub source_len_samples: usize,
    /// FNV-1a 64 hash of the mono analysis signal ([`hash_samples`]).
    pub content_hash: u64,
    /// Pre-analysis artifact (beat grid, onsets, key, loudness).
    pub artifact: Option<PreAnalysisArtifact>,
    /// 3-band waveform peaks pyramid.
    pub peaks: Option<BandPeaks>,
}

impl AnalysisFile {
    /// Empty container bound to a mono analysis signal's identity.
    pub fn for_source(mono: &[f32], sample_rate: u32) -> Self {
        Self {
            sample_rate,
            source_len_samples: mono.len(),
            content_hash: hash_samples(mono),
            artifact: None,
            peaks: None,
        }
    }

    /// Encode the container. Chunks are emitted for present fields only;
    /// a header-only container (no analysis yet) is valid.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&TSA_CONTAINER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(self.source_len_samples as u64).to_le_bytes());
        bytes.extend_from_slice(&self.content_hash.to_le_bytes());
        debug_assert_eq!(bytes.len(), FILE_HEADER_LEN);

        if let Some(artifact) = &self.artifact {
            let payload = serde_json::to_vec(artifact)
                .expect("PreAnalysisArtifact JSON serialization cannot fail");
            push_chunk_header(
                &mut bytes,
                TAG_ARTIFACT,
                ARTIFACT_CHUNK_VERSION,
                payload.len(),
            );
            bytes.extend_from_slice(&payload);
        }

        if let Some(peaks) = &self.peaks {
            let base = peaks.level(0);
            let num_buckets = base.num_buckets();
            push_chunk_header(
                &mut bytes,
                TAG_PEAKS,
                PEAKS_CHUNK_VERSION,
                PEAKS_PARAMS_LEN + 6 * num_buckets,
            );
            bytes.extend_from_slice(&base.buckets_per_sec.to_le_bytes());
            bytes.extend_from_slice(&CROSSOVER_LOW_HZ.to_le_bytes());
            bytes.extend_from_slice(&CROSSOVER_HIGH_HZ.to_le_bytes());
            bytes.extend_from_slice(&(num_buckets as u64).to_le_bytes());
            for band in 0..NUM_BANDS {
                bytes.extend(base.pos[band].iter().map(|&v| quantize_unit(v)));
            }
            for band in 0..NUM_BANDS {
                bytes.extend(base.neg[band].iter().map(|&v| quantize_unit(-v)));
            }
        }

        bytes
    }

    /// Structural decode: envelope errors (bad magic/version, truncation,
    /// duplicate or overlong chunks, trailing bytes) are `Err`; a chunk
    /// whose *payload* fails to decode (e.g. corrupt artifact JSON)
    /// degrades to the field being `None`. Identity is returned, not
    /// checked against anything.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StretchError> {
        parse(bytes, None)
    }

    /// Load-boundary decode: `None` unless the container's identity equals
    /// the given one. Surviving chunks are additionally gated — an
    /// artifact that fails [`PreAnalysisArtifact::matches_identity`] or a
    /// peaks chunk built with different analyzer parameters (buckets/s,
    /// crossovers) or an unexpected bucket count is dropped to `None`
    /// while the rest of the container stays usable.
    pub fn from_bytes_validated(
        bytes: &[u8],
        sample_rate: u32,
        source_len_samples: usize,
        content_hash: u64,
    ) -> Option<Self> {
        parse(bytes, Some((sample_rate, source_len_samples, content_hash))).ok()
    }
}

fn push_chunk_header(bytes: &mut Vec<u8>, tag: [u8; 4], version: u32, payload_len: usize) {
    bytes.extend_from_slice(&tag);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(payload_len as u64).to_le_bytes());
}

fn invalid(msg: &str) -> StretchError {
    StretchError::InvalidFormat(format!(".tsa container: {msg}"))
}

/// `v` in `[0, 1]` to a u8 step; out-of-range clamps.
fn quantize_unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn dequantize_unit(q: u8) -> f32 {
    q as f32 / 255.0
}

/// Shared decoder. With `expected` identity, the header must match it
/// exactly and chunk payloads are gated per-chunk (mismatches degrade to
/// `None` fields); without, chunks decode as stored.
fn parse(bytes: &[u8], expected: Option<(u32, usize, u64)>) -> Result<AnalysisFile, StretchError> {
    if bytes.len() < FILE_HEADER_LEN {
        return Err(invalid("shorter than the file header"));
    }
    let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());

    if bytes[0..4] != MAGIC {
        return Err(invalid("bad magic"));
    }
    if u32_at(4) != TSA_CONTAINER_VERSION {
        return Err(invalid("unsupported container version"));
    }
    let sample_rate = u32_at(8);
    let source_len_samples = usize::try_from(u64_at(12))
        .map_err(|_| invalid("source length exceeds addressable memory"))?;
    let content_hash = u64_at(20);

    if let Some((want_sr, want_len, want_hash)) = expected
        && (sample_rate != want_sr || source_len_samples != want_len || content_hash != want_hash)
    {
        return Err(invalid("identity mismatch"));
    }

    let mut file = AnalysisFile {
        sample_rate,
        source_len_samples,
        content_hash,
        artifact: None,
        peaks: None,
    };
    let mut artifact_seen = false;
    let mut peaks_seen = false;

    let mut cursor = FILE_HEADER_LEN;
    while cursor < bytes.len() {
        if bytes.len() - cursor < CHUNK_HEADER_LEN {
            return Err(invalid("truncated chunk header"));
        }
        let tag: [u8; 4] = bytes[cursor..cursor + 4].try_into().unwrap();
        let chunk_version = u32_at(cursor + 4);
        let payload_len = usize::try_from(u64_at(cursor + 8))
            .map_err(|_| invalid("chunk length exceeds addressable memory"))?;
        cursor += CHUNK_HEADER_LEN;
        // Bounds first: every later slice of the payload is in range, and
        // allocations stay bounded by the actual byte count present.
        if bytes.len() - cursor < payload_len {
            return Err(invalid("chunk payload extends past end of data"));
        }
        let payload = &bytes[cursor..cursor + payload_len];
        cursor += payload_len;

        match (tag, chunk_version) {
            (TAG_ARTIFACT, ARTIFACT_CHUNK_VERSION) => {
                if artifact_seen {
                    return Err(invalid("duplicate ARTF chunk"));
                }
                artifact_seen = true;
                // Payload-level decode failure degrades to "absent", like
                // an unknown chunk version — the peaks stay usable.
                let artifact: Option<PreAnalysisArtifact> = serde_json::from_slice(payload).ok();
                file.artifact = match (artifact, expected) {
                    (Some(a), Some((sr, len, hash))) => {
                        a.matches_identity(sr, len, hash).then_some(a)
                    }
                    (a, None) => a,
                    (None, _) => None,
                };
            }
            (TAG_PEAKS, PEAKS_CHUNK_VERSION) => {
                if peaks_seen {
                    return Err(invalid("duplicate PEAK chunk"));
                }
                peaks_seen = true;
                file.peaks = decode_peaks(payload, expected.map(|(sr, len, _)| (sr, len)))?;
            }
            // Unknown tag, or a known tag from a future envelope revision:
            // skip — degrades to "chunk absent", never an error.
            _ => {}
        }
    }

    Ok(file)
}

/// Decode a PEAK payload. Envelope errors (wrong payload length) are
/// `Err`; with `expected` identity, parameter or bucket-count mismatches
/// degrade to `Ok(None)` (stale cache, rebuild) instead.
fn decode_peaks(
    payload: &[u8],
    expected: Option<(u32, usize)>,
) -> Result<Option<BandPeaks>, StretchError> {
    if payload.len() < PEAKS_PARAMS_LEN {
        return Err(invalid("PEAK payload shorter than its parameter block"));
    }
    let u64_at = |off: usize| u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
    let buckets_per_sec = f64::from_bits(u64_at(0));
    let crossover_low_bits = u64_at(8);
    let crossover_high_bits = u64_at(16);
    let num_buckets = usize::try_from(u64_at(24))
        .map_err(|_| invalid("PEAK bucket count exceeds addressable memory"))?;
    if payload.len() != PEAKS_PARAMS_LEN + 6 * num_buckets {
        return Err(invalid("PEAK payload length disagrees with bucket count"));
    }

    if let Some((sample_rate, source_len_samples)) = expected {
        // Stale analyzer parameters or a bucket count that doesn't match
        // the audio: not this cache's audio/format anymore — rebuild.
        if buckets_per_sec.to_bits() != BASE_BUCKETS_PER_SEC.to_bits()
            || crossover_low_bits != CROSSOVER_LOW_HZ.to_bits()
            || crossover_high_bits != CROSSOVER_HIGH_HZ.to_bits()
            || num_buckets != base_num_buckets(source_len_samples, sample_rate)
        {
            return Ok(None);
        }
    }

    let plane = |idx: usize| {
        let start = PEAKS_PARAMS_LEN + idx * num_buckets;
        &payload[start..start + num_buckets]
    };
    let pos: [Vec<f32>; NUM_BANDS] =
        std::array::from_fn(|band| plane(band).iter().map(|&q| dequantize_unit(q)).collect());
    let neg: [Vec<f32>; NUM_BANDS] = std::array::from_fn(|band| {
        plane(NUM_BANDS + band)
            .iter()
            .map(|&q| -dequantize_unit(q))
            .collect()
    });
    Ok(Some(BandPeaks::from_base_level(PeakLevel {
        buckets_per_sec,
        pos,
        neg,
    })))
}

/// Sidecar path: `<audio>.tsa` (suffix-append, so distinct source
/// extensions never collide).
pub fn analysis_file_path(audio_path: &Path) -> PathBuf {
    let mut os = audio_path.as_os_str().to_os_string();
    os.push(".tsa");
    PathBuf::from(os)
}

/// Read and structurally decode a `.tsa` file (no identity check).
pub fn read_analysis_file(path: &Path) -> Result<AnalysisFile, StretchError> {
    let bytes = std::fs::read(path)?;
    AnalysisFile::from_bytes(&bytes)
}

/// Load-boundary read: `None` on a missing, corrupt, or
/// identity-mismatched file; per-chunk gating as in
/// [`AnalysisFile::from_bytes_validated`].
pub fn read_analysis_file_validated(
    path: &Path,
    sample_rate: u32,
    source_len_samples: usize,
    content_hash: u64,
) -> Option<AnalysisFile> {
    let bytes = std::fs::read(path).ok()?;
    AnalysisFile::from_bytes_validated(&bytes, sample_rate, source_len_samples, content_hash)
}

/// Write the container atomically: serialize, write a pid-suffixed temp
/// sibling, rename over the destination. A crash mid-write can't leave a
/// truncated file shadowing the real cache.
pub fn write_analysis_file(path: &Path, file: &AnalysisFile) -> Result<(), StretchError> {
    let bytes = file.to_bytes();
    let mut temp_os = path.as_os_str().to_os_string();
    temp_os.push(format!(".tmp{}", std::process::id()));
    let temp = PathBuf::from(temp_os);
    std::fs::write(&temp, &bytes).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })?;
    std::fs::rename(&temp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    /// Unique temp dir per test (tests run in parallel in one process).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tsa_test_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// One second of mixed low+high tone, mono.
    fn test_mono() -> Vec<f32> {
        let n = SR as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / SR as f64;
                (0.7 * (std::f64::consts::TAU * 60.0 * t).sin()
                    + 0.2 * (std::f64::consts::TAU * 8_000.0 * t).sin()) as f32
            })
            .collect()
    }

    fn test_artifact(mono: &[f32]) -> PreAnalysisArtifact {
        PreAnalysisArtifact {
            version: crate::core::preanalysis::PREANALYSIS_VERSION,
            sample_rate: SR,
            bpm: 128.0,
            confidence: 0.9,
            beat_positions_fractional: vec![100.5, 200.25],
            downbeat_beat_indices: vec![0],
            source_len_samples: mono.len(),
            content_hash: hash_samples(mono),
            ..Default::default()
        }
    }

    /// A fully-populated container for the test signal.
    fn full_file() -> (AnalysisFile, Vec<f32>) {
        let mono = test_mono();
        let mut af = AnalysisFile::for_source(&mono, SR);
        af.artifact = Some(test_artifact(&mono));
        af.peaks = Some(BandPeaks::compute(&mono, 1, SR));
        (af, mono)
    }

    fn identity(af: &AnalysisFile) -> (u32, usize, u64) {
        (af.sample_rate, af.source_len_samples, af.content_hash)
    }

    #[test]
    fn roundtrip_both_chunks() {
        let (af, _) = full_file();
        let bytes = af.to_bytes();
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&bytes, sr, len, hash)
            .expect("valid container must load");
        let artifact = back.artifact.expect("artifact chunk survives");
        assert_eq!(artifact.bpm, 128.0);
        assert_eq!(artifact.beat_positions_fractional, vec![100.5, 200.25]);
        let peaks = back.peaks.expect("peaks chunk survives");
        let (a, b) = (peaks.level(0), af.peaks.as_ref().unwrap().level(0));
        assert_eq!(a.num_buckets(), b.num_buckets());
        for band in 0..NUM_BANDS {
            for (x, y) in a.pos[band].iter().zip(&b.pos[band]) {
                assert!((x - y).abs() <= 0.5 / 255.0 + f32::EPSILON, "{x} vs {y}");
            }
            for (x, y) in a.neg[band].iter().zip(&b.neg[band]) {
                assert!((x - y).abs() <= 0.5 / 255.0 + f32::EPSILON, "{x} vs {y}");
            }
        }
        // Rebuilt pyramid has the same level structure.
        assert_eq!(
            peaks.level_index_for(1.0),
            af.peaks.as_ref().unwrap().level_index_for(1.0)
        );
    }

    #[test]
    fn roundtrip_partial_and_empty_containers() {
        let mono = test_mono();
        for (with_artifact, with_peaks) in [(true, false), (false, true), (false, false)] {
            let mut af = AnalysisFile::for_source(&mono, SR);
            if with_artifact {
                af.artifact = Some(test_artifact(&mono));
            }
            if with_peaks {
                af.peaks = Some(BandPeaks::compute(&mono, 1, SR));
            }
            let (sr, len, hash) = identity(&af);
            let back = AnalysisFile::from_bytes_validated(&af.to_bytes(), sr, len, hash)
                .expect("container must load");
            assert_eq!(back.artifact.is_some(), with_artifact);
            assert_eq!(back.peaks.is_some(), with_peaks);
            assert_eq!(back.sample_rate, SR);
            assert_eq!(back.source_len_samples, mono.len());
        }
    }

    #[test]
    fn empty_track_single_bucket_roundtrips() {
        let mut af = AnalysisFile::for_source(&[], SR);
        af.peaks = Some(BandPeaks::compute(&[], 1, SR));
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&af.to_bytes(), sr, len, hash).unwrap();
        assert_eq!(back.peaks.unwrap().level(0).num_buckets(), 1);
    }

    #[test]
    fn quantize_clamps_out_of_range() {
        assert_eq!(quantize_unit(1.5), 255);
        assert_eq!(quantize_unit(-0.1), 0);
        assert_eq!(quantize_unit(0.0), 0);
        assert_eq!(quantize_unit(1.0), 255);
        assert_eq!(quantize_unit(-(-1.5f32)), 255);
        assert_eq!(quantize_unit(-(0.1f32)), 0);
    }

    #[test]
    fn missing_file_read_paths() {
        let path = Path::new("/nonexistent/x.tsa");
        assert!(read_analysis_file(path).is_err());
        assert!(read_analysis_file_validated(path, SR, 100, 1).is_none());
    }

    /// Encode a valid container, apply one mutation, assert both decode
    /// paths reject it structurally.
    fn corrupt_and_check(mutate: impl FnOnce(&mut Vec<u8>)) {
        let (af, _) = full_file();
        let mut bytes = af.to_bytes();
        mutate(&mut bytes);
        let (sr, len, hash) = identity(&af);
        assert!(AnalysisFile::from_bytes(&bytes).is_err());
        assert!(AnalysisFile::from_bytes_validated(&bytes, sr, len, hash).is_none());
    }

    #[test]
    fn wrong_magic_rejected() {
        corrupt_and_check(|b| b[0] = b'X');
    }

    #[test]
    fn wrong_container_version_rejected() {
        corrupt_and_check(|b| b[4..8].copy_from_slice(&99u32.to_le_bytes()));
    }

    #[test]
    fn truncated_file_header_rejected() {
        corrupt_and_check(|b| b.truncate(FILE_HEADER_LEN - 1));
    }

    #[test]
    fn truncated_chunk_header_rejected() {
        corrupt_and_check(|b| b.truncate(FILE_HEADER_LEN + CHUNK_HEADER_LEN - 3));
    }

    #[test]
    fn truncated_chunk_payload_rejected() {
        corrupt_and_check(|b| {
            let n = b.len();
            b.truncate(n - 7);
        });
    }

    #[test]
    fn oversized_chunk_length_rejected() {
        // Inflate the first chunk's payload_len beyond the data present.
        corrupt_and_check(|b| {
            b[FILE_HEADER_LEN + 8..FILE_HEADER_LEN + 16].copy_from_slice(&u64::MAX.to_le_bytes());
        });
    }

    #[test]
    fn peak_length_bucket_count_disagreement_rejected() {
        // A peaks-only container whose stored bucket count is off by one.
        let mono = test_mono();
        let mut af = AnalysisFile::for_source(&mono, SR);
        af.peaks = Some(BandPeaks::compute(&mono, 1, SR));
        let mut bytes = af.to_bytes();
        let count_off = FILE_HEADER_LEN + CHUNK_HEADER_LEN + 24;
        let stored = u64::from_le_bytes(bytes[count_off..count_off + 8].try_into().unwrap());
        bytes[count_off..count_off + 8].copy_from_slice(&(stored + 1).to_le_bytes());
        assert!(AnalysisFile::from_bytes(&bytes).is_err());
    }

    #[test]
    fn duplicate_chunk_rejected() {
        // Append a second copy of the ARTF chunk (chunk 1) after the end.
        let mono = test_mono();
        let mut af = AnalysisFile::for_source(&mono, SR);
        af.artifact = Some(test_artifact(&mono));
        let mut bytes = af.to_bytes();
        let chunk = bytes[FILE_HEADER_LEN..].to_vec();
        bytes.extend_from_slice(&chunk);
        assert!(AnalysisFile::from_bytes(&bytes).is_err());
    }

    #[test]
    fn unknown_chunk_tag_skipped() {
        let (af, _) = full_file();
        let mut bytes = af.to_bytes();
        push_chunk_header(&mut bytes, *b"XXXX", 7, 5);
        bytes.extend_from_slice(&[1, 2, 3, 4, 5]);
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&bytes, sr, len, hash)
            .expect("unknown chunks must be skipped");
        assert!(back.artifact.is_some());
        assert!(back.peaks.is_some());
    }

    #[test]
    fn unknown_chunk_version_skipped_as_absent() {
        // An ARTF chunk from a future envelope revision: skipped, no error.
        let mono = test_mono();
        let af = AnalysisFile::for_source(&mono, SR);
        let mut bytes = af.to_bytes();
        let payload = b"not even json";
        push_chunk_header(&mut bytes, TAG_ARTIFACT, 99, payload.len());
        bytes.extend_from_slice(payload);
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&bytes, sr, len, hash).unwrap();
        assert!(back.artifact.is_none());
    }

    #[test]
    fn garbage_artifact_json_degrades_to_none() {
        // Corrupt the ARTF payload in place: structure intact, JSON not.
        let (af, _) = full_file();
        let mut bytes = af.to_bytes();
        let json_start = FILE_HEADER_LEN + CHUNK_HEADER_LEN;
        bytes[json_start] = b'!';
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&bytes, sr, len, hash)
            .expect("file stays structurally valid");
        assert!(back.artifact.is_none(), "garbage JSON must degrade");
        assert!(back.peaks.is_some(), "peaks must survive");
        // The structural read behaves identically.
        assert!(AnalysisFile::from_bytes(&bytes).unwrap().artifact.is_none());
    }

    #[test]
    fn identity_mismatches_reject_validated_read_only() {
        let (af, _) = full_file();
        let bytes = af.to_bytes();
        let (sr, len, hash) = identity(&af);
        assert!(AnalysisFile::from_bytes_validated(&bytes, 48_000, len, hash).is_none());
        assert!(AnalysisFile::from_bytes_validated(&bytes, sr, len + 1, hash).is_none());
        assert!(AnalysisFile::from_bytes_validated(&bytes, sr, len, hash ^ 1).is_none());
        // The structural read doesn't care.
        assert!(AnalysisFile::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn stale_peaks_params_dropped_on_validated_read() {
        let (af, _) = full_file();
        let mut bytes = af.to_bytes();
        // The PEAK chunk follows the ARTF chunk; corrupt its crossover-low
        // field (payload offset 8).
        let artf_payload = u64::from_le_bytes(
            bytes[FILE_HEADER_LEN + 8..FILE_HEADER_LEN + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let peak_payload_start =
            FILE_HEADER_LEN + CHUNK_HEADER_LEN + artf_payload + CHUNK_HEADER_LEN;
        bytes[peak_payload_start + 8..peak_payload_start + 16]
            .copy_from_slice(&250.0f64.to_le_bytes());
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&bytes, sr, len, hash).unwrap();
        assert!(back.peaks.is_none(), "stale-parameter peaks must drop");
        assert!(back.artifact.is_some(), "artifact must survive");
        // Structural read keeps the peaks as stored.
        assert!(AnalysisFile::from_bytes(&bytes).unwrap().peaks.is_some());
    }

    #[test]
    fn stale_artifact_schema_dropped_on_validated_read() {
        let mono = test_mono();
        let mut af = AnalysisFile::for_source(&mono, SR);
        let mut artifact = test_artifact(&mono);
        artifact.version = 2; // pre-MIN_COMPATIBLE_VERSION
        af.artifact = Some(artifact);
        af.peaks = Some(BandPeaks::compute(&mono, 1, SR));
        let (sr, len, hash) = identity(&af);
        let back = AnalysisFile::from_bytes_validated(&af.to_bytes(), sr, len, hash).unwrap();
        assert!(back.artifact.is_none(), "incompatible schema must drop");
        assert!(back.peaks.is_some());
    }

    #[test]
    fn file_wrappers_roundtrip_and_write_atomically() {
        let dir = temp_dir("wrappers");
        let (af, _) = full_file();
        let path = dir.join("track.wav.tsa");
        write_analysis_file(&path, &af).unwrap();
        let (sr, len, hash) = identity(&af);
        assert!(read_analysis_file(&path).is_ok());
        assert!(read_analysis_file_validated(&path, sr, len, hash).is_some());
        // Only the final file — no temp left behind.
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries.len(), 1, "only the final file: {entries:?}");
        // Failed write (missing parent): Err and no stray temp anywhere.
        let bad = dir.join("missing_subdir").join("x.tsa");
        assert!(write_analysis_file(&bad, &af).is_err());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn analysis_file_path_appends_suffix() {
        assert_eq!(
            analysis_file_path(Path::new("/a/b.mp3")),
            PathBuf::from("/a/b.mp3.tsa")
        );
    }

    #[test]
    fn matches_identity_parity_with_matches_source() {
        let mono = test_mono();
        let artifact = test_artifact(&mono);
        assert!(artifact.matches_source(&mono, SR));
        assert!(artifact.matches_identity(SR, mono.len(), hash_samples(&mono)));
        assert!(!artifact.matches_identity(48_000, mono.len(), hash_samples(&mono)));
        assert!(!artifact.matches_identity(SR, mono.len() + 1, hash_samples(&mono)));
        assert!(!artifact.matches_identity(SR, mono.len(), 12345));
        // Zero bindings are skipped, exactly like matches_source.
        let unbound = PreAnalysisArtifact {
            source_len_samples: 0,
            content_hash: 0,
            ..test_artifact(&mono)
        };
        assert!(unbound.matches_identity(SR, 999, 999));
    }
}
