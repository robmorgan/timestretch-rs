//! Persisted waveform-peaks sidecar.
//!
//! A track's base pyramid level is written next to the audio file as
//! `<audio>.tspeaks`: a fixed little-endian header followed by six planes
//! of u8-quantized per-band peaks. Only the base level is stored; the
//! upper levels rebuild in microseconds via the halving pyramid. The file
//! is keyed by the decoded mono signal's identity (length + FNV hash, the
//! same scheme as the `.tsanalysis.json` sidecar), so renamed or retagged
//! files keep their analysis. Any mismatch, truncation, or corruption
//! reads as a miss — never an error the UI has to handle.

use std::path::{Path, PathBuf};

use super::peaks::{
    BASE_BUCKETS_PER_SEC, BandPeaks, CROSSOVER_HIGH_HZ, CROSSOVER_LOW_HZ, NUM_BANDS, PeakLevel,
    base_num_buckets,
};

const MAGIC: [u8; 4] = *b"TSPK";
const FORMAT_VERSION: u32 = 1;
/// Header size in bytes; the six N-byte planes follow immediately.
const HEADER_LEN: usize = 60;

/// Sidecar peaks-cache path: `<audio>.tspeaks` (suffix-append, like the
/// `.tsanalysis.json` sidecar, so distinct extensions never collide).
pub fn peaks_cache_path(audio_path: &Path) -> PathBuf {
    let mut os = audio_path.as_os_str().to_os_string();
    os.push(".tspeaks");
    PathBuf::from(os)
}

/// `v` in `[0, 1]` to a u8 step; out-of-range clamps.
fn quantize_unit(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn dequantize_unit(q: u8) -> f32 {
    q as f32 / 255.0
}

/// Read and validate a cached pyramid for a mono source of
/// `source_len_samples` frames hashing to `content_hash`. Returns `None`
/// on any mismatch, truncation, or corruption — the caller recomputes.
pub fn read_validated(
    path: &Path,
    sample_rate: u32,
    source_len_samples: usize,
    content_hash: u64,
) -> Option<BandPeaks> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < HEADER_LEN {
        return None;
    }
    let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());

    if bytes[0..4] != MAGIC || u32_at(4) != FORMAT_VERSION || u32_at(8) != sample_rate {
        return None;
    }
    // Analysis parameters compare by bit pattern: a cache built with
    // different buckets-per-sec or crossovers renders differently and
    // must be rebuilt.
    if u64_at(12) != BASE_BUCKETS_PER_SEC.to_bits()
        || u64_at(20) != CROSSOVER_LOW_HZ.to_bits()
        || u64_at(28) != CROSSOVER_HIGH_HZ.to_bits()
    {
        return None;
    }
    let num_buckets = u64_at(36) as usize;
    // The bucket count must match what the source length implies — this
    // also bounds the allocation below to a value derived from the
    // caller's trusted source length, not from file contents.
    if num_buckets != base_num_buckets(source_len_samples, sample_rate)
        || u64_at(44) != source_len_samples as u64
        || u64_at(52) != content_hash
    {
        return None;
    }
    // Exact length: rejects truncation and trailing garbage alike.
    if bytes.len() != HEADER_LEN + 6 * num_buckets {
        return None;
    }

    let plane = |idx: usize| {
        let start = HEADER_LEN + idx * num_buckets;
        &bytes[start..start + num_buckets]
    };
    let pos: [Vec<f32>; NUM_BANDS] =
        std::array::from_fn(|band| plane(band).iter().map(|&q| dequantize_unit(q)).collect());
    let neg: [Vec<f32>; NUM_BANDS] = std::array::from_fn(|band| {
        plane(NUM_BANDS + band)
            .iter()
            .map(|&q| -dequantize_unit(q))
            .collect()
    });
    Some(BandPeaks::from_base_level(PeakLevel {
        buckets_per_sec: BASE_BUCKETS_PER_SEC,
        pos,
        neg,
    }))
}

/// Write the base level of `peaks` atomically (temp sibling + rename), so
/// a crash mid-write can't leave a truncated file that shadows the real
/// cache until the next hash change.
pub fn write(
    path: &Path,
    peaks: &BandPeaks,
    sample_rate: u32,
    source_len_samples: usize,
    content_hash: u64,
) -> std::io::Result<()> {
    let base = peaks.level(0);
    let num_buckets = base.num_buckets();

    let mut bytes = Vec::with_capacity(HEADER_LEN + 6 * num_buckets);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&BASE_BUCKETS_PER_SEC.to_le_bytes());
    bytes.extend_from_slice(&CROSSOVER_LOW_HZ.to_le_bytes());
    bytes.extend_from_slice(&CROSSOVER_HIGH_HZ.to_le_bytes());
    bytes.extend_from_slice(&(num_buckets as u64).to_le_bytes());
    bytes.extend_from_slice(&(source_len_samples as u64).to_le_bytes());
    bytes.extend_from_slice(&content_hash.to_le_bytes());
    debug_assert_eq!(bytes.len(), HEADER_LEN);
    for band in 0..NUM_BANDS {
        bytes.extend(base.pos[band].iter().map(|&v| quantize_unit(v)));
    }
    for band in 0..NUM_BANDS {
        bytes.extend(base.neg[band].iter().map(|&v| quantize_unit(-v)));
    }

    // Pid-suffixed temp name so two app instances can't collide.
    let mut temp_os = path.as_os_str().to_os_string();
    temp_os.push(format!(".tmp{}", std::process::id()));
    let temp = PathBuf::from(temp_os);
    std::fs::write(&temp, &bytes).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })?;
    std::fs::rename(&temp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    /// Unique temp dir per test (parallel test threads share the process).
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tspeaks_test_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A second of mixed low+high tone, mono, plus its identity.
    fn test_source() -> (Vec<f32>, u64) {
        let n = SR as usize;
        let mono: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / SR as f64;
                (0.7 * (std::f64::consts::TAU * 60.0 * t).sin()
                    + 0.2 * (std::f64::consts::TAU * 8_000.0 * t).sin()) as f32
            })
            .collect();
        let hash = timestretch::hash_samples(&mono);
        (mono, hash)
    }

    /// Write a valid cache file for the test source; returns
    /// (cache path, original peaks, mono length, hash).
    fn valid_file(dir: &Path) -> (PathBuf, BandPeaks, usize, u64) {
        let (mono, hash) = test_source();
        let peaks = BandPeaks::compute(&mono, 1, SR);
        let path = dir.join("track.wav.tspeaks");
        write(&path, &peaks, SR, mono.len(), hash).unwrap();
        (path, peaks, mono.len(), hash)
    }

    #[test]
    fn roundtrip_within_quantization_error() {
        let dir = temp_dir("roundtrip");
        let (path, original, len, hash) = valid_file(&dir);
        let loaded = read_validated(&path, SR, len, hash).expect("valid file must load");
        let (a, b) = (loaded.level(0), original.level(0));
        assert_eq!(a.num_buckets(), b.num_buckets());
        for band in 0..NUM_BANDS {
            for (x, y) in a.pos[band].iter().zip(&b.pos[band]) {
                assert!((x - y).abs() <= 0.5 / 255.0 + f32::EPSILON, "{x} vs {y}");
            }
            for (x, y) in a.neg[band].iter().zip(&b.neg[band]) {
                assert!((x - y).abs() <= 0.5 / 255.0 + f32::EPSILON, "{x} vs {y}");
            }
        }
        // The rebuilt pyramid has the same shape as a computed one.
        assert_eq!(
            loaded.level_index_for(1.0),
            original.level_index_for(1.0),
            "coarsest level should match"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_clamps_out_of_range() {
        assert_eq!(quantize_unit(1.5), 255);
        assert_eq!(quantize_unit(-0.1), 0);
        assert_eq!(quantize_unit(0.0), 0);
        assert_eq!(quantize_unit(1.0), 255);
        // Negative peaks are stored as magnitudes: a hot -1.5 clamps full.
        assert_eq!(quantize_unit(-(-1.5f32)), 255);
        assert_eq!(quantize_unit(-(0.1f32)), 0);
    }

    #[test]
    fn single_bucket_empty_track_roundtrips() {
        let dir = temp_dir("empty");
        let peaks = BandPeaks::compute(&[], 1, SR);
        let hash = timestretch::hash_samples(&[]);
        let path = dir.join("empty.wav.tspeaks");
        write(&path, &peaks, SR, 0, hash).unwrap();
        let loaded = read_validated(&path, SR, 0, hash).expect("empty track must roundtrip");
        assert_eq!(loaded.level(0).num_buckets(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_returns_none() {
        assert!(read_validated(Path::new("/nonexistent/x.tspeaks"), SR, 100, 1).is_none());
    }

    /// Each corruption writes a valid file, applies one mutation, and
    /// asserts rejection.
    fn corrupt_and_check(tag: &str, mutate: impl FnOnce(&mut Vec<u8>)) {
        let dir = temp_dir(tag);
        let (path, _, len, hash) = valid_file(&dir);
        let mut bytes = std::fs::read(&path).unwrap();
        mutate(&mut bytes);
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            read_validated(&path, SR, len, hash).is_none(),
            "corrupted file ({tag}) must be rejected"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_magic_rejected() {
        corrupt_and_check("magic", |b| b[0] = b'X');
    }

    #[test]
    fn wrong_version_rejected() {
        corrupt_and_check("version", |b| b[4..8].copy_from_slice(&99u32.to_le_bytes()));
    }

    #[test]
    fn crossover_mismatch_rejected() {
        corrupt_and_check("crossover", |b| {
            b[20..28].copy_from_slice(&250.0f64.to_le_bytes())
        });
    }

    #[test]
    fn truncated_header_rejected() {
        corrupt_and_check("hdr_trunc", |b| b.truncate(30));
    }

    #[test]
    fn truncated_payload_rejected() {
        corrupt_and_check("payload_trunc", |b| {
            let n = b.len();
            b.truncate(n - 7);
        });
    }

    #[test]
    fn trailing_garbage_rejected() {
        corrupt_and_check("trailing", |b| b.extend_from_slice(&[0u8; 3]));
    }

    #[test]
    fn sample_rate_mismatch_rejected() {
        let dir = temp_dir("sr");
        let (path, _, len, hash) = valid_file(&dir);
        assert!(read_validated(&path, 48_000, len, hash).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_mismatch_rejected() {
        let dir = temp_dir("hash");
        let (path, _, len, hash) = valid_file(&dir);
        assert!(read_validated(&path, SR, len, hash ^ 1).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn source_len_mismatch_rejected() {
        let dir = temp_dir("len");
        let (path, _, len, hash) = valid_file(&dir);
        assert!(read_validated(&path, SR, len + 1, hash).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_leaves_no_temp_file() {
        let dir = temp_dir("no_temp");
        let (path, ..) = valid_file(&dir);
        let entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries.len(), 1, "only the final file: {entries:?}");
        assert_eq!(entries[0], path.file_name().unwrap());

        // Failed write (parent doesn't exist): Err, and no stray temp in
        // any existing directory.
        let peaks = BandPeaks::compute(&[], 1, SR);
        let bad = dir.join("missing_subdir").join("x.tspeaks");
        assert!(write(&bad, &peaks, SR, 0, 0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peaks_cache_path_appends_suffix() {
        assert_eq!(
            peaks_cache_path(Path::new("/a/b.mp3")),
            PathBuf::from("/a/b.mp3.tspeaks")
        );
    }
}
