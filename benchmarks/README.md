# Reference Audio Benchmarks

This directory contains a benchmark system for comparing time-stretch quality against professional DJ software (Ableton Live, Serato, Traktor, etc.).

## Directory Layout

```
benchmarks/
  manifest.toml          # Track + reference descriptions (checked into git)
  README.md              # This file
  audio/                 # NOT checked into git
    originals/           # Place original WAV files here
    references/          # Place professionally-stretched WAVs here
    output/              # Library output written during benchmark runs
```

## Setup

1. Create the audio directories:

   ```bash
   mkdir -p benchmarks/audio/originals benchmarks/audio/references
   ```

2. Place your original WAV files in `benchmarks/audio/originals/`.

3. Time-stretch each original in your professional software of choice and save the result to `benchmarks/audio/references/`.

4. Edit `benchmarks/manifest.toml` to describe your tracks and references. Each track entry needs:
   - `id` - A short identifier (used in filenames)
   - `description` - Human-readable description
   - `original` - Path relative to `benchmarks/audio/` (e.g., `originals/my-track.wav`)
   - `original_sha256` - SHA-256 of the original WAV (for corpus lock)
   - `bpm` - Original tempo in BPM

   Each reference entry needs:
   - `file` - Path relative to `benchmarks/audio/`
   - `file_sha256` - SHA-256 of the reference WAV (for corpus lock)
   - `target_bpm` - Target tempo the reference was stretched to
   - `software` - Software used (e.g., "Ableton Live 11")
   - `algorithm` - Algorithm/mode used (e.g., "Complex Pro")

## Running Benchmarks

```bash
# M0 baseline command (strict validation + archive)
./benchmarks/run_m0_baseline.sh
```

This command:

- Enables strict validation (`TIMESTRETCH_STRICT_REFERENCE_BENCHMARK=1`)
- Uses a fixed 30-second analysis window (`TIMESTRETCH_REFERENCE_MAX_SECONDS=30`)
- Fails on missing files, invalid paths, or checksum mismatches
- Processes each original with all 5 EDM presets
- Prints timing/spectral/transient/loudness/length metrics
- Writes JSON metrics to `benchmarks/audio/output/report.json`
- Archives the baseline to `benchmarks/baselines/`

For ad-hoc runs without strict enforcement:

```bash
cargo test --test reference_quality -- --nocapture
```

You can optionally set `TIMESTRETCH_REFERENCE_MAX_SECONDS=<seconds>` for deterministic
short-window analysis during ad-hoc runs as well.

## CI Quality Gate Subset

CI enforces a corpus-independent benchmark subset via:

```bash
cargo test --test quality_gates -- --nocapture
```

This test uses synthetic DJ-like material and fails on regressions for:

- duration error
- transient alignment
- cross-correlation timing coherence
- loudness deviation
- spectral similarity by band

## Metrics

| Metric | Description |
|--------|-------------|
| Length accuracy | Sample count deviation from reference |
| RMS difference | Overall loudness comparison in dB |
| Spectral similarity | STFT cosine similarity (0.0-1.0) |
| Band spectral similarity | Per-band STFT cosine similarity (sub-bass/low/mid/high) |
| Transient match rate | Percentage of reference onsets matched within 10ms |
| Cross-correlation | Peak correlation value and timing offset |

## Notes

- Audio files are gitignored. Each contributor must supply their own copies.
- The benchmark uses copyrighted audio that cannot be distributed.
- Output files are written to `benchmarks/audio/output/` and also gitignored.
