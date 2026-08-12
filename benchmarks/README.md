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
    bpm-corpus/          # Place BPM-accuracy corpus tracks here (wav/mp3/aiff)
    public-corpus/       # Redistributable CC corpus (scripts/fetch_public_corpus.sh)
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

## BPM Accuracy Corpus

The BPM accuracy harness (`qa/bpm_accuracy.rs`) scores tempo detection against
every manifest track with a `bpm` field. To grow the corpus beyond the
reference tracks, add `bpm_only = true` entries — these carry no stretched
references, may be `.wav`, `.mp3`, or `.aiff`, and are ignored by the
reference quality benchmark:

1. Drop the audio file under `benchmarks/audio/bpm-corpus/`.
2. Compute its hash: `shasum -a 256 benchmarks/audio/bpm-corpus/<file>`.
3. Add a manifest entry:

   ```toml
   [[track]]
   id = "dark-techno-01"
   description = "Dark techno, 140 BPM"
   original = "bpm-corpus/dark_techno_01.mp3"
   original_sha256 = "<sha256 of audio file>"
   bpm = 140.0
   bpm_only = true
   ```

4. Run the harness:

   ```bash
   cargo test --features qa-harnesses --release --test bpm_accuracy -- --nocapture
   ```

Per-track `METRIC` lines and a `SUMMARY` line (acc1 = % within ±2%,
acc2 = % allowing octave errors) are printed, and a JSON report is written to
`target/bpm_accuracy_report.json` for diffing between detector changes. See
`qa/README.md` for the harness's environment variables.

## QM Vamp Baseline Column (external beat-tracker reference)

Annotated tracks can carry a committed baseline from the QM bar/beat
tracker (`qm-vamp-plugins`, the Queen Mary reference implementation) so
the harness reports how an established external tracker scores against
the same hand-corrected annotations. This is the ROADMAP Stage 10 exit
criterion "no row where the QM baseline wins by more than noise".

Only the plugin's **output** (beat/downbeat times, JSON under
`benchmarks/baselines/qm/<track-id>.json`) is committed — qm-dsp is GPL
and its source never enters this repository.

Generate or refresh baselines for whatever corpus audio is present
locally:

```bash
python3 benchmarks/generate_qm_baseline.py            # all present tracks
python3 benchmarks/generate_qm_baseline.py <track-id> # one track
```

Requirements (macOS):

- `sonic-annotator` on PATH (or `$SONIC_ANNOTATOR`) — universal binary
  from <https://github.com/sonic-visualiser/sonic-annotator/releases>
- `qm-vamp-plugins.dylib` in `~/Library/Audio/Plug-Ins/Vamp/` — build
  from <https://github.com/c4dm/qm-vamp-plugins> (with `qm-dsp` and
  `vamp-plugin-sdk` cloned into `lib/`, `make -f build/osx/Makefile.osx`)

The harness picks baselines up automatically: each annotated track with
a baseline gains a `qm_beat_f` / `qm_downbeat_f` / `qm_beat_f_margin`
METRIC line and the summary reports `qm_baselined`, mean `qm_beat_f`,
and `max_qm_beat_f_margin` (the worst row's QM-minus-ours beat F, in
percentage points — negative means our tracker wins every row). Setting
`TIMESTRETCH_BPM_QM_MARGIN=<pp>` turns the margin into a hard floor.

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
cargo test --features qa-harnesses --test reference_quality -- --nocapture
```

You can optionally set `TIMESTRETCH_REFERENCE_MAX_SECONDS=<seconds>` for deterministic
short-window analysis during ad-hoc runs as well.

### External Rubber Band Scenario

For a focused one-off benchmark against a Rubber Band render (black-box reference),
provide paths through environment variables:

```bash
TIMESTRETCH_RUBBERBAND_ORIGINAL_WAV=benchmarks/audio/originals/my-loop.wav \
TIMESTRETCH_RUBBERBAND_REFERENCE_WAV=benchmarks/audio/references/my-loop_rubberband.wav \
TIMESTRETCH_RUBBERBAND_RATIO=1.024 \
cargo test --features qa-harnesses --test rubberband_comparison -- --nocapture
```

Optional variables:

- `TIMESTRETCH_RUBBERBAND_SOURCE_BPM` + `TIMESTRETCH_RUBBERBAND_TARGET_BPM` (used if `..._RATIO` is not set)
- `TIMESTRETCH_RUBBERBAND_MAX_SECONDS` (default `20.0`, trims analysis window for determinism)

Outputs are written to `target/rubberband_benchmark/`:

- `rubberband_comparison_report.csv`
- `timestretch_vs_rubberband_output.wav`

## CI Quality Gate Subset

CI enforces corpus-independent quality gates via the engine harnesses:

```bash
cargo test --features qa-harnesses --release --test engine_ab_matrix -- --nocapture
cargo test --features qa-harnesses --release --test engine_keylock -- --nocapture
```

CI also enforces a strict worst-case callback-time budget gate in release mode:

```bash
TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --features qa-harnesses --release --test engine_wcet -- --nocapture
```

To emit machine-readable dashboard artifacts locally:

```bash
TIMESTRETCH_QUALITY_DASHBOARD_DIR=target/quality_dashboard cargo test --features qa-harnesses --release --test engine_ab_matrix -- --nocapture
```

This writes the `ab_matrix.csv` dashboard under `target/quality_dashboard/`.

These harnesses use synthetic DJ-like material and fail on regressions for:

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

## Public Corpus (redistributable)

The public corpus (ROADMAP Stage 7) is a set of Creative Commons netlabel
club tracks hosted on the Internet Archive — every license permits
redistribution and derivative works (CC0 / CC-BY / CC-BY-SA), so stretched
renders and CI artifacts built from it are safe to publish with
attribution. Fetch it with:

```bash
scripts/fetch_public_corpus.sh
```

Every file is verified against a pinned SHA-256. The scored entries carry
tempo ground truth confirmed by two independent estimators (the repo
detector and a coherent onset-envelope Goertzel scan; all scored tracks
peak at an exact integer BPM, consistent across four 60 s windows).
Entries with `bpm = 0.0` are rhythmically ambiguous (swung UK garage,
broken techno) and await ear verification — the BPM harness skips them,
but they remain valuable listening material.

CI fetches this corpus (cached) and runs `qa/bpm_accuracy.rs` against it
on every push — see the `public-corpus` job in `.github/workflows/ci.yml`.
The tracks in `bpm-corpus/` are commercial purchases for local runs only
and are never fetched or redistributed.
