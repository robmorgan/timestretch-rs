# QA Harnesses

These harnesses are excluded from default `cargo test`.

Run them explicitly with `--features qa-harnesses`, for example:

- `cargo test --features qa-harnesses --release --test benchmarks -- --nocapture`
- `cargo test --features qa-harnesses --test quality_gates -- --nocapture`
- `cargo test --features qa-harnesses --test reference_quality -- --nocapture`
- `cargo test --features qa-harnesses --test quality_benchmark -- --ignored --nocapture`
- `cargo test --features qa-harnesses --release --test bpm_accuracy -- --nocapture`
- `cargo test --features qa-harnesses --release --test varispeed_keylock -- --nocapture`

## Varispeed Keylock (`varispeed_keylock`)

Stage 15 pitch-stability gate: streams a pure 440 Hz tone through the
±8%/2 s DJ ratio ride on both control paths and measures instantaneous
frequency deviation in cents (interpolated zero crossings, 100 ms windows).
The varispeed-first path is gated absolutely on the Live profile (p95/max)
and relative to the vocoder-tempo baseline on every profile; the baseline
rows are printed for comparison.

## BPM Accuracy (`bpm_accuracy`)

Scores BPM detection against every `[[track]]` in `benchmarks/manifest.toml`
(including `bpm_only = true` corpus entries, which may be `.wav`, `.mp3`, or
`.aiff`). Each detected tempo is classified as EXACT (within tolerance),
OCTAVE (within tolerance of 1/2x, 2x, 1/3x, or 3x), WRONG, or FAILED. The
headline scores are `acc1` (% EXACT) and `acc2` (% EXACT or OCTAVE); a JSON
report is written to `target/bpm_accuracy_report.json` for diffing between
detector changes. See `benchmarks/README.md` for how to add corpus tracks.

Environment variables:

- `TIMESTRETCH_BPM_TOLERANCE` — relative tolerance (default `0.02` = ±2%)
- `TIMESTRETCH_BPM_MAX_SECONDS` — trim each track before analysis
- `TIMESTRETCH_STRICT_BPM_BENCHMARK=1` — missing files and skips become failures
- `TIMESTRETCH_BPM_MIN_ACC1` / `TIMESTRETCH_BPM_MIN_ACC2` — accuracy floors (0–100); the test fails below them
