# QA Harnesses

These harnesses are excluded from default `cargo test`.

Run them explicitly with `--features qa-harnesses`, for example:

- `cargo test --features qa-harnesses --release --test benchmarks -- --nocapture`
- `cargo test --features qa-harnesses --release --test engine_ab_matrix -- --nocapture`
- `cargo test --features qa-harnesses --release --test engine_keylock -- --nocapture`
- `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --features qa-harnesses --release --test engine_wcet -- --nocapture`
- `cargo test --features qa-harnesses --test reference_quality -- --nocapture`
- `cargo test --features qa-harnesses --release --test bpm_accuracy -- --nocapture`

## Engine Keylock (`engine_keylock`)

Keylock-chain pitch-stability gate: streams a pure 440 Hz tone through the
±8%/2 s DJ ratio ride and measures instantaneous frequency deviation in
cents (interpolated zero crossings, 100 ms windows), gated absolutely on
p95/max, plus a crossover-seam re-summation gate at the band seam.

## Engine A/B Matrix (`engine_ab_matrix`)

Full metric dashboard: every gated metric family, one machine-readable
report (`ab_matrix.csv` in `TIMESTRETCH_QUALITY_DASHBOARD_DIR` or
`target/ab_matrix/`) with one row per (metric, fixture, arm) and an
absolute gate per metric.

## Engine WCET (`engine_wcet`)

Worst-case callback budget gates: per-callback wall time over the
callback's audio duration, gated on p99.9. Timing assertions are enabled
with `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1`; without it the harness
measures and prints only.

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


## Blind A/B listening harness (`scripts/ab.sh`)

Owner listening is the binding quality gate; this makes the loop cheap
when tweaking the algorithm:

```bash
# Render current working tree vs a baseline ref (plus optionally Rubber
# Band), level-matched and blinded with a sealed key:
scripts/ab.sh render my-tweak --base main --rates 0.92,1.08 --rb \
    "benchmarks/audio/bpm-corpus/<track>.wav:90"

# Listen in the blind TUI (hot-switch arms position-synced with a-e or
# space on the hovered arm, S = source reference, looping is on by
# default — l turns it off, Enter to note an arm (←/→ edit the note),
# w to pick a winner, Ctrl-S to save):
scripts/ab.sh listen my-tweak
```

Saving writes `target/ab/my-tweak/results.json` — per condition the
letter-keyed notes, the winner, and the letter→arm mapping merged from
the sealed key at save time (pass `--no-unblind` to the `ab-tui` binary
to keep it sealed). The results path is the TUI's last stdout line, so
a driving LLM can launch the tool, wait for exit, and parse the file.
Re-launching with an existing results file resumes the session;
`scripts/ab.sh unblind` remains for key inspection without the TUI.

Every arm gets identical treatment (RMS-matched to source, one common
no-clip trim per condition, 32-bit float, per-condition shuffled
letters) — the validity rules learned in the Stage 16 review (mono arms,
level deltas, and clipping all unblind a set). The baseline builds in a
temporary git worktree, so any historical ref can be an arm. Prerendered
full-track references (Elastique exports from an Ableton host) join a
set with `--ref-arm <label>:<dir>`; the export protocol and directory
layout are in `benchmarks/README.md` under "Elastique reference
renders".
