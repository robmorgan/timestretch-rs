# Autoresearch: streaming audio quality

## Objective
Improve the perceived audio quality of the **real-time streaming engine**
(`Engine` with `EngineProfile::Keylock`, pull-based, 256-frame callbacks) in
`timestretch-rs`. The benchmark streams real music (Music Sounds Better With
You, 124 BPM corpus WAV, 40 s segment) and synthetic fixtures through the
engine exactly like an audio callback and scores reference-free quality
metrics. Do NOT overfit to the benchmark or cheat (e.g. special-casing the
fixture, detecting the benchmark, hard-coding rates). Improvements must be
general DSP/engine quality wins.

## Metrics
- **Primary**: `quality` (0–100, higher is better) — composite:
  0.30·spec_sim + 0.30·transient_f1 + 0.20·exp(−p95_cents/10)
  + 0.10·identity_corr + 0.10·exp(−clicks/50), halved if any underruns.
- **Secondary** (tradeoff monitors):
  - `spec_sim` — mean-spectrum cosine similarity, input vs stretched (avg of 124→115 and 124→132 BPM renders). Timbre preservation.
  - `transient_f1` — F1 of onsets (input onsets mapped through rate vs output onsets, 15 ms tolerance). Transient smearing/doubling.
  - `pitch_p95_cents` / `pitch_max_cents` — 440 Hz sine under ±8%/2 s DJ ride. Keylock pitch stability.
  - `identity_corr` — xcorr peak at rate 1.0 (transparency).
  - `clicks_per_m` — discontinuities per million samples (worst of slow/fast/ride renders).
  - `underruns` — must stay 0.
  - `realtime_x` — throughput (media s / process s). Watch for perf collapse; keep > ~20x.

The benchmark is **deterministic** — identical code gives identical metrics.
Any change in `quality` is real. Noise floor ≈ 0.

## How to Run
`./.auto/measure.sh` — outputs `METRIC name=value` lines. ~15 s.
Checks: `./.auto/checks.sh` (fmt + clippy -D warnings + full release test
suite) runs automatically after passing benchmarks (~1–2 min warm).

## Files in Scope
- `src/engine/**` — real-time engine: graph, stages (varispeed, keylock chain, crossover), control, source.
- `src/stretch/**` — phase vocoder, phase locking, SOLA/envelope params.
- `src/core/**` — resample, crossover filters, windows, FFT helpers, ring buffer.
- `src/analysis/**` — ONLY if it feeds the engine path (transient-aware processing); do not tune `analysis/comparison.rs` or `analysis/transient.rs` to game the metrics.

## Off Limits
- `examples/stream_quality_bench.rs` and `.auto/**` — the benchmark itself (only touch to add instrumentation/fix bugs, never to inflate scores).
- `src/analysis/comparison.rs`, `src/analysis/transient.rs` — metric code.
- `qa/**`, `tests/**` — may not be weakened. Adding tests is fine.
- `desktop/**`.

## Constraints
- `cargo test --release`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check` must pass (enforced by checks.sh).
- No new dependencies. No unsafe (crate forbids it).
- Real-time path must stay allocation-free (tests/engine_realtime_allocations.rs gates this).
- Keep `underruns` at 0 and `realtime_x` comfortably above realtime.

## Baseline (2026-02-13, calibrated harness)
quality=95.34 | spec_sim=0.957 (slow 0.978 / fast 0.985 / slow2 0.916 / fast2 0.949)
transient_f1=0.919 (0.917/0.933/0.902/0.924) | identity_corr=0.991
pitch p95=0.43c max=0.53c | clicks=0 underruns=0 realtime_x≈55

Harness calibration notes (2026-02-13): initial harness had a latency bias
(engine reports ~13 ms constant pipeline latency; compensating raised
transient_f1 from 0.61→0.92 with zero code change), and used waveform xcorr
for identity (LR8 crossover is allpass → inaudible phase punished; switched
to latency-aligned frame-wise magnitude spectral similarity). Also added hard
ratios ±14–16% (RATE_SLOW2=104/124, RATE_FAST2=140/124) for sensitivity.

**Biggest levers**: spec_sim at hard ratios (slow2 0.916 — spectral damage at
−16%), transient_f1 across the board (~0.92). Renders use pre_analysis: None
(cold streaming path) — the SOLA onset protection runs on its online energy
heuristic only.

## What's Been Tried
(update as experiments accumulate)
