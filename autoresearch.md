# Autoresearch: Streaming Algorithm Quality Improvement

## Objective
Improve the audio quality of the streaming time-stretch algorithm (`StreamProcessor`). The streaming path currently uses PV-only processing (no hybrid WSOLA+PV), which produces significantly worse quality on percussive/transient-heavy content compared to the batch `stretch()` function. We measure quality by comparing streaming output against batch (hybrid) output as reference using spectral similarity, perceptual weighting, cross-correlation, and spectral flux metrics.

## Metrics
- **Primary**: `quality_score` (unitless 0-1000, higher is better) — weighted composite of spectral similarity, perceptual spectral similarity, cross-correlation, and spectral flux similarity, averaged across 8 test cases (3 signal types × 2-3 ratios)
- **Secondary**: `percussive_score` — average composite for percussive-only cases (weakest area)
- **Secondary**: `edm_score` — average composite for EDM signal cases
- **Secondary**: `harmonic_score` — average composite for harmonic signal cases

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `src/stream/processor.rs` — Real-time streaming processor (ring buffers, PV orchestration, ratio smoothing) — **primary target**
- `src/stream/transient_scheduler.rs` — Transient event scheduling for streaming
- `src/stretch/phase_vocoder.rs` — Phase vocoder core (shared between batch and streaming)
- `src/stretch/phase_locking.rs` — Phase locking algorithms
- `src/stretch/hybrid.rs` — Batch hybrid stretcher (reference implementation, read for understanding)
- `src/stretch/wsola.rs` — WSOLA time-domain stretching (could be integrated into streaming)
- `src/analysis/transient.rs` — Transient detection
- `src/analysis/adaptive_snapshot.rs` — Adaptive analysis for segmentation
- `src/core/types.rs` — Parameters and presets

## Off Limits
- Test files (`tests/`, `qa/`) — except `qa/streaming_quality.rs` which is our benchmark
- Batch `stretch()` function behavior — must not change (it's the reference)
- `Cargo.toml` dependencies — no new deps
- Existing passing tests — must continue to pass

## Constraints
- `cargo test --lib` must pass (780 pass, ≤9 known failures)
- `cargo fmt --all --check` must pass
- No `unsafe` code (`#![forbid(unsafe_code)]`)
- No new dependencies
- Real-time safety: no heap allocations in the streaming `process_into` callback path after init
- Streaming latency must not increase significantly (currently ~35-139ms depending on profile)
- Changes should improve ALL signal types or at least not regress any

## What's Been Tried
(Updated as experiments accumulate)

### Baseline
- quality_score=490.7: PV-only streaming, no transient handling
  - Harmonic: ~0.636
  - EDM: ~0.617
  - Percussive: ~0.268 (weakest — no WSOLA transient preservation)
