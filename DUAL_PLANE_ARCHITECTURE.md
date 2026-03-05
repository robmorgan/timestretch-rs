# Dual-Plane Target Architecture

## Goals
- Move adaptive analysis/policy out of callback execution.
- Keep callback CPU cost bounded with fixed-size kernels.
- Replace ratio-only control with continuous warp-map control.
- Support deterministic quality tiering under overload.

## Module Boundaries
- `src/dual_plane/rt.rs`
  - Hard-RT callback plane.
  - Strict API: `prepare()`, `process_block()`, `flush()`.
  - Fixed-capacity rings and preallocated lane buffers.
  - Poll-only control updates (`try_recv`) and no blocking calls.
- `src/dual_plane/analysis_plane.rs`
  - Async analysis workers.
  - Computes transient, beat, tonal/noise confidence and publishes snapshots.
- `src/dual_plane/warp_map.rs`
  - Continuous `t_in -> t_out` mapping with anchors.
  - Piecewise-linear local slope for per-kernel ratio control.
- `src/dual_plane/quality.rs`
  - `Q0..Q4` ladder and callback-budget governor.
  - Latency profiles: `Scratch`, `Mix`, `Render`.
- `src/dual_plane/engine.rs`
  - Orchestrator joining RT and analysis planes.
  - Optional convenience for async analysis submission.

## Thread Model
- Audio callback thread:
  - Calls `RtProcessor::process_block`.
  - Runs fixed-size kernel pass at most once per callback.
  - Consumes latest snapshots via lock-free `try_recv`.
- Analysis worker thread(s):
  - Receives copied analysis jobs.
  - Computes hints and publishes immutable `Arc<RenderHints>`.
- Control producer thread(s):
  - Publishes warp-map and hint snapshots via `RtControlSender`.
  - No direct synchronization with callback thread.

## Data Flow
1. Input block enters RT input ring.
2. RT plane polls latest warp/hint snapshots.
3. Warp-map slope determines kernel ratio (`ratio_over_range`).
4. Multi-lane render runs:
   - tonal lane: phase vocoder core
   - transient lane: time-domain resample path
   - residual lane: decorrelated high-pass residual
5. Governor updates quality tier based on callback budget.
6. Tier changes crossfade lane weights across profile-defined block counts.

## Overload Policy
- Input/output ring pressure is handled without blocking.
- On overflow:
  - Oldest ring data is discarded to preserve callback completion.
  - Governor is forced to demote one tier.
- Tier transitions are smoothed by lane-weight crossfades.

## Latency Profiles
- `Scratch`: lower startup tier + short tier crossfade window.
- `Mix`: balanced startup tier and crossfade window.
- `Render`: highest startup tier + longest crossfade window.

## Migration Plan
1. Introduce dual-plane modules in parallel with `StreamProcessor` (done).
2. Add parity tests between `StreamProcessor` deterministic mode and RT plane.
3. Route `StreamingEngine::Deterministic` to dual-plane RT core behind opt-in flag.
4. Move remaining adaptive logic from `src/stretch/hybrid.rs` into analysis snapshots.
5. Add CI gates for callback p99/p999, RT heap activity, long-run drift, and fast-modulation artifacts.

## Notes
- The legacy monolithic stream path remains unchanged for compatibility.
- The new architecture is additive and can be switched in incrementally.
