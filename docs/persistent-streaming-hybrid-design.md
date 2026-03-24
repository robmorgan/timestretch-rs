# Persistent Streaming Hybrid Engine — Migration Design Note

> Internal reference for contributors. This document describes the migration
> strategy from the legacy rolling-window hybrid re-render path to the new
> persistent streaming hybrid engine.

## Status

**Scaffolding only.** The new modules compile but contain no runtime logic.
The existing `StreamProcessor` and its two engines (`Deterministic`,
`LegacyHybridRerender`) are unchanged.

## Architecture overview

```
input audio
    │
    ▼
┌──────────────┐
│StreamAnalyzer │  ← spectral + transient analysis, timeline-addressed
└──────┬───────┘
       │ Vec<AnalysisEvent>
       ▼
┌─────────────┐
│ HybridRouter │  ← decides tonal vs transient vs blend per region
└──────┬──────┘
       │ Vec<RoutedHybridOp>
       ▼
┌──────────────────────┐   ┌───────────────────────────┐
│PersistentTonalRenderer│   │PersistentTransientRenderer │
└──────────┬───────────┘   └────────────┬──────────────┘
           │ Vec<ScheduledPatch>         │
           └──────────┬─────────────────┘
                      ▼
               ┌─────────────┐
               │TimelineMixer │  ← overlap-aware merge, interleave
               └──────┬──────┘
                      │
                      ▼
               output buffer
```

## Key design shift: timeline-based, not chunk-based

The legacy `LegacyHybridRerender` engine operates per-chunk: it snapshots
buffered history, re-runs `HybridStretcher` from scratch, skips
already-emitted output, and crossfades the delta. This has several drawbacks:

1. **Redundant work** — re-rendering the entire rolling window every callback.
2. **Phase discontinuities** — fresh PV state on each re-render requires a
   `HYBRID_STREAM_CROSSFADE_SAMPLES`-wide crossfade to hide seams.
3. **Callback-size coupling** — quality varies with host buffer size.

The new engine addresses all three by making every stage timeline-addressed:

- `AnalysisEvent.input_pos` — absolute input sample offset.
- `RoutedHybridOp.input_start` — absolute region boundary.
- `ScheduledPatch.output_pos` — absolute output sample offset.

Because renderers maintain persistent state (phase accumulators, overlap
buffers, correlation search position), there is no re-rendering and no
chunk-boundary crossfade.

## Migration strategy

1. **Scaffolding (this step).** Add module files with types and docs. No
   runtime changes.
2. **Wire analyzer.** Replace the existing transient-detection call in the
   hybrid path with `StreamAnalyzer`, keeping output identical.
3. **Wire router.** Introduce `HybridRouter` behind the existing
   `LegacyHybridRerender` code path so routing decisions can be A/B tested.
4. **Wire renderers.** Implement `PersistentTonalRenderer` and
   `PersistentTransientRenderer` using the existing `PhaseVocoder` and `Wsola`
   primitives, but with persistent state.
5. **Wire mixer.** Replace the delta + crossfade logic with `TimelineMixer`.
6. **New engine variant.** Expose `StreamingEngine::PersistentHybrid` and
   default to it for new streams.
7. **Deprecate legacy.** Mark `LegacyHybridRerender` as deprecated; remove
   after one release cycle.

Each step is independently testable and the old engine remains available as a
fallback throughout.

## Module locations

| Module | File | Key type(s) |
|--------|------|-------------|
| Analyzer | `src/stream/analyzer.rs` | `StreamAnalyzer`, `AnalysisEvent` |
| Router | `src/stream/router.rs` | `HybridRouter`, `RoutedHybridOp`, `RenderPath` |
| Renderers | `src/stream/render.rs` | `PersistentTonalRenderer`, `PersistentTransientRenderer`, `ScheduledPatch` |
| Mixer | `src/stream/mixer.rs` | `TimelineMixer` |
