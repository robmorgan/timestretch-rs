# Plan: Persistent Stream Hybrid Build Prompts

These prompts are designed to be run sequentially with Claude Opus 4.6.

Guidelines:
- Run one prompt at a time.
- Review the diff after each step before moving on.
- Keep each step bounded; do not ask the model to rewrite the whole stream stack in one pass.
- Keep the current deterministic engine and legacy hybrid engine available during the migration.

## Prompt 1: Scaffolding

```text
You are working in the `timestretch-rs` repository.

Task: prepare the codebase for a new persistent streaming hybrid engine, but do not change runtime behavior yet.

Context:
- The current streaming path lives in `src/stream/processor.rs`.
- The current "hybrid" stream mode is a legacy rolling-window re-render path that snapshots buffered history, re-runs `HybridStretcher`, skips already-emitted output, and crossfades deltas.
- I want to replace that with a true persistent streaming hybrid engine built from:
  - `src/stream/analyzer.rs`
  - `src/stream/router.rs`
  - `src/stream/render.rs`
  - `src/stream/mixer.rs`

Constraints:
- Do not change the public behavior of `StreamProcessor` in this prompt.
- Do not add a new public `StreamingEngine` variant yet.
- Do not remove or refactor the existing deterministic or legacy hybrid engine beyond what is needed for scaffolding.
- Do not revert unrelated local changes.
- Keep the diff focused and compilable.

What to implement:
- Add the new module files under `src/stream/`.
- Export the modules from `src/stream/mod.rs` only as needed for internal use.
- Define conservative initial internal types and docs, for example:
  - `StreamAnalyzer`
  - `HybridRouter`
  - `PersistentTonalRenderer`
  - `PersistentTransientRenderer`
  - `TimelineMixer`
  - `AnalysisEvent`
  - `RoutedHybridOp`
  - `ScheduledPatch`
- Add clear module-level docs describing the intended responsibilities and invariants.
- Add a short internal design note as Rust doc comments or a small markdown file under the repo root, explaining the migration strategy:
  - old engine stays in place
  - new engine will be integrated incrementally
  - decisions become timeline-based, not chunk-based

Acceptance criteria:
- The project compiles.
- No runtime behavior changes.
- The new modules exist with coherent interfaces and comments.
- The scaffolding makes the later prompts easier rather than adding abstractions with no purpose.

Validation:
- Run the smallest reasonable test/build commands to confirm the crate still compiles.
- In your final message, list the files you changed and summarize the scaffolding decisions.
```

## Prompt 2: Analyzer

```text
You are working in the `timestretch-rs` repository.

Task: implement the first usable version of `src/stream/analyzer.rs`.

Goal:
Build a bounded rolling analyzer that produces absolute, monotonic transient/control events for the stream. This analyzer is allowed to re-analyze a bounded recent horizon internally, but it must emit deduplicated events on an absolute timeline.

Important:
- Do not try to build a mathematically perfect fully-incremental onset detector in this prompt.
- It is acceptable to re-run analysis on the last ~0.5-2.0 seconds of audio inside the analyzer, as long as:
  - the cost is bounded,
  - the horizon is fixed-capacity,
  - emitted events are absolute-frame-based,
  - emitted events are deduplicated and monotonic,
  - finalized events never move once emitted.

Use existing code where appropriate:
- `src/analysis/transient.rs`
- `src/analysis/adaptive_snapshot.rs`

What to implement:
- `StreamAnalyzer` with bounded rolling audio history and absolute frame cursors.
- Absolute timeline state such as:
  - `received_until_frame`
  - `analyzed_until_frame`
  - `finalized_until_frame`
- Event types for:
  - transient events with strength and optional band/reset metadata
  - optional beat anchors or future control hints if easy to add cleanly
- A configurable lookahead/finalization rule:
  - events inside the unstable tail are not emitted yet
  - only events safely behind the lookahead boundary are finalized
- Strong dedupe logic so re-analysis of the recent horizon does not produce duplicate events.
- Unit tests focused on:
  - monotonic output
  - dedupe across overlapping analyzer passes
  - no event drift after finalization
  - absolute positions are stable across chunk boundaries
  - sub-hop callback overlap does not create duplicate events

Constraints:
- Keep this module internal for now.
- Do not integrate it into `StreamProcessor` yet.
- Keep allocations bounded after warmup where practical, but correctness and clean architecture matter more than total RT-safety in this prompt.

Acceptance criteria:
- Analyzer can ingest successive chunks and emit finalized events on an absolute timeline.
- The analyzer contract is clear enough for a router to consume next.
- Tests are added for the core invariants.

Validation:
- Run targeted tests for the new analyzer module.
- Also run a small existing streaming test subset to ensure no unrelated regressions.
- In your final message, explain the analyzer horizon/finalization strategy and list changed files.
```

## Prompt 3: Router

```text
You are working in the `timestretch-rs` repository.

Task: implement `src/stream/router.rs` as a monotonic timeline router that converts analyzer output into finalized render work.

Goal:
Turn absolute transient/control events into immutable routed operations such as:
- `TonalSpan { input_start, input_end }`
- `TransientEvent { center, start, end, strength, class, reset_mask }`

Design requirements:
- The router must be monotonic.
- Once a region/op is emitted as finalized, it must never be revised.
- The router must own the decision boundary between tonal background and transient events.
- Use enough lookahead to avoid premature splitting.
- Keep the model simple and robust before trying to make it fancy.

Use the current batch hybrid as guidance, not as an implementation model:
- `src/stretch/hybrid.rs`
- `src/analysis/adaptive_snapshot.rs`

What to implement:
- `HybridRouter`
- Routed op types with absolute input-frame coordinates
- Logic to:
  - convert analyzer transient events into event-centered transient regions
  - create tonal spans between transient regions
  - merge or suppress pathological micro-regions
  - classify transients coarsely if possible using existing heuristics
  - derive or attach per-band reset masks if cleanly supported
- Invariants:
  - no overlap between finalized tonal spans and transient regions unless explicitly modeled
  - no gaps in finalized routed coverage once a range is committed
  - routed output is ordered and immutable

Keep scope tight:
- Do not integrate with the renderers yet.
- Do not attempt residual/noise routing in this prompt.
- Do not attempt beat-grid-aware elastic timing unless it falls out naturally and cleanly.

Tests:
- Add focused router tests for:
  - simple tonal-only input
  - isolated transient events
  - multiple nearby transient events
  - boundary handling across analyzer chunk overlaps
  - no duplicate routed work
  - no uncovered gaps in finalized committed ranges

Acceptance criteria:
- The router can consume finalized analyzer events incrementally and emit a stable ordered stream of routed ops.
- The data model is good enough for a renderer to consume next.
- The router is clearly timeline-based, not chunk-based.

Validation:
- Run targeted tests for the router and any directly affected analyzer tests.
- In your final message, describe the routed op model and list changed files.
```

## Prompt 4: Tonal Renderer

```text
You are working in the `timestretch-rs` repository.

Task: implement the first version of the persistent tonal renderer in `src/stream/render.rs`.

Goal:
Create a continuous tonal rendering path that uses long-lived phase-vocoder state and never re-renders already-processed history.

Guidance:
- Reuse the existing `PhaseVocoder` implementation rather than inventing a new PV in this prompt.
- The new renderer should be a wrapper around persistent per-channel or per-plane `PhaseVocoder` state.
- The renderer must operate on finalized routed tonal spans expressed in absolute input-frame coordinates.
- The renderer must return output positioned on a stable output timeline, not just a chunk-local slice.

What to implement:
- `PersistentTonalRenderer`
- A render result type that includes:
  - absolute output start/end or equivalent placement metadata
  - produced samples
  - consumed input range
  - any holdback/tail metadata needed by the mixer
- Support for:
  - mono
  - stereo
  - existing stereo mode semantics, especially Mid/Side where relevant
  - ratio changes via existing `set_stretch_ratio` style mechanics
- Preserve the existing streaming PV continuity behavior as much as possible.
- Add careful comments describing how input absolute frame ranges map to output timeline positions.

Scope limits:
- Do not integrate transient overlays yet.
- Do not integrate into `StreamProcessor` yet.
- Do not replace the existing deterministic stream engine yet.

Tests:
- Add tests that prove:
  - tonal rendering is persistent across successive calls
  - no chunk-boundary re-render artifacts from internal API design
  - absolute output placement is monotonic
  - ratio changes do not reset the renderer or invalidate prior placement
  - flush behavior has a coherent contract

Acceptance criteria:
- There is a usable continuous tonal renderer with a clear API.
- It is obviously different from the old rolling hybrid re-render approach.
- The renderer is ready for a mixer and transient renderer to sit beside it.

Validation:
- Run targeted renderer tests and a small existing streaming subset if practical.
- In your final message, describe the output-placement contract and list changed files.
```

## Prompt 5: Transient Renderer And Mixer

```text
You are working in the `timestretch-rs` repository.

Task: implement the first version of the persistent transient renderer and the timeline mixer.

Goal:
Build event-local transient patches that can be placed over a continuous tonal bed on a stable output timeline.

Design choice:
- The tonal path is the continuous background.
- The transient path renders localized patches around routed transient events.
- The mixer stitches those patches onto the tonal timeline without ever rewriting already-emitted output.

Use existing code as ingredient sources:
- transient handling ideas from `src/stretch/hybrid.rs`
- WSOLA implementation from `src/stretch/wsola.rs`

What to implement:
- `PersistentTransientRenderer`
- `TimelineMixer`
- Data types such as:
  - `ScheduledPatch`
  - `MixWindow`
  - `PatchEnvelope`
- A first practical transient strategy:
  - attack-copy or attack-anchor behavior for transient onset
  - WSOLA-based local transient body/decay stretch where appropriate
  - output patches expressed in absolute output coordinates
- Mixer behavior:
  - accepts tonal slices and transient patches
  - applies scheduled crossfades/envelopes
  - maintains a holdback/commit horizon so it never emits samples that may still need future patch overlap
  - enforces "no retroactive writes before emitted position"

Keep this prompt practical:
- You do not need to build a perfect final-quality transient renderer yet.
- Focus on getting the architecture correct and seam-safe.
- Avoid batch-style re-rendering and delta crossfading.

Tests:
- Add targeted tests for:
  - patch scheduling on absolute timeline
  - holdback/commit behavior
  - no duplicate patch application
  - transient patch overlap with tonal bed
  - flush drains remaining holdback cleanly
  - simple attack preservation behavior on synthetic click/kick-like input

Acceptance criteria:
- There is now a coherent persistent render stack:
  - analyzer
  - router
  - tonal renderer
  - transient renderer
  - mixer
- The mixer owns the output timeline and commit horizon.
- No part of this design depends on re-running `HybridStretcher` over rolling history.

Validation:
- Run targeted tests for render/mixer logic.
- In your final message, explain the holdback/commit model and list changed files.
```

## Prompt 6: Integrate Into StreamProcessor

```text
You are working in the `timestretch-rs` repository.

Task: integrate the new persistent streaming hybrid engine into `StreamProcessor` behind a new opt-in engine mode.

Goal:
Wire the analyzer, router, renderers, and mixer into `StreamProcessor` as a new engine path while keeping the existing deterministic engine and legacy hybrid engine available during migration.

What to do:
- Add a new `StreamingEngine::PersistentHybrid` variant.
- Add the necessary state to `StreamProcessor` to own the new engine components.
- Integrate the new path into `process_into`, `flush_into`, and any relevant engine-selection APIs.
- Keep the default behavior conservative unless there is a compelling reason to change it in this prompt.
- Leave the legacy hybrid path intact for comparison and rollback.

Behavior requirements:
- Public API shape of `StreamProcessor` should remain stable.
- `set_stretch_ratio`, `set_tempo`, flush, and basic transport semantics must work in the new engine.
- Output must be emitted only from finalized/committable mixer state.
- No retroactive rewriting of emitted output.
- Preserve current error-handling style and fixed-capacity expectations where possible.

Important constraints:
- Do not remove the existing deterministic PV engine.
- Do not remove the existing legacy hybrid engine.
- Do not perform unrelated refactors.
- If some edge cases are still TODO, keep them explicit and narrow.

Tests:
- Add integration tests for:
  - basic `PersistentHybrid` output
  - ratio change mid-stream
  - tempo change mid-stream if applicable
  - mono and stereo
  - flush behavior
  - callback-size consistency on a small subset
- Update or add engine-selection tests as needed.

Acceptance criteria:
- `StreamProcessor` can run the new engine end to end.
- The new engine is opt-in and does not break existing callers.
- The codepath is clear and maintainable.

Validation:
- Run targeted streaming tests that exercise the new engine.
- In your final message, describe how engine selection now works and list changed files.
```

## Prompt 7: Hardening And QA

```text
You are working in the `timestretch-rs` repository.

Task: harden the new `PersistentHybrid` streaming engine for realtime DJ-style use and bring the test coverage up to the repo's standards.

Goal:
Move the new engine from "architecturally correct" to "credible realtime candidate".

Focus areas:
- bounded memory behavior after warmup
- callback-size invariance
- flush continuity
- rapid ratio/tempo automation continuity
- transient preservation under streaming
- no regressions to existing deterministic engine behavior

Use existing tests and harnesses as anchors:
- `tests/streaming.rs`
- `tests/streaming_edge_cases.rs`
- `tests/realtime_dj_conditions.rs`
- `tests/realtime_allocations.rs`
- `qa/streaming_quality.rs`

What to do:
- Make the new engine allocation behavior as bounded as practical after warmup.
- Add or update tests specific to `PersistentHybrid`.
- Ensure rapid ratio changes do not create obvious boundary discontinuities.
- Improve holdback, patch scheduling, and flush logic if the tests expose weak points.
- Keep the changes focused on the new engine and shared streaming helpers.

Do not do these things in this prompt:
- Do not make `PersistentHybrid` the default unless the results clearly justify it.
- Do not delete `LegacyHybridRerender`.
- Do not start a new PV algorithm rewrite here.

Acceptance criteria:
- The new engine passes a meaningful targeted test subset.
- Allocation behavior after warmup is understood and improved.
- Rapid automation and flush continuity are materially better than a naive patching implementation.
- The diff leaves the codebase in a state where a final cutover decision can be made based on evidence.

Validation:
- Run the relevant targeted tests.
- If feasible, run `qa/streaming_quality.rs` or a narrow useful subset.
- In your final message:
  - list changed files
  - summarize remaining risks
  - say whether `PersistentHybrid` is ready for wider use, still experimental, or not ready
```

## Prompt 8: Cutover Decision

```text
You are working in the `timestretch-rs` repository.

Task: evaluate whether the new `PersistentHybrid` engine should replace the legacy streaming hybrid path or become the preferred DJ-quality streaming engine.

Do not implement a broad rewrite immediately. First:
- inspect the current `PersistentHybrid` implementation
- compare it to the deterministic engine and legacy hybrid engine
- inspect the relevant tests and any available quality evidence
- identify the smallest safe cutover step

Then do one of these:
- if `PersistentHybrid` is clearly better and sufficiently stable, make it the preferred opt-in engine or default quality-mode engine
- if it is not ready, keep it experimental and tighten the interfaces, docs, and engine-selection story instead

Constraints:
- Do not delete fallback engines unless there is strong evidence and the diff remains low risk.
- Do not overscope.
- Keep product behavior explicit and reversible.

Acceptance criteria:
- The repo ends this prompt with a clear engine-selection story.
- The change is supported by tests and evidence, not optimism.
- The final message should explicitly state:
  - what changed
  - what engine is now preferred for live DJ use
  - what residual risks remain
```
