# Autoresearch Ideas Backlog

## Current Best
- **954.0 / 1000** on current branch head
- Best kept path so far: **per-bin PV flux tracking used only to drive post-gain flux-adaptive resample blending**
- Important constraint: **PV-internal per-bin magnitude/phase modifications have consistently hurt quality**

## Pruned / Closed Paths
- Per-bin identity locking overrides
- Per-bin magnitude boosts
- Conservative / local-median / broadband-gated per-bin transient detectors inside PV
- Frame-level phase-gradient attenuation inside PV
- Lowering resample-blend threshold below ratio-distance `> 0.5`
- Uniform WSOLA blend in streaming
- Anti-aliased LP on blended resample path

These are stale unless a fundamentally new mechanism appears. Do not retry minor threshold variants.

## Promising Next Ideas
- **Stateful streaming hybrid overlay**
  - Keep the deterministic PV backbone, but add a tiny stateful WSOLA-style transient overlay path only for detected onsets.
  - Key requirement: maintain input/output correspondence across callbacks so the overlay is time-aligned, unlike the failed naive WSOLA blend.
  - This is the most promising broader redesign if local helper tweaks are exhausted.

- **Streaming transient attack copy / decay handoff**
  - Borrow the batch hybrid idea more directly: copy a very short attack anchor from input/resample path, then hand off to PV for decay.
  - Needs explicit stream-state bookkeeping for attack windows and overlap boundaries.

- **Unified onset model for stream mode**
  - Combine PV per-bin flux with low-frequency onset-energy tracking, since remaining weakness is percussive 2.0x and likely kick-dominated.
  - Could drive either helper blending or a future hybrid overlay more reliably than spectral flux alone.

- **Band-limited blend path**
  - Keep the PV as the full-band backbone, but blend only an attack-focused band from the cubic-resampled signal.
  - Pure high-pass-only blend was tried and regressed (`953.6`), so any revisit should retain some low/full-band component.
  - Candidate: mixed full-band + band-limited helper rather than HP-only.
  - Rationale: current full-band blend helps percussive 2.0x but is limited by aliasing / tonal contamination.

- **Flux-adaptive blend EQ on the blended path only**
  - On strong transient frames, shape only the blended signal (not the PV output) with a gentle transient-focused tilt.
  - Candidate: slightly attenuate sub-bass and/or emphasize attack band before mixing.
  - Rationale: preserve the resample path’s transient edge while reducing the low-frequency artifacts that cap further blend increases.

- **Transient residual blend instead of raw resample blend**
  - Tried as a lightweight one-pole residual helper and tied the current best.
  - If revisited, it should be as part of a broader diagnostic/architectural change, not another local helper variant.

- **Per-case diagnostic pass for percussive 2.0x**
  - Inspect whether remaining loss is mostly `40 Hz`, `200 Hz`, or batch similarity mismatch.
  - Rationale: the dominant remaining weakness is still percussive `2.0x`; targeted diagnostics may reveal a cleaner downstream-only lever.

## Guardrails
- Avoid PV-internal magnitude edits unless they also rethink/replace the downstream energy compensation.
- Avoid per-bin phase-locking overrides; they disrupt IF accumulation.
- Prefer downstream/post-gain shaping over core PV changes.
- Scalar retunes of the current adaptive full-band blend (base/floor/cap/helper mix) are effectively exhausted; do not keep sweeping them blindly.
- Do not overfit to benchmark quirks; keep changes plausible for real audio, not just synthetic cases.
