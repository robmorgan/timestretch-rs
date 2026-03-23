# Autoresearch Ideas Backlog

## Current Best
- **953.9 / 1000** on current branch head
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
- **Band-limited blend path**
  - Keep the PV as the full-band backbone, but blend only an attack-focused band from the cubic-resampled signal.
  - Candidate: high-passed or mid/high-only blended path so the resample contributes transient edge without destabilizing low-frequency PV coherence.
  - Rationale: current full-band blend helps percussive 2.0x but is limited by aliasing / tonal contamination.

- **Flux-adaptive blend EQ on the blended path only**
  - On strong transient frames, shape only the blended signal (not the PV output) with a gentle transient-focused tilt.
  - Candidate: slightly attenuate sub-bass and/or emphasize attack band before mixing.
  - Rationale: preserve the resample path’s transient edge while reducing the low-frequency artifacts that cap further blend increases.

- **Transient residual blend instead of raw resample blend**
  - Derive a lightweight residual from the input (e.g. input minus a cheap low-passed or smoothed version) and blend that residual into the PV output.
  - Rationale: batch/reference gap is mostly attack sharpness, not sustained tonal content.

- **Per-case diagnostic pass for percussive 2.0x**
  - Inspect whether remaining loss is mostly `40 Hz`, `200 Hz`, or batch similarity mismatch.
  - Rationale: the dominant remaining weakness is still percussive `2.0x`; targeted diagnostics may reveal a cleaner downstream-only lever.

## Guardrails
- Avoid PV-internal magnitude edits unless they also rethink/replace the downstream energy compensation.
- Avoid per-bin phase-locking overrides; they disrupt IF accumulation.
- Prefer downstream/post-gain shaping over core PV changes.
- Do not overfit to benchmark quirks; keep changes plausible for real audio, not just synthetic cases.
