# Stretch Quality Parity Notes

This suite adds strict, signal-driven regression gates modeled after established C++ DSP test patterns, implemented natively for this crate.

- Test file: `tests/stretch_quality_regressions.rs`
- Helper module: `tests/common/mod.rs`

## Scope Covered

- Offline identity behavior on pure tones.
- Ratio correctness and pitch stability under time-stretching.
- Streaming robustness across callback sizes.
- Large-block streaming stress behavior.
- Transient survival for impulse/click-rich signals.
- Small ratio sweeps and mixed-signal spectral checks.

## New Tests

- `test_sinusoid_unchanged_offline_ratio_1_strict`
- `test_sinusoid_2x_offline_preserves_pitch_and_shape`
- `test_streaming_chunk_sweep_zero_crossings_and_safety`
- `test_streaming_chunk_sweep_amplitude_mapping`
- `test_streaming_large_block_robustness_80k`
- `test_ratio_sweep_sine_length_and_pitch`
- `test_ratio_sweep_two_tone_peak_bins`
- `test_ratio_sweep_impulse_train_transient_count_and_sharpness`
- `test_ratio_sweep_click_pad_transient_survival`
- `test_realtime_pitch_scale_sweep_requires_new_hook` (`#[ignore]`)

## Failure Signal Meaning

- Identity/tone drift: phase tracking, windowing, or overlap-add instability.
- Ratio/pitch drift: hop/ratio mapping and phase progression mismatch.
- Chunk-size sensitivity: streaming state/latency compensation dependence on callback size.
- Large-block failures: buffering/length accounting edge conditions.
- Transient losses: onset handling, WSOLA boundary decisions, or crossfade behavior.

