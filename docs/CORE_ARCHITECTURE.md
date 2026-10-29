# Core architecture

`timestretch` exposes one synchronous, single-owner `StreamSession` adapter.
Offline rendering, the CLI and desktop drive the same source-demand, control,
output and consumption protocol. See [the full contract](SESSION_CONTRACT.md)
and [API migration](MIGRATION_0_16.md).

- `timestretch-core` owns native DSP, bounded buffers, resumable stages and
  per-instance numerical providers. Application code uses the facade.
- `timestretch` owns startup/EOF, rate/layout conversion, achieved controls,
  timeline/epoch accounting, pause/seek/loop semantics and prepared replacement.
- `timestretch-accelerate` isolates the unsafe macOS numerical boundary. The
  other library crates forbid unsafe code. Auto resolves to Accelerate on macOS
  when available, otherwise Portable; explicit unavailable choices are errors.
- `timestretch-analysis` owns independent beat/key/loudness/waveform analysis.
  Analysis artifacts and `.tsa` caches are not DSP inputs.
- The desktop owns thread scheduling, reserve, queues, device access and UI.
  Its callback delivers completed media and returns consumption receipts.
  Preparation, DSP, reset, allocation and heavy destruction occur off callback.

Inside `timestretch-core`, `HybridCore` coordinates spectral processing,
pitch-mark analysis, and grain synthesis. Its inner `SpectralCore` manages
phase-vocoder processing and resampling. `Direct` manages buffering and timing
around each core in the `Direct → HybridCore → Direct → SpectralCore` schedule.

Native rates are 44.1/48/96 kHz, mono/stereo. Explicit conversion handles
32–192 kHz through a 48 kHz core when necessary. Tempo spans 1/32–4, duration 0.25–32 and pitch
0.25–4, with the joint constraint 0.25 <= pitch/tempo <= 32. Achieved ratios, control
boundaries, converter delay and pending output are observable. Formant
preservation and coherent multichannel processing are not advertised.

Work quotas count resumable stages rather than microseconds. Host scheduling
must be measured separately from DSP throughput. The library owns no worker
runtime and requires no analysis file or external reference implementation.

[Qualification](../qa/README.md) uses checked-in fixtures, deterministic signals,
quality gates, lifecycle/determinism tests and allocator/performance harnesses.
