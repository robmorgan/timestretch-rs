# Architecture Review and Target Design

Date: 2026-02-28

## 1. Current Codebase Audit

## 1.1 Library Layer (`src`)

| File | Current Algorithm / Responsibility | Buffer Management | Limitations / Gaps |
|---|---|---|---|
| `src/lib.rs` | Public API (`stretch`, `pitch_shift`, BPM, buffer wrappers), channel de/interleave, normalization | Batch alloc per call plus `*_into` variants | Some APIs still allocate by design (`stretch`, `pitch_shift`) |
| `src/error.rs` | Error model | N/A | None significant |
| `src/cli.rs` | CLI wrapper around library | File-sized batch | Not callback/RT target |
| `src/io/mod.rs` | I/O module glue | N/A | Minimal |
| `src/io/wav.rs` | WAV decode/encode helpers | Full-file load/write | Not streaming decode path |
| `src/core/mod.rs` | Core module glue | N/A | None |
| `src/core/fft.rs` | FFT constants/helpers | N/A | Thin wrapper |
| `src/core/window.rs` | Window generation (Hann/BH/Kaiser) | Allocates per window creation | Good coverage; no precomputed static tables |
| `src/core/ring_buffer.rs` | Fixed-capacity ring buffer used by stream path | Bounded push/pop/peek/discard | Requires up-front capacity tuning |
| `src/core/resample.rs` | Linear/cubic/sinc resamplers | Mostly caller-buffer driven | Some high-quality paths are CPU-expensive |
| `src/core/types.rs` | Params, presets, enums, buffer types | N/A | Large config surface can drift without strict profiling gates |
| `src/core/crossover.rs` | Subband split/recombine utilities | Frame-wise spectral split | Added FFT cost for split modes |
| `src/core/preanalysis.rs` | Artifact serialization and metadata | N/A | Offline dependency for best beat-aligned behavior |
| `src/analysis/mod.rs` | Analysis module glue | N/A | None |
| `src/analysis/transient.rs` | Multi-feature onset detection (flux + energy + phase deviation), adaptive thresholding, conservative lookahead confirmation | Per-call FFT frame buffers | Lookahead policy is currently fixed in code (not externally configurable) |
| `src/analysis/beat.rs` | Beat detection and subdivision snapping | Frame-wise onset/interval analysis | Confidence can degrade on sparse/non-EDM material |
| `src/analysis/hpss.rs` | HPSS decomposition support | Spectrogram-domain allocation | Offline/heavier than pure RT path |
| `src/analysis/frequency.rs` | Frequency/bin utilities | N/A | None |
| `src/analysis/preanalysis.rs` | Offline DJ pre-analysis pipeline | Artifact buffers | Requires offline run for best robustness |
| `src/analysis/comparison.rs` | Objective quality metrics and reports | Batch frame analysis | Not in audio callback path |
| `src/stretch/mod.rs` | Stretch module glue | N/A | None |
| `src/stretch/params.rs` | Param validation and limits | N/A | None |
| `src/stretch/wsola.rs` | WSOLA/transient-style time-domain path | Segment buffers + search windows | Search cost grows with range; can stutter on bad matches |
| `src/stretch/phase_vocoder.rs` | STFT PV core with phase propagation and locking support | Stateful frame buffers; streaming methods present | Transient-heavy material still depends on hybrid routing quality |
| `src/stretch/phase_locking.rs` | Identity/ROI locking and peak/trough logic | Per-frame peak/trough vectors | No selective lock mode toggle based on local confidence yet |
| `src/stretch/envelope.rs` | Spectral envelope/formant preservation helpers | Frame-domain buffers | Can add compute overhead for vocal modes |
| `src/stretch/multi_resolution.rs` | Multi-resolution PV strategy | Multiple FFT paths | Higher CPU footprint |
| `src/stretch/stereo.rs` | Stereo modes (incl. mid/side) | Channel-coupled processing | Added complexity in maintaining strict mono compatibility |
| `src/stretch/hybrid.rs` | Hybrid WSOLA + PV + beat/onset segmentation + recombination + timeline correction | Segment-wise staging and overlap compensation | Most complex path; parameter sensitivity at extreme ratios |
| `src/stream/mod.rs` | Stream module glue | N/A | None |
| `src/stream/processor.rs` | Real-time stream processor with fixed-capacity rings, ratio smoothing, optional persistent hybrid mode | Bounded ring buffers, preallocated scratch, `process_into` zero-growth intent | Hard real-time depends on capacity config and caller avoiding reallocation of output vec |

## 2. Gap Analysis Against Research Target

## 2.1 Missing or Weak Areas
- Transient detector includes a conservative lookahead-confirmation stage, but policy is fixed in code and not externally configurable.
- Selective phase locking policy is present (identity/ROI), but not yet fully content-adaptive (e.g., per-band lock-strength modulation from confidence/SNR classes and stretch-ratio).
- Hybrid route is strong, but explicit three-way decomposition (harmonic/percussive/noise) is not a mandatory first-class pathway in the default RT path.
- Real-time guarantees are strong in stream mode, but not formalized with hard budget assertions in CI for callback-size worst-case.
- Benchmark harness existed in fragments; required unified before/after matrix with ratios and exported spectrogram CSV did not exist.

## 2.2 Already Strong
- Practical hybrid architecture (WSOLA + PV + beat/onset logic).
- Identity/ROI phase-lock implementation.
- Stereo/mid-side coherence support.
- Pre-analysis artifact pipeline for DJ workflows.
- Broad test coverage and many edge-case checks.

---

## 3. Target Architecture Specification

```text
Input RingBuffer (interleaved)
  -> Transient Detector (flux + energy + phase-deviation + lookahead confirm)
  -> Segment Router
      -> Transient Path (WSOLA / short time-domain copy-stretch)
      -> Steady Path (Multi-Resolution Phase Vocoder + Adaptive Phase Locking)
      -> Optional Noise/Residual Path (relaxed lock + diffusion control)
  -> Boundary Crossfade + Overlap Compensation
  -> Output RingBuffer
```

## 3.1 Module Decomposition and Interfaces

### Public API Boundary
- `StretchProcessor::new(config) -> Self`
- `push_input(interleaved: &[f32]) -> Result<(), StretchError>`
- `pull_output(out: &mut [f32]) -> usize`
- `set_ratio(r: f64)`
- `set_pitch(semitones: f32)`
- `flush(out: &mut Vec<f32>)`

### Internal Traits
- `TransientDetector`:
  - `analyze(frame) -> TransientEvents`
- `SteadyStateStretcher`:
  - `process(segment, ratio) -> Vec<f32>`
- `TransientStretcher`:
  - `process(segment, ratio) -> Vec<f32>`
- `Recombiner`:
  - `append_segment(output, segment, boundary_state)`

## 3.2 Signal-Flow Parameters (Default Balanced @ 44.1 kHz)
- Analysis FFT sizes: `1024`, `2048`, `4096` (multi-resolution path)
- Analysis hop: `fft/4`
- Windows:
  - Analysis: Blackman-Harris (or Hann in low-latency mode)
  - Synthesis: Hann for stable OLA normalization
- WSOLA:
  - segment size: `20-40 ms` content-dependent
  - search range: `8-20 ms`
  - crossfade: raised cosine
- Transient detector:
  - flux bands: `<100 Hz`, `100-500 Hz`, `500-4 kHz`, `>4 kHz`
  - adaptive threshold: median/robust stat + sensitivity scaling
  - lookahead confirm: `1-3` analysis hops

## 3.3 Transient Detector Design
- Features:
  - positive spectral flux (weighted high band emphasis),
  - onset energy slope,
  - phase-deviation magnitude.
- Adaptive thresholding:
  - robust local baseline + floor,
  - minimum onset spacing.
- Lookahead:
  - provisional trigger at frame `t`,
  - confirm if `flux(t+1..t+k)` retains spike signature,
  - otherwise downgrade to non-transient.
- Output:
  - onset index,
  - fractional refinement,
  - strength score.

## 3.4 Phase Vocoder Design
- Use instantaneous-frequency propagation with unwrapped phase.
- Peak/trough-driven identity/ROI locking.
- Adaptive lock policy:
  - tighter below ~200 Hz,
  - moderate in low-mid,
  - relaxed in upper bands at large ratios.
- Optional envelope preservation stage for vocal/formant-sensitive pitch shifts.

## 3.5 Time-Domain Path Design (WSOLA)
- Correlation search around expected anchor.
- Search window scales with transient class and ratio.
- Keep transient attack copy window short (`~5-10 ms`) before decay-stretch region.
- Raised-cosine overlap for joins.

## 3.6 Recombination Strategy
- Segment boundary crossfade length computed from local period estimate and transient strength.
- Maintain explicit overlap bookkeeping to guarantee exact final output length.
- Prefer transient-path dominance near event center; blend toward steady path in tails.

## 3.7 Real-Time Guarantees
- No heap growth in callback path after initialization:
  - all rings/scratch vectors preallocated.
- No locks in callback path.
- Bounded loops by configured max callback frames.
- Capacity formula:
  - `capacity_frames >= lookahead + max_callback + fft_size`.
- Worst-case latency target:
  - low-latency mode: `<10 ms` practical path at 44.1 kHz for DJ control surfaces,
  - balanced/high-quality modes trade latency for quality.

## 3.8 SIMD / Hot Loop Targets
- Window multiply + overlap-add kernels.
- Magnitude/phase extraction loops.
- Cross-correlation in WSOLA search.
- Optional architecture-specific acceleration:
  - x86 AVX2,
  - ARM NEON,
  - scalar fallback always maintained.

## 3.9 Pitch-Shifting Extension
- Primary path:
  1. time-stretch by `1/pitch_factor`,
  2. high-quality resample back to original duration.
- Formant-preserving mode:
  - estimate spectral envelope,
  - apply envelope remap during resynthesis.
- Transient policy:
  - preserve event timing and attack shape across pitch shifts.

## 3.10 Presets

| Preset | FFT / Hop | Detector Sensitivity | WSOLA Search | Phase Locking | Intended Content |
|---|---|---|---|---|---|
| `DjBeatmatch` | 2048 / 512 | Low-Med | Short | Strong low-band ROI | House/techno live sync |
| `EDM Percussive` | 2048 / 512 | High | Medium | Medium | Drum-forward EDM |
| `Vocal Formant` | 4096 / 1024 | Med | Short | ROI + envelope preserve | Vocals and hooks |
| `Speech LowLatency` | 1024 / 256 | Med | Medium | Light locking | MC/talkover |
| `Ambient MaxQuality` | 4096 / 1024 | Low | Long | Adaptive relaxed highs | Pads, transitions |

---

## 4. Implementation Status in This Overhaul
- Applied research, architecture, and implementation-plan artifacts directly in this repository.
- Added dedicated benchmark harness: `tests/quality_benchmark.rs`.
- Harness generates:
  - multi-ratio objective report CSV,
  - per-case WAV outputs (via `hound`),
  - per-case spectrogram CSV for external plotting.
- Validated existing hybrid streaming persistent-buffer behavior with full
  `tests/hybrid_streaming.rs` pass (chunk-size parity and batch-length checks).
- Added conservative transient lookahead confirmation in adaptive thresholding to
  reduce isolated one-frame false-positive onsets while preserving strong attacks.
- Exposed transient lookahead confirmation aggressiveness via `StretchParams`
  (frames/threshold-relax/peak-retain/strong-spike-bypass controls).
- Implemented confidence-driven adaptive phase-lock mode selection in the
  phase vocoder (ROI on noisy/low-confidence frames, identity on harmonic
  near-unity frames, with `StretchParams::adaptive_phase_locking` control).
