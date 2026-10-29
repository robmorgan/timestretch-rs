0.16 replaces the engine and parameter surface. There is no compatibility DSP
path, profile selector or legacy feature flag.

| Before | Now |
|---|---|
| `stretch(input, StretchParams)` | `render(input, SessionConfig)` returns samples and a report |
| Duration `ratio` | `DurationRatio::new(ratio)?.tempo()`; native controls use `TempoRate` |
| `pitch_shift` | `Parameters { pitch: PitchRatio, .. }` through the same renderer/session |
| `Engine::build`, controller, source producer, processor | `PreparedSession` and single-owner `StreamSession`; host owns cross-thread queues |
| `EngineProfile`, Standard, Wide, opt-in native processor | timestretch is always the sole production algorithm |
| Tape profile / keylock bypass | `Parameters::linked` / `Parameters::keylock` |
| Profile-specific latency and reset hints | Session timing, source requests, epochs and consumption receipts |
| Old algorithm, `apple-qa` and `qa-harnesses` feature gates | Production `accelerate`; repository diagnostics use the QA package |
| Formant/envelope/window/FFT presets | Removed; no formant-preservation claim |
| Pre-analysis passed into DSP | Optional independent `analysis`; artifacts remain usable by applications |
| Implicit output normalization | Explicit CLI `--normalize`; gain recorded in the export report |
| Unchecked audio-buffer constructors | `AudioBuffer::new`, `from_mono`, `from_stereo` return `Result` under `io` |

Auto selects the measured Apple provider when available; `--no-default-features`
keeps the library portable. Pin a provider for deterministic exports. Changing a
provider while playing requires preparing a replacement and the session's epoch
transaction. A callback must never prepare, drop or reset a processor.

Desktop `Deck → Tape/Keylock` is pitch intent. The profile menu and presets are
gone. A render worker drives the same session as offline rendering. The callback
only delivers completed audio and the scrub voice. Scrub consumption receipts
drive library position, and release uses a library seek. The tempo fader uses
the full library range without a brake fallback.

Tempo now spans 1/32–4, duration 0.25–32, and pitch remains 0.25–4. Combined
pitch/tempo must be 0.25–32. `Parameters::linked(tempo)` now returns `Result`:
use `Parameters::linked(tempo)?`, since the wider tempo domain includes rates
outside the supported pitch domain. `MIN`, `MAX` and `PitchRatio::tempo_bounds()`
let hosts derive controls from library capabilities.

Native fixture data is unchanged. Sine-generated auxiliary windows allow one ULP
of platform rounding outside macOS; zero-filled regions remain exact. The portable
44.1 kHz stereo hold/toggle case allows 0.012 peak / 0.0015 RMS audio drift because
FFT rounding can select a different phase reference between nearly equal peaks.
The synthesis branch fixture also allows 0.000001 absolute error in the processed
spectrum on Windows, accounting for scalar sine/cosine rounding. Its phase,
magnitude, and state checks remain exact. All other numerical limits, including
the Accelerate scheduler limits, remain unchanged, and scheduler and timing
checks remain exact. Retained product quality bounds are exercised by the current
QA host without private inputs. The former mid/side implementation's exact L=R
bit-identity assertion was implementation specific; the existing 60 dB center/side
rejection product bound remains active. Historical
sources and measurements are preserved separately and are not build dependencies.
