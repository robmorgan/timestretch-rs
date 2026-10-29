# timestretch

Audio tempo and pitch processing in safe Rust, with an optional
Apple Accelerate numerical provider. **0.16 is a breaking redesign.**

There is one production processor and one host-driven execution adapter,
`StreamSession`. Offline rendering, the CLI and desktop playback use that same
adapter. The library owns source demand, startup, EOF, controls, seeking, loops,
output epochs and playback position. Applications own threads, wakeups, device
queues and presentation.

```rust
use timestretch::{render, Parameters, SessionConfig, TempoRate};

let input = vec![0.0f32; 48_000];
let audio = render(&input, SessionConfig {
    source_sample_rate: 48_000,
    output_sample_rate: 48_000,
    channels: 1,
    parameters: Parameters::keylock(TempoRate::from_bpm(128.0, 100.0)?),
    ..SessionConfig::default()
})?;
println!("achieved tempo: {}", audio.report.parameters.tempo());
# Ok::<(), timestretch::SessionError>(())
```

The controls use explicit units:

| Type | Meaning | Inclusive range |
|---|---|---|
| `TempoRate` | Source time / output time; 2 means twice as fast | 1/32–4 (−96.875% to +300% tempo) |
| `DurationRatio` | Output time / source time; inverse of tempo | 0.25–32 |
| `PitchRatio` | Frequency multiplier; 1 leaves pitch unchanged | 0.25–4 (−24 to +24 semitones) |

The combined factor **pitch / tempo must be 0.25–32**. For example, unchanged
pitch permits 32× duration, while +12 semitones permits at most 16× duration.
Unsupported combinations are rejected before processing. Each ratio exposes
`MIN` and `MAX`; `PitchRatio::tempo_bounds()` gives the valid tempo range at a
chosen pitch. Applications can impose narrower limits. Zero tempo is not a
finite stretch; use `StreamSession::set_transport` to pause or stop playback.

`Parameters::keylock(tempo)` requests unchanged pitch. For independent pitch
control, construct `Parameters` explicitly. `Parameters::linked(tempo)` links
pitch to tempo and returns `Result`, since linked rates must fit the pitch range:

```rust
use timestretch::{DurationRatio, Parameters, PitchRatio, TempoRate};

// Sixteen times longer, one octave higher.
let parameters = Parameters {
    tempo: DurationRatio::new(16.0)?.tempo(),
    pitch: PitchRatio::from_semitones(12.0)?,
    ..Parameters::default()
};
parameters.validate()?;

// Half speed with pitch following tempo: one octave lower.
let linked = Parameters::linked(TempoRate::new(0.5)?)?;
# Ok::<(), timestretch::SessionError>(())
```

Pass these parameters to `SessionConfig` for a new stream, or to
`StreamSession::schedule` for a change during playback. The default
`RatioConstraint::PreservePitch` keeps the selected pitch and allows duration
to follow hop quantization. `PreserveDuration` instead allows pitch to adjust.
`AppliedParameters` reports the actual quantized result, command ID and output
position where it took effect.

The processor supports mono/stereo at native 44.1/48/96 kHz. `AllowConversion`
accepts source/output rates from 32–192 kHz, using a 48 kHz core for non-native
sources and explicit bandlimited conversion. `NativeOnly` rejects conversion.
Source, native and output rates are reported separately. Multichannel and
formant preservation are outside this processor's scope.

`Backend::Auto` resolves to Accelerate on macOS when compiled in, otherwise
Portable. Explicit unavailable backends return an error. Pin `Portable` or
`Accelerate` for reproducible renders. Backend choice belongs to each prepared
instance; there is no thread-local provider selection or silent fallback.

| Feature | Capability |
|---|---|
| `accelerate` (default) | Isolated macOS FFT/vector bindings; has no platform dependency on other targets |
| `analysis` | Independent beat, key, loudness, transient and waveform analysis |
| `io` | Validated audio-buffer constructors and WAV I/O |
| `analysis` + `io` | Compatible `.tsa` analysis containers |
| `cli` | WAV rendering and analysis command |
| `qa` | Diagnostic access for repository qualification tools |

A minimal portable dependency uses `default-features = false`. Analysis is not
an input to timestretch and is not a dependency of the native DSP crate.
The desktop's optional `.tsa` files cache waveform and beat-grid analysis;
the library does not require them to process audio.
The library crates declare Rust 1.85; development and desktop builds use the
pinned toolchain in `rust-toolchain.toml`.

For streaming, prepare once, then satisfy `source_request`, call `advance` with
a work quota, collect stamped audio with `read_output`, and `acknowledge` its
actual consumption. Collection alone never advances playback position.
`NeedOutput` means release queued/collected output; `NeedAcknowledgement` means
drain control acknowledgements with `next_applied`. Source shortages and faults
are distinct. See [the streaming example](examples/realtime_stream.rs) and
[the session contract](docs/SESSION_CONTRACT.md).

Preparation allocates plans and bounded storage. Processing performs no
allocation or locking. A work quota counts stages, not microseconds: FFTs,
startup and reset still contain indivisible operations. The desktop runs these
on its own worker. Callback deadlines and the 50% headroom gate are unchanged;
[qualification](benchmarks/QUALIFICATION.md) records observed
failures and the scope of each result. `TimingInfo` separates startup, converter
lookahead, seek history, pending output and the control-boundary upper bound.
`owned_memory()` counts owned buffers separately from opaque FFT plans and
host-owned queues.

```sh
cargo run --release --features cli --bin timestretch-cli -- input.wav output.wav \
  --tempo 0.75 --backend auto --report output.json
cargo run --release --features cli --bin timestretch-cli -- analyze input.wav
cargo run --release -p timestretch-desktop -- input.wav
```

The [desktop reference player](desktop/README.md) exposes the same library
limits. **Keylock** holds the selected pitch independently of tempo; **Tape**
links pitch to tempo. The tempo fader does not add a pitch-changing brake at slow
rates. Use Pause/Stop for zero speed. The desktop reports achieved tempo and
pitch alongside its backend and timing information.

Rendering retains `round(source_frames * achieved_stretch * output_rate /
source_rate)` frames using the session's startup and EOF accounting. It does not
normalize, align, crop to a requested nominal duration or insert user-visible
padding. `--normalize` explicitly adds an export gain and records it.

Checked-in component and scheduler fixtures retain their numerical tolerances.
Self-contained quality, transport, allocation and generated-signal checks run
without private audio or external implementations. Human listening remains a
separate result.

The former Standard/Wide profiles, engine graph/stages, `StretchParams`,
standalone stretch/pitch implementations and tuning presets have been removed.
See [migration](docs/MIGRATION_0_16.md), [desktop usage](desktop/README.md) and
[the architecture](docs/CORE_ARCHITECTURE.md).

```sh
cargo test --release --workspace --all-features
cargo test -p timestretch-desktop --release
cargo clippy --workspace --all-features --all-targets -- -D warnings
python3 tools/qualify.py --output target/qualification-new
```

Normal tests and qualification use checked-in fixtures and generated signals.
The full wall-clock matrix takes approximately 45 minutes. Physical-device tests
are opt-in with `--device-seconds 15`; missing device evidence is reported as
unmeasured. See [QA usage](qa/README.md).
