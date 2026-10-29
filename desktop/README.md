Run the workspace desktop player:

```sh
cargo run --release -p timestretch-desktop -- path/to/track.wav
```

The desktop is the library's reference player. **Deck → Keylock** holds the
selected pitch (unchanged at 0 semitones). At unity pitch, tempo spans
**1/32–4×** (−96.875% to +300%), or 32× to 0.25× duration. The logarithmic
fader and typed BPM input use library limits, including the combined pitch/tempo
constraint. **Pitch** spans −24 to +24 semitones and adjusts the available tempo
range. **Tape** links pitch to tempo, so its range is 0.25–4×.

There is no pitch-changing fallback at slow tempos. Use **Pause/Stop** for zero
speed. Waveforms, beat grids, analysis caches, beat jumps, loops and scrub
gestures remain available; scrubbing is an application transport effect.

**Backend → Auto** is the default: Apple Accelerate on compiled macOS builds,
Portable elsewhere. You can pin Portable or Apple on macOS. The status shows
resolved backend, native processing rate, achieved tempo and pitch, and queue latency.
Backend changes prepare a new session on the worker and retain the consumed
position and paused state through the library's epoch/seek transaction.

Source and device rates are independent. Native sources use 44.1/48/96 kHz;
other supported sources convert through 48 kHz. The library performs conversion,
startup, tail accounting, source mapping, seeks and continuous loop wraps. The
worker chooses when to call it. Device callbacks never run or reset DSP, acquire
the UI mutex, allocate, log or clear the processor's buffers. Device errors are
reported to the UI through an atomic flag.

The application queue holds 32,768 frames. The current reserve target is 50 ms
(2,205 / 2,400 / 4,800 frames at 44.1 / 48 / 96 kHz), with work quota 8 and
1 ms requested worker sleeps. macOS uses an owned Mach audio time-constraint
scheduling guard. Preparation and paused operation use normal scheduling; UI
state contention cannot block rendering. Priority errors are reported in the
timing tooltip and fail qualification. The
worker delivers completed session audio before advancing more DSP work.
Reserve, pending session audio, algorithm lookahead and device latency are
reported separately. This is a candidate scheduling
policy, **not a universal realtime qualification**. The complete wall-clock
matrix has recorded starvation/timing failures under workstation load. Physical
callback budgets and the half-budget p99.9 target remain unchanged. See
[the qualification report](../benchmarks/QUALIFICATION.md).

The actual callback body is `delivery.rs`, shared with allocation audits. It
chunks large device buffers through prepared storage, transports cumulative
consumption receipts, and invalidates queued epochs in constant time. Host-held
queue/source guards prevent the audio thread owning their final large `Arc`.

```sh
cargo test --release -p timestretch-desktop
cargo run --release -p timestretch-qa -- callback-audit
python3 tools/qualify.py --stage playback
```

The executable is `target/release/timestretch-desktop`. Concealed synchronized
listening is generated separately from the app; the report records unscaled,
common-gain and loudness-matched audio without timing correction. No listening
preference is inferred from numerical agreement.
