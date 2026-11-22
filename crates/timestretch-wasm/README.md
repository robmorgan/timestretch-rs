# timestretch-wasm

WebAssembly bindings for [`timestretch`](../../README.md), built with
`wasm-bindgen`. The package exports the same host-driven session the native
library uses: `render` for one-shot processing and `Session` for streaming.
TypeScript declarations are generated with the package.

The library crates compile for `wasm32-unknown-unknown` unchanged; this crate
only adds the JavaScript boundary. It is not published to crates.io or npm.

## Build

Requires the `wasm32-unknown-unknown` target (installed by
`rust-toolchain.toml`) and [`wasm-pack`](https://github.com/drager/wasm-pack).
From the repository root:

```sh
wasm-pack build crates/timestretch-wasm --target web       # browsers, ES module, -> pkg/
wasm-pack build crates/timestretch-wasm --target bundler   # webpack/vite/etc.
wasm-pack build crates/timestretch-wasm --target nodejs --out-dir pkg-node
node crates/timestretch-wasm/tests/smoke.mjs               # needs the nodejs build
```

To use the Rust library directly from your own wasm crate instead, depend on
`timestretch` with `default-features = false`.

## Offline rendering

```js
import init, { SessionOptions, render } from "./pkg/timestretch_wasm.js";
await init();

const options = new SessionOptions(); // 44.1 kHz stereo, unity tempo and pitch
options.sourceSampleRate = buffer.sampleRate;
options.outputSampleRate = buffer.sampleRate;
options.channels = 2;
options.tempo = 1.08;                 // source time / output time, 1/32–4
options.pitch = 2 ** (-2 / 12);       // frequency ratio, 0.25–4
const output = render(interleaved, options); // Float32Array in, Float32Array out
```

Samples are interleaved, finite `f32`. Mono and stereo are supported, at
32–192 kHz. `render` blocks its thread, so call it from a Web Worker for
anything longer than a short clip.

[`example/`](example/) is a complete page: decode a file with
`decodeAudioData`, render, and play. Run it with:

```sh
wasm-pack build crates/timestretch-wasm --target web
python3 -m http.server -d crates/timestretch-wasm 8000
# open http://localhost:8000/example/
```

## Streaming

`Session` follows the [session contract](../../docs/SESSION_CONTRACT.md):

```js
const session = new Session(options);
const block = new Float32Array(1024 * channels);
while (!session.finished) {
  if (session.advance(32) === ProcessStatus.NeedSource) {
    const request = session.sourceRequest(); // { epoch, startFrame, frames }
    session.supply(request, source.subarray(/* request span, interleaved */));
    // or session.finishInput() at the end of a source of unknown length
  }
  while (session.nextApplied()) {}           // drain control acknowledgements
  const stamp = session.readOutput(block);   // writes stamp.frames frames
  // ...queue block[0 .. stamp.frames * channels] with its stamp...
  session.acknowledge(stamp.epoch, stamp.startFrame + framesActuallyPlayed);
}
```

Also available: `schedule(id, outputFrame, tempo, pitch)`, `seek`, `setLoop` /
`clearLoop` (both need `options.seekable = true`), `setTransport`, and the
`position`, `epoch`, `outputReady`, `commitHorizon`, `controlQuantumFrames`,
`renderedFinished`, `finished` and `fault` getters. Errors are thrown as
`Error` with the library's error name as the message (`StaleSource`,
`InvalidParameter`, …). [`tests/smoke.mjs`](tests/smoke.mjs) exercises the
whole protocol.

Frame counts and epochs are JS numbers, exact up to 2^53. Objects returned by
the bindings (`SourceRequest`, `OutputStamp`, `AppliedParameters`) are small
wasm-backed classes: copy their fields into plain objects before sending them
through `postMessage`.

## Real-time playback with AudioWorklet

The session never touches a device or a thread, so the browser host supplies
both. Use the same split as the desktop player:

- **Web Worker** owns the `Session` and the decoded source. It keeps a
  reserve of rendered audio ahead of the device (the desktop uses 50 ms) by
  looping `advance` / `supply` / `readOutput`, and posts each block with its
  `{ epoch, startFrame, frames }` stamp.
- **AudioWorkletProcessor** only copies queued blocks into `process()` output
  and posts back cumulative receipts `{ epoch, endFrame }`. The worker passes
  them to `acknowledge`, which is what moves `position`.
- **On seek or loop change** the worker gets a new epoch from `seek` /
  `setLoop` and tells the worklet to drop everything queued under older
  epochs. Late receipts from an old epoch are ignored by the session.

Do not run the session inside `process()`. Seek and startup priming are
bounded but indivisible, and can take longer than one 128-frame render
quantum. Wake the worker from receipts or a timer; it has no access to the
audio clock.

Either transport works between the two:

- `postMessage` with transferred `Float32Array` blocks needs no special
  headers and works on any static host.
- A `SharedArrayBuffer` ring buffer avoids per-block messages but requires
  cross-origin isolation (`Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`).

Report control latency as the session's pending output plus your queue
reserve plus the device's own `outputLatency`.

The streaming protocol across a queue is covered by the Node smoke test. The
repository's real-time browser player is the [deck app](../../desktop/README.md)
compiled to WebAssembly. It is a Rust application using `timestretch`
directly, not these bindings, and it takes the simpler route of running the
session and audio scheduling on the page's main thread.

## Limitations

- Single-threaded: one session per worker. Nothing here needs wasm threads.
- Portable backend only. `Backend::Auto` resolves to it; Apple Accelerate is
  macOS-native.
- No explicit wasm SIMD. rustfft's `wasm_simd` feature is not enabled.
- wasm32 addresses 4 GB. A decoded stereo track uses 8 bytes per frame
  (about 42 MB per minute at 44.1 kHz), and `render` holds input and output.
- In your own wasm crate, the path-based helpers behind the `io` feature
  (`read_wav_file`, `.tsa` files) compile but have no filesystem in a browser, and
  the `qa` feature uses clocks that are unavailable on this target.
