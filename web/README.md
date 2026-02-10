# timestretch-rs Web Demo

Real-time audio time stretching in the browser via WebAssembly.

## Prerequisites

- Rust toolchain with the `wasm32-unknown-unknown` target:
  ```
  rustup target add wasm32-unknown-unknown
  ```
- wasm-pack:
  ```
  cargo install wasm-pack
  ```

## Build

```bash
cd web
wasm-pack build --target web --out-dir pkg
```

## Run

Serve the `web/` directory with any static file server:

```bash
cd web
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser.

## Usage

1. Drag & drop an audio file (WAV, MP3, etc.) onto the drop zone
2. The file will be decoded and BPM will be detected automatically
3. Click **Play** to start playback
4. Adjust the **Stretch Ratio** slider to slow down or speed up without changing pitch
5. Adjust the **Pitch Shift** slider to change pitch without affecting the stretch
6. Select an **EDM Preset** for optimized processing parameters (requires restart)
