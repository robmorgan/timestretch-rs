# Qualification

Builds, fixture tests and generated-signal qualification need only this checkout,
Rust and Python 3.11+. The desktop/QA audio dependencies require the platform SDK
(on Linux, install ALSA development headers). The workflow uses checked-in
fixtures and generates its audio signals locally.

```sh
cargo test --release --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo run --release -p timestretch-qa -- quality --strict
python3 -B tools/qualify.py --output target/qualification-new --strict
```

The full host matrix takes about 45 minutes. `--quick` exercises each stage with
fewer conditions; it cannot pass full protocol gates. Device tests are opt-in:
add `--device-seconds 15` for muted playback on the default device. Without that
option the report says device testing was not measured. With it, `--strict`
also requires the physical-device gates. Human listening is a separate review.

Stages are `fixtures`, `renders`, `micro`, `quality`, `core`, `session`, `audit`,
`playback`, `boundary`, `device`, `listening` and `report`. Existing directories
pin their executable and source revision. Use a fresh directory after changes.
Reports retain raw JSON, scalar CSV, source/binary hashes and an HTML overview.
No missing measurements are counted as passes.

Checked-in native fixtures preserve independent component/scheduler expectations
and numerical tolerances. Quality checks use one fixed portable evaluator.
Generated signals cover impulses, tones, multitones, noise, silence and stereo
(including antiphase), at native rates, mono/stereo and three duration ratios.
The same-configuration `render_corpus.py --compare PREVIOUS_RENDER_DIRECTORY`
option detects output regressions; it is not an independent quality oracle.

```sh
# Optional user-supplied float WAVs; generated signals are also included.
python3 -B tools/listen.py --output target/audition track.wav
python3 -m http.server 8000 --directory target/audition
```

The listener conceals current provider labels and synchronizes playback. Raw,
common-gain and loudness-matched copies have recorded gains. Its descriptive
scores use Portable output as the comparison anchor, not an ideal audio target.
Neither alignment correction nor listening gain affects scored files.

The public-corpus BPM and Rubber Band integration checks remain optional local
checks and dedicated CI jobs. `scripts/fetch_public_corpus.sh` obtains the public
inputs; `rubberband_reference_gate` uses the publicly available Rubber Band CLI
with its existing bounds. They are separate from self-contained qualification.
