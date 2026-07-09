# Changelog

## Unreleased

### Breaking changes

- Removed the legacy hybrid streaming engine from `StreamProcessor`:
  - `StreamingEngine::LegacyHybridRerender` variant removed
    (`StreamingEngine::Deterministic` remains and is still the default).
  - `StreamProcessor::set_hybrid_mode` removed.

  The deterministic engine is the only streaming path. The offline hybrid
  stretcher used by `stretch_buffer` and the batch APIs is unaffected.

## 0.5.0

See the [release notes](https://github.com/robmorgan/timestretch-rs/releases)
for changes in 0.5.0 and earlier.
