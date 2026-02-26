# Automated Time-Stretch Quality Optimization Loop

This system autonomously improves the audio quality of the `timestretch-rs` library by looping through evaluation, scoring, and AI-driven code modification.

## How it Works

1. **Evaluate**: Runs the library against a suite of test audio cases.
2. **Score**: Compares outputs against high-quality reference files using perceptual metrics.
3. **Optimize**: Feeds the scores and context into an AI agent (Claude Code or Codex).
4. **Refine**: The agent modifies the Rust DSP source and commits improvements.
5. **Iterate**: Loops until the target score (e.g., 92.0) is reached or max iterations hit.

## Prerequisites

- **Rust Toolchain**: `cargo` and `rustc`.
- **Python 3.10+**: with `librosa`, `numpy`, `soundfile`, `pandas`, and `matplotlib`.
- **uv**: for fast Python dependency management.
- **Reference Tools**: `rubberband` (recommended), `soundstretch`, or `sox`.
- **AI CLI**: `claude` (Claude Code) or `codex` (OpenAI CLI).

## Quick Start

```bash
cd optimize
make install-deps     # Install Python libraries
make refs             # Generate reference files (needs rubberband-cli)
make loop             # Start the optimization loop
```

## Configuration

All parameters are tunable in `optimize/config.toml`:
- `max_iterations`: How many times to loop.
- `target_score`: The goal average score (0-100).
- `weights`: Relative importance of different metrics (transients vs. frequency).
- `agent`: Choice of AI agent (claude_code, codex).

## Directory Structure

```text
optimize/
├── Makefile                     # Top-level commands
├── config.toml                  # All tunable parameters
├── references/                  # High-quality golden files
├── outputs/                     # Library outputs for current iteration
├── samples/                     # Source audio for testing
├── scripts/
│   ├── generate_references.sh   # Creates golden files
│   ├── run_test_suite.py        # Runs timestretch-rs
│   ├── score.py                 # Perceptual comparison engine
│   ├── optimize_loop.sh         # Orchestration logic
│   ├── agent_prompt.md.tmpl     # AI prompt template
│   └── report.py                # Final summary generator
├── agents/                      # AI CLI wrappers
└── logs/                        # Progress history, plots, and agent logs
```

## Metrics Explained

- **Spectral Convergence**: Magnitude distance in frequency domain.
- **Log Spectral Distance**: dB-scale spectral difference.
- **MFCC Distance**: Timbral similarity using Mel-frequency cepstral coefficients.
- **Transient Preservation**: Onset detection alignment and sharpness.
