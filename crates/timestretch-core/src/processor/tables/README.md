# Window tables

These six files contain little-endian IEEE 754 `f32` values with no header.
`tables.rs` embeds them at compile time and decodes them during construction, so
ordinary builds on every platform use the same values.

| Files | Entries per file | Purpose |
| --- | ---: | --- |
| `cosine-{512,1024,2048,4096,8192}.bin` | The size in the filename | Cosine seeds for phase-vocoder analysis and synthesis windows |
| `flank.bin` | 16,384 | Rising sine-cubed gains for grain overlap; falling gains use `1 - value` |

## Check or regenerate

The Rust example in `timestretch-accelerate` computes the tables using the existing
Accelerate sine/cosine adapter. It needs macOS and Apple Accelerate. From the
workspace root:

```sh
# Compare every byte without modifying files. This is also the default mode.
cargo run -p timestretch-accelerate --example generate_tables -- --check

# Generate into a separate directory for review.
cargo run -p timestretch-accelerate --example generate_tables -- --write --output target/generated-tables
cargo run -p timestretch-accelerate --example generate_tables -- --check --output target/generated-tables

# Regenerate the checked-in tables explicitly.
cargo run -p timestretch-accelerate --example generate_tables -- --write
```

The default directory is resolved from the crate's location, independent of the
working directory. An explicit `--output` path is relative to the working
directory. `--check` exits unsuccessfully on missing files, differing lengths, or
any differing float bits. For equal-length files it reports the first differing
float index and both bit patterns. `--write` creates the directory if needed and
replaces only the six table files.

## Arithmetic and reproducibility

For a cosine table of length `N`, compute `step = TAU / (N - 1)` in `f32`, then
`angle[i] = (i as f32) * step`. Take the cosine output of the vector sine/cosine
adapter. For the flank table use `step = FRAC_PI_2 / 16383`, take the sine output,
then cube each value as `(s * s) * s` with two `f32` multiplications.

The constants are Rust's `std::f32::consts::{TAU, FRAC_PI_2}`. Rounding the step
before multiplying by the index is intentional. Using `f64`, moving the division,
using scalar trigonometry, or using `powf` can change the resulting bits.

The generator reproduces all 32,256 checked-in floats exactly on ARM64 macOS
27.0.1. Accelerate's rounding may differ across OS versions or architectures;
`--check` detects those differences without rewriting the established tables.
Keep the binary tables checked in so builds do not depend on local math-provider
rounding. Regeneration is a contributor tool, not a build step.

`provenance.json` records the original capture's processor geometry and numerical
provider. The Rust generator neither reads nor rewrites this metadata. It also
leaves numerical and scheduler fixtures intact.
