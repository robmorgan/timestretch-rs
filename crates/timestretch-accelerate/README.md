# timestretch-accelerate

Optional macOS companion crate for `timestretch` QA and desktop auditioning. This crate
contains the unsafe Apple framework boundary; the main library remains safe
Rust. It is not selected by default, even when the `apple-qa` feature is enabled.

`RealFft` owns a radix-2 plan and 64-byte-aligned split buffers matching the
reference. Ordinary `Vec<f32>` alignment allowed different Accelerate numerical
results in the measured music corpus; alignment is part of this adapter’s contract. All lengths are validated before
FFI; processing requires an exclusive mutable borrow. Plans can move to the audio
thread (`Send`) and are released on drop. No plan cache or callback allocation
is used. The packed output places DC at index 0 and Nyquist at N/2. The forward
transform is conventionally scaled and the inverse is unnormalized.

`polar`, `sin_cos` and `reciprocal` call vDSP/vForce with
length-checked, disjoint slices. The polar adapter temporarily zeros the packed
Nyquist slot so it is not treated as DC's imaginary component, then restores it.

`AllocationProbe` optionally discovers the QA DYLD allocation observer. The
observer is injected only into separate allocation audit processes, never into
the performance measurements or desktop app.

```sh
cargo test -p timestretch-accelerate --release
cargo test --features apple-qa --release --lib processor::qa
```

See [the experiment protocol](../../benchmarks/BACKENDS.md).

## Window table generation

The `generate_tables` example regenerates the core's six window tables from
explicit `f32` formulas using this crate's sine/cosine adapter. It requires macOS
and Apple Accelerate. Check the committed bytes with:

```sh
cargo run -p timestretch-accelerate --example generate_tables -- --check
```

The default mode also checks without writing. Use `--write` to regenerate and
`--output DIRECTORY` to select a different directory. See the [table documentation](../timestretch-core/src/processor/tables/README.md)
for formulas, file format, and provider reproducibility details.
