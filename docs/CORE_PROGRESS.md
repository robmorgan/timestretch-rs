# Implementation status

The single-session architecture and all consumers are implemented. The desktop
uses worker-owned processing and bounded delivery queues. The maintained code
uses neutral component names and contributor workflows need no private inputs.

See [architecture](CORE_ARCHITECTURE.md), [contracts](SESSION_CONTRACT.md),
[qualification](../benchmarks/QUALIFICATION.md) and [historical measurements](../benchmarks/baselines/historical-core.json).

Historical timing results remain attached to their original binaries. Current
changes are checked through fixture, quality, deterministic render, lifecycle,
allocation, build and focused playback tests. Broader platform/device runtime
qualification and human listening require separate recorded runs.

Naming and workflow cleanup verification (28 October 2026):

- Zero former-name matches in maintained filenames and contents; original
  sources and historical manifests were checksum-verified before relocation.
- All 504 generated-signal renders match the pre-cleanup executable byte for
  byte. Numerical payloads of all three checked-in fixture files are unchanged.
- An isolated checkout with fresh build output and no private inputs passes
  all 549 workspace tests (one intentional extended campaign remains ignored),
  the generated-signal qualification smoke workflow and 11 Python protocol tests.
- Clippy, formatting, rustdoc, Rust 1.85 library checks, feature checks and
  Linux/Windows library cross-checks pass. Cross-checking is not device/runtime
  qualification on those systems.
- The smoke workflow passes fixture, quality, callback allocation and session
  allocation checks. Its reduced timing matrix is explicitly incomplete;
  strict mode rejects it, and device status is explicitly unmeasured.
