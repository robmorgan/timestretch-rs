# Qualification

The maintained workflow is self-contained. It uses checked-in native fixtures,
generated audio, a fixed quality evaluator, allocator audits and the current
numerical providers. It does not require historical measurements or external
proprietary implementations. See [commands and protocols](../qa/README.md).

```sh
python3 -B tools/qualify.py --output target/qualification-new --strict
# Explicit physical-device qualification, muted:
python3 -B tools/qualify.py --output target/device-new --device-seconds 15 --strict
python3 -m http.server 8000 --directory target/qualification-new
```

Core/session throughput, worker bursts and delivery callbacks are separate
measurements. The full host matrix screens 44.1/48/96 kHz, mono/stereo,
64/256/1024 frames and tempos 0.5/1/2 three times at 1,000 callbacks. Each
provider's worst maximum/budget and p99/budget cells, plus 96 kHz stereo/64/2,
receive five repetitions of 10,000 callbacks. Reserve and joint-boundary stress
runs are separate. Require zero observed underruns/deadline misses and p99.9
within half the callback period where adequately sampled. These gates are
unchanged; a quick smoke run does not establish qualification.

The desktop retains its measured 50 ms reserve and macOS audio scheduling.
Reserve is additional host latency, separate from algorithm lookahead, pending
output, control quantization and device latency. Other platforms report their
actual policy. Physical-device tests are opt-in, use the default device at its
unchanged rate, and provide no hardware loopback/xrun measurement.

[Historical results](baselines/historical-core.json) summarize the earlier
local measurement campaign. Those results remain attributed to their original
binaries; they are not fresh measurements of a renamed build. Earlier ordinary
worker scheduling trials failed; the final local campaign passed 369 host runs,
18 device conditions and 12 boundary runs. This does not qualify other hardware
or record a human listening preference. Original evidence is preserved privately;
contributors do not need it to run the maintained workflow.
