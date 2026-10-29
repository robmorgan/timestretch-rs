# Benchmarks

Use [qualification](QUALIFICATION.md) for the current generated-signal, DSP,
worker/callback and listening workflows. JSON/CSV/HTML artifacts include source
and executable hashes. Numerical bounds and physical callback deadlines are
independent of backend selection.

The repository manifest also describes optional public audio for BPM and Rubber
Band integration tests. Run `scripts/fetch_public_corpus.sh` to obtain the public
corpus. Locally supplied tracks are optional; ordinary tests do not require them.

Older local timing results are summarized in
[historical-core.json](baselines/historical-core.json). They are historical
measurements, not dependencies of current tests or claims about other devices.
