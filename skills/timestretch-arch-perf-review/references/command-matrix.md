# Tier Command Matrix

## Default Tier

Default to `runtime` unless the user explicitly requests `static` or `deep`.

## Static Tier (Inspection Only)

Use repository inspection commands only:
1. `rg --files src tests benchmarks`
2. `sed -n '1,220p' ARCHITECTURE.md`
3. `sed -n '1,260p' README.md`
4. `rg -n '(realtime|latency|callback|allocation|hybrid|phase_vocoder|wsola|budget|parity)' src tests README.md ARCHITECTURE.md`

## Runtime Tier (Default)

Run these commands and capture pass/fail:
1. `cargo test --release --test realtime_dj_conditions -- --nocapture`
2. `cargo test --release --test realtime_allocations -- --nocapture`
3. `TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --release --test quality_gates quality_gate_streaming_worst_case_callback_budget -- --nocapture`
4. `cargo test --release --test streaming_batch_parity -- --nocapture`
5. `cargo test --release --test benchmarks bench_streaming -- --nocapture`

## Deep Tier (Explicit Request Only)

Only run when explicitly requested:
1. `cargo test --release --test quality_gates quality_gate_batch_vs_stream_hybrid_subset -- --nocapture`
2. `./benchmarks/run_m0_baseline.sh`

Deep tier may require external corpus/dependencies; if missing, mark command as skipped with reason.
Optional timeout guard:
1. `TIMESTRETCH_REVIEW_CMD_TIMEOUT_SECS=<seconds> bash skills/timestretch-arch-perf-review/scripts/run_review_tier.sh deep`
