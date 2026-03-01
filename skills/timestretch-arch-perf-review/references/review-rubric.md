# Architecture/Performance Review Rubric

## Severity Model

Use these priorities for findings:
1. `P0`: Hard realtime/safety break (allocation in callback path, callback budget failure, severe dropouts risk).
2. `P1`: Major behavior/reliability risk (streaming parity drift, realtime behavior mismatch, likely audible regressions in live use).
3. `P2`: Quality/performance degradation (benchmark regression, deep quality-gate failure, CPU headroom collapse under load).
4. `P3`: Informational/maintainability gaps (missing docs, skipped optional checks, instrumentation holes).

## Required Finding Format

Each finding line must include:
1. Priority (`P0`..`P3`)
2. File and line reference
3. Impact statement
4. Evidence command
5. Remediation hint

Recommended single-line format:

`P1 | file: tests/streaming_batch_parity.rs:1 | impact: stream vs batch divergence risk | evidence: cargo test --release --test streaming_batch_parity -- --nocapture | remediation: inspect hybrid stream state transitions and add parity regression coverage`

## Findings-First Rule

Always report findings before summaries. Order findings by severity (`P0` then `P1`, `P2`, `P3`).

## Mandatory Report Sections

Reports must include all sections exactly:
1. `Findings (severity-ordered)`
2. `Architecture Notes`
3. `Realtime Budget`
4. `Benchmark Snapshot`
5. `Risks`
6. `Recommended Next Actions`

## Evidence Policy

1. Findings must cite the command that produced the evidence.
2. If a command is skipped, record the reason and classify as `P3` unless it blocks a required contract item.
3. If no failures occur, emit a single `P3` informational line indicating no command-level failures.

