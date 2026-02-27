# Quality And Benchmark Policy

## Tiered Execution Policy

1. Default to `fast` tier for regular maintenance.
2. Use `quality` tier for quality-focused tasks and DSP regression investigations.
3. Use `full` tier only when user explicitly requests heavy checks.

## Heavy Check Guidance

1. Treat `tests/reference_quality.rs` as long-running and CPU intensive.
2. Treat `cargo run --release --example benchmark_quality` as a heavyweight analysis workflow.
3. Avoid running heavy checks implicitly during routine maintenance.
4. Explain expected runtime and output behavior before starting heavy runs.

## Practical Workflow

1. Run `fast` tier first and fix any compile/lint/test failures.
2. Escalate to `quality` tier only if requested or if fast checks pass but quality risk remains.
3. Escalate to `full` tier only with explicit user intent.
4. Record which tier was executed and why in final reporting.

## Reporting Requirements

1. Report findings first, ordered by severity.
2. Report commands and tiers executed.
3. Report unresolved risks:
- Potential quality drift not covered by selected tier.
- Heavy checks skipped due policy or user scope.
