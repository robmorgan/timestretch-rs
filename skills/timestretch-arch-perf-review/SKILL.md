---
name: timestretch-arch-perf-review
description: Review architecture and realtime performance risks in timestretch-rs. Use for architecture review, performance review, realtime readiness, latency/callback budget analysis, DSP path comparison, and batch-vs-stream quality risk assessment.
---

# Timestretch Architecture/Performance Review

## Overview

Run a tiered review of DSP architecture and realtime performance using deterministic scripts and a fixed report contract.

Default review mode is `runtime`.

## Trigger Intent

Use this skill when the request matches any of the following:
1. Architecture review
2. Performance review
3. Realtime readiness check
4. Latency or callback budget validation
5. DSP path comparison
6. Batch-vs-stream quality risk review

## Workflow

1. Select a tier (`runtime` by default).
2. Run `bash skills/timestretch-arch-perf-review/scripts/run_review_tier.sh <tier>`.
3. Open the generated report in `target/arch_perf_review/<timestamp>_<tier>/report.md`.
4. Present findings first, severity-ordered, with evidence command and remediation hints.

## Tier Policy

1. `static`: Repository inspection only.
2. `runtime`: Default path for live-readiness and callback behavior.
3. `deep`: Run only when explicitly requested. Includes heavier quality/baseline checks and optional external corpus dependency.

Exact commands and tier gating are defined in:
1. [`references/command-matrix.md`](references/command-matrix.md)

## Output Contract

Every review report must include these sections, in this order:
1. `Findings (severity-ordered)`
2. `Architecture Notes`
3. `Realtime Budget`
4. `Benchmark Snapshot`
5. `Risks`
6. `Recommended Next Actions`

Findings must follow the rubric in:
1. [`references/review-rubric.md`](references/review-rubric.md)

## Stop Conditions

Stop and report if any condition is true:
1. External benchmark corpus/dependencies are missing for requested deep checks.
2. A required command is unavailable in the current environment.
3. The request asks for unexpected non-review mutations outside architecture/performance review scope.

## Command Policy

1. Keep this skill review-focused.
2. Continue running remaining commands even after failures to capture full status.
3. Do not alter Rust library APIs as part of this review skill.
4. Prefer deterministic script execution over ad-hoc command rewriting.

## Scripts

1. Tier runner:
`bash skills/timestretch-arch-perf-review/scripts/run_review_tier.sh <static|runtime|deep>`
2. Report renderer:
`bash skills/timestretch-arch-perf-review/scripts/render_report.sh <run_dir>`

