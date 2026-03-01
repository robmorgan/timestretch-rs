---
name: timestretch-arch-perf-review
description: Review DSP architecture and realtime performance. Use when asked for architecture review, performance review, realtime readiness, latency/callback budget checks, DSP path comparison, or batch-vs-stream quality risk analysis.
---

# Timestretch Arch/Perf Review (Codex Wrapper)

This is the Codex entry point. All canonical instructions live in the shared location.

## Instructions

Read and follow the shared skill document:

- **Skill instructions**: `skills/timestretch-arch-perf-review/SKILL.md`
- **Review rubric**: `skills/timestretch-arch-perf-review/references/review-rubric.md`
- **Command matrix**: `skills/timestretch-arch-perf-review/references/command-matrix.md`

## Scripts

- **Tiered review runner**: `bash skills/timestretch-arch-perf-review/scripts/run_review_tier.sh <static|runtime|deep>`
- **Report renderer**: `bash skills/timestretch-arch-perf-review/scripts/render_report.sh <run_dir>`
