---
name: timestretch-maintainer
description: Maintain and update the timestretch-rs Rust DSP codebase with safe, tiered checks and conservative fixes. Use when asked to run maintenance passes, keep CI green, upgrade dependencies safely, investigate quality regressions, or prepare low-risk upkeep changes across Cargo, tests, benchmarks, and GitHub Actions CI.
---

# Timestretch Maintainer (Codex Wrapper)

This is the Codex entry point. All canonical instructions live in the shared location.

## Instructions

Read and follow the shared skill document:

- **Skill instructions**: `skills/timestretch-maintainer/SKILL.md`
- **Repo maintenance map**: `skills/timestretch-maintainer/references/repo-maintenance-map.md`
- **Quality & benchmark policy**: `skills/timestretch-maintainer/references/quality-and-benchmark-policy.md`

## Scripts

- **Tiered checks**: `bash skills/timestretch-maintainer/scripts/check_tiers.sh <fast|quality|full>`
- **Full maintenance**: `bash skills/timestretch-maintainer/scripts/maintain.sh --mode <tier>`
