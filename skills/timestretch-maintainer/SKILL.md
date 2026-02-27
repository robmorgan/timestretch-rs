# Timestretch Maintainer

## Overview

Run repeatable maintenance workflows for this Rust DSP project. Execute fast hygiene checks by default, run quality checks on request, and run heavy benchmark/reference checks only when explicitly requested.

## Run Workflow

1. Inspect repository state with `git status --short`.
2. Preserve unrelated user changes; never revert files unless explicitly requested.
3. Select the tier:
- `fast` for daily maintenance.
- `quality` for quality-focused maintenance.
- `full` only when explicitly requested for heavy DSP validation.
4. Execute checks via `bash scripts/check_tiers.sh`.
5. Apply only conservative, low-risk fixes.
6. Stop and report when fixes require broad refactors, behavioral redesign, or risky dependency jumps.

## Tier Matrix

| Tier | Intended Use | Commands |
| --- | --- | --- |
| `fast` | Keep CI green quickly | `cargo fmt --all --check`; `cargo clippy --all-targets -- -D warnings`; `cargo test --all-targets` |
| `quality` | Investigate quality stability without full benchmark runtime | `fast` + `cargo test --test quality -- --nocapture`; `cargo test --test quality_gates -- --nocapture` |
| `full` | Perform explicit heavy DSP validation | `quality` + `cargo test --test reference_quality -- --nocapture`; `cargo run --release --example benchmark_quality` |

## Use Scripts

1. Run tiered checks directly:

```bash
bash skills/timestretch-maintainer/scripts/check_tiers.sh fast
```

2. Run full maintenance flow:

```bash
bash skills/timestretch-maintainer/scripts/maintain.sh --mode quality
```

3. Preview commands without execution:

```bash
bash skills/timestretch-maintainer/scripts/maintain.sh --mode full --dry-run
```

4. Include conservative dependency updates when requested:

```bash
bash skills/timestretch-maintainer/scripts/maintain.sh --mode fast --update-deps
```

## Conservative Fix Policy

1. Prefer targeted fixes over refactors.
2. Avoid changing public behavior unless tests or requested scope require it.
3. Avoid speculative optimizations during maintenance passes.
4. Keep patch size small and easy to review.

## Stop Conditions

Stop and report instead of continuing when any condition is true:
1. Required fix would need architectural redesign.
2. Required dependency upgrade would force major-version migration.
3. Heavy tests are needed but were not explicitly requested.
4. Unexpected unrelated repository changes appear during the task.

## Reporting Format

Report in this order:
1. Findings first, ordered by severity.
2. Changes made and why.
3. Checks executed and pass/fail status.
4. Remaining risks and follow-up actions.

## Example Triggers

Use this skill for prompts like:
1. "Run a maintenance pass and keep CI green."
2. "Do a conservative dependency update and verify tests."
3. "Check whether quality regressions appeared after recent DSP changes."
4. "Run full maintenance including benchmark quality checks."

## References

1. Use [`references/repo-maintenance-map.md`](references/repo-maintenance-map.md) for repository-specific command and file map.
2. Use [`references/quality-and-benchmark-policy.md`](references/quality-and-benchmark-policy.md) for heavy-test and DSP quality guidance.
