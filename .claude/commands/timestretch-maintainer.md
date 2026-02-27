Run the timestretch-maintainer skill to perform safe, tiered maintenance on this Rust DSP codebase.

## Setup

Before doing anything else, read these files in full:

1. `skills/timestretch-maintainer/SKILL.md` — canonical skill instructions
2. `skills/timestretch-maintainer/references/repo-maintenance-map.md` — repo command and file map
3. `skills/timestretch-maintainer/references/quality-and-benchmark-policy.md` — heavy-test and DSP quality guidance

## Tier Selection

Use tier: **$ARGUMENTS** (default to `fast` if blank).

Valid tiers: `fast`, `quality`, `full`.

## Scripts

- Tiered checks: `bash skills/timestretch-maintainer/scripts/check_tiers.sh <tier>`
- Full maintenance flow: `bash skills/timestretch-maintainer/scripts/maintain.sh --mode <tier>`

## Execution

Follow the "Run Workflow" section from the shared SKILL.md exactly. Report findings using the "Reporting Format" described there.
