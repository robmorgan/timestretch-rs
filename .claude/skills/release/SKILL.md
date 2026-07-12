---
name: release
description: Cut a timestretch release — CHANGELOG fold, version bump, CI checks, signed tag, GitHub release, crates.io publish. Use when Rob asks to release/publish/ship a new version of the crate. Usage: /release [X.Y.Z]
---

# Release timestretch

Cut a release of the `timestretch` crate end to end. A release is consistent only
when **five places agree**: `Cargo.toml` version, `CHANGELOG.md` section, README
install snippet, annotated git tag `vX.Y.Z`, GitHub release, and the crates.io
publish. Past releases have been left half-done (v0.5.0 had a tag and a crates.io
publish but no GitHub release for months) — so always audit first, repair second,
release third.

## Phase 1 — Preflight & state audit (read-only)

1. Get on a current main: `git checkout main && git pull --ff-only`. If the working
   tree has uncommitted WIP (e.g. ROADMAP rewrites), stash it with a named stash
   (`git stash push -m "wip-during-release" <files>`) and **restore it at the very
   end**. Never commit WIP into the release; never use `--allow-dirty`.
2. Audit and report the current state before changing anything:
   - Last tag: `git tag | sort -V | tail -3` and `git log -1 --oneline <tag>`
   - GitHub releases: `gh release list --limit 3`
   - crates.io: `curl -s https://index.crates.io/ti/me/timestretch | tail -1`
     (sparse index; one JSON line per version — the crates.io API at
     `crates.io/api/v1/...` blocks non-browser user agents, don't use it)
   - `Cargo.toml` `version`, and whether `CHANGELOG.md` has an `## Unreleased` section
3. If a previous version is half-released (tag exists but no GitHub release, publish
   without tag, etc.), tell Rob and repair that first — e.g. backfill a GitHub
   release from the existing tag with `gh release create <tag> --verify-tag` and
   notes written per Phase 5. Do not move or re-point an already-pushed tag.
4. If everything already agrees and there is nothing unreleased, report "nothing to
   do" and stop.

## Phase 2 — Version choice

If Rob gave a version argument, use it. Otherwise propose one from the `Unreleased`
section — while the crate is 0.x, breaking changes mean a **minor** bump, otherwise
patch — and confirm the choice with AskUserQuestion before touching files.

## Phase 3 — File updates (four release files only)

1. `CHANGELOG.md`: rename `## Unreleased` → `## X.Y.Z`. Then cross-check the section
   against `git log --oneline vPREV..HEAD` for public API changes that were never
   changelogged, and add them (the v0.6.0 varispeed API was missing until this
   check). Keep entries crate-focused; repo tooling goes under a `### QA` heading.
2. `README.md`: bump the install snippet (`timestretch = "X.Y.Z"`). Badges already
   exist at the top; don't duplicate them.
3. `Cargo.toml`: bump `version`, then `cargo check` to refresh `Cargo.lock`.
4. `desktop/` and `web/` use `path = ".."` dependencies with no version pin — no
   updates needed there.

## Phase 4 — CI checks (before committing; run as one background command, ~3 min)

All four must pass — same set as CI:

```
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo test --all-targets
```

The `.claude/settings.json` pre-commit hook re-runs fmt+clippy at commit time; that
is a safety net, not a substitute. Note clippy is also expected to pass with
`--features qa-harnesses` — check it if qa files changed since the last release.

## Phase 5 — Commit, tag, push

Commit and tag signing hangs on a 1Password `op-ssh-sign` approval prompt: run these
in **background Bash** and tell Rob to approve the prompt.

1. Stage only the four release files (`Cargo.toml Cargo.lock CHANGELOG.md README.md`).
2. Commit in house style — subject `Release vX.Y.Z`, body of 3–6 lines summarising
   the span since the previous version ("Since X.Y this release lands ..."), ending
   with the `Co-Authored-By` trailer.
3. `git tag -a vX.Y.Z -m "vX.Y.Z"`, then `git push origin main vX.Y.Z`.

## Phase 6 — GitHub release

Write notes to a scratchpad file in the house style used by v0.4.0/v0.5.0/v0.6.0:

- `## What's Changed` header, then a one-sentence framing of the release with a link
  to `https://crates.io/crates/timestretch/X.Y.Z`
- Themed `###` sections; **Breaking Changes first** when present; include measured
  numbers (latencies, quality scores, cents) where the CHANGELOG or PR descriptions
  have them — pull from `gh pr view <n>` for merged PRs in the span
- Footer: `**Full Changelog**: https://github.com/robmorgan/timestretch-rs/compare/vPREV...vX.Y.Z`

Then: `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <file> --verify-tag --latest`

## Phase 7 — crates.io publish

1. Working tree must be clean: stash any remaining WIP (see Phase 1).
2. `cargo publish --dry-run` — must verify cleanly.
3. `cargo publish` (credentials are in `~/.cargo/credentials.toml`).
4. Pop the stash; confirm with `git status --short` that WIP is restored.

## Phase 8 — Report

Give Rob: the GitHub release URL, the crates.io URL, a note that docs.rs takes a few
minutes to build the new version, and confirmation that stashed WIP was restored.
