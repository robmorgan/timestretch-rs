---
name: release
description: Cut a timestretch release — CHANGELOG fold, workspace version bump, CI checks, signed tag, GitHub release, multi-crate crates.io publish. Use when Rob asks to release/publish/ship a new version of the crates. Invoke as /release [X.Y.Z] (Claude Code) or $release (Codex).
---

# Release timestretch

Cut a release of the timestretch workspace end to end. Four library crates are
published together:

| Crate | Path | Version |
|---|---|---|
| `timestretch-accelerate` | `crates/timestretch-accelerate` | own line (`0.1.x`) |
| `timestretch-core` | `crates/timestretch-core` | lockstep `X.Y.Z` |
| `timestretch-analysis` | `crates/timestretch-analysis` | lockstep `X.Y.Z` |
| `timestretch` | repo root | lockstep `X.Y.Z` |

`timestretch-desktop`, `ab-tui` and `timestretch-qa` are `publish = false`.

A release is consistent only when **these agree**: the lockstep crate versions
and the root's dependency pins, the `CHANGELOG.md` section, the annotated git tag
`vX.Y.Z`, the GitHub release, and crates.io for every published crate. Past releases have been left half-done
(v0.5.0 had a tag and a crates.io publish but no GitHub release for months) — so
always audit first, repair second, release third.

## Phase 1 — Preflight & state audit (read-only)

1. Get on a current main: `git checkout main && git pull --ff-only`. If the working
   tree has uncommitted WIP (e.g. ROADMAP rewrites), stash it with a named stash
   (`git stash push -m "wip-during-release" <files>`) and **restore it at the very
   end**. Never commit WIP into the release; never use `--allow-dirty`.
2. Audit and report the current state before changing anything:
   - Last tag: `git tag | sort -V | tail -3` and `git log -1 --oneline <tag>`
   - GitHub releases: `gh release list --limit 3`
   - crates.io, per crate: `curl -s https://index.crates.io/ti/me/<crate> | tail -1`
     for `timestretch`, `timestretch-core`, `timestretch-analysis`,
     `timestretch-accelerate` (sparse index; one JSON line per version, and a
     `NoSuchKey` error means never published — the crates.io API at
     `crates.io/api/v1/...` blocks non-browser user agents, don't use it)
   - Each crate's `version`, and whether `CHANGELOG.md` has an
     `## X.Y.Z (unreleased)` section
   - `RELEASE_CHECKLIST.md`: walk Rob through any unticked items; it records
     release-specific gates (qualification, listening, allocation audits) that
     this skill does not run
3. If a previous version is half-released (tag exists but no GitHub release, publish
   without tag, etc.), tell Rob and repair that first — e.g. backfill a GitHub
   release from the existing tag with `gh release create <tag> --verify-tag` and
   notes written per Phase 6. Do not move or re-point an already-pushed tag.
4. If everything already agrees and there is nothing unreleased, report "nothing to
   do" and stop.

## Phase 2 — Version choice

If Rob gave a version argument, use it. Otherwise propose one from the unreleased
`CHANGELOG.md` section — while the crates are 0.x, breaking changes mean a
**minor** bump, otherwise patch — and confirm the choice with Rob before touching files. Also decide whether
`timestretch-accelerate` needs a release: only if
`git diff vPREV -- crates/timestretch-accelerate` is non-empty or it has never
been published; then bump its own patch/minor.

## Phase 3 — File updates

1. `CHANGELOG.md`: rename `## X.Y.Z (unreleased)` → `## X.Y.Z`. Then cross-check
   the section against `git log --oneline vPREV..HEAD` for public API changes that
   were never changelogged, and add them (the v0.6.0 varispeed API was missing
   until this check). Keep entries crate-focused; repo tooling goes under a
   `### QA` heading.
2. Lockstep versions → `X.Y.Z` in `Cargo.toml` (`[package] version` **and** the
   `timestretch-core` / `timestretch-analysis` dependency `version` pins),
   `crates/timestretch-core/Cargo.toml`, `crates/timestretch-analysis/Cargo.toml`,
   and `tools/qa/Cargo.toml` (unpublished but kept in lockstep). Development may
   already have set them; verify rather than assume.
3. If `timestretch-accelerate` is being released, bump
   `crates/timestretch-accelerate/Cargo.toml` and the matching pin in
   `crates/timestretch-core/Cargo.toml`.
4. `cargo check --workspace` to refresh `Cargo.lock`, then
   `grep -rn 'version = "PREV' Cargo.toml crates/*/Cargo.toml tools/qa/Cargo.toml`
   should find nothing stale.
5. `README.md` has no install snippet; only touch it if it names a version.
   `desktop/`, `tools/ab-tui` and `tools/qa` use path dependencies — no pins.

## Phase 4 — CI checks (before committing; run as one background command)

All must pass — the `AGENTS.md` verification set plus CI's no-default-features
contract test:

```
cargo test --release --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
python3 -B -m unittest discover -s tools -p 'test_*.py'
cargo test -p timestretch --no-default-features --lib --test session_contracts
```

Run these yourself before committing. Claude Code's `.claude/settings.json`
pre-commit hook re-runs fmt+clippy, but other agents (Codex) have no such hook.
CI also runs `timestretch-qa -- quality --strict` and `callback-audit`; run them
if DSP or callback code changed since the last release.

## Phase 5 — Commit, tag, push

Commit and tag signing hangs on a 1Password `op-ssh-sign` approval prompt: run these
in the **background** and tell Rob to approve the prompt.

1. Stage only the release files: `CHANGELOG.md Cargo.lock` and the manifests
   bumped in Phase 3 (plus `README.md` if touched).
2. Commit in house style — subject `Release vX.Y.Z`, body of 3–6 lines summarising
   the span since the previous version ("Since X.Y this release lands ...").
   No AI attribution trailer.
3. `git tag -a vX.Y.Z -m "vX.Y.Z"`, then `git push origin main vX.Y.Z`.

## Phase 6 — GitHub release

Write notes to a scratchpad file in the house style used by v0.4.0/v0.5.0/v0.6.0:

- `## What's Changed` header, then a one-sentence framing of the release with a link
  to `https://crates.io/crates/timestretch/X.Y.Z` (mention the companion crates
  and their versions when they ship)
- Themed `###` sections; **Breaking Changes first** when present; include measured
  numbers (latencies, quality scores, cents) where the CHANGELOG or PR descriptions
  have them — pull from `gh pr view <n>` for merged PRs in the span
- Footer: `**Full Changelog**: https://github.com/robmorgan/timestretch-rs/compare/vPREV...vX.Y.Z`

Then: `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <file> --verify-tag --latest`

## Phase 7 — crates.io publish

Cargo publishes multiple packages in dependency order in one command
(accelerate → core/analysis → timestretch) and waits for each to appear in the
index before the next. List only crates whose new version isn't on crates.io yet —
drop `-p timestretch-accelerate` when it wasn't bumped.

1. Working tree must be clean: stash any remaining WIP (see Phase 1).
2. Dry run — must verify cleanly:
   `cargo publish --dry-run -p timestretch-accelerate -p timestretch-core -p timestretch-analysis -p timestretch`
   (the "no documentation, homepage or repository" warnings on the sub-crates are
   known and non-blocking).
3. Same command without `--dry-run` (credentials are in `~/.cargo/credentials.toml`).
   If it fails partway, re-run with only the crates not yet published — never
   bump a version just to retry.
4. Pop the stash; confirm with `git status --short` that WIP is restored.

## Phase 8 — Report

Give Rob: the GitHub release URL, the crates.io URL for each published crate, a note that docs.rs takes a few
minutes to build the new version, and confirmation that stashed WIP was restored.
