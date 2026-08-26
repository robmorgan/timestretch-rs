---
name: readme-screenshot
description: Capture a fresh presentation-quality shot of the desktop app and refresh the README image (docs/desktop-app.png). Use after big UI changes when Rob asks to update/refresh the README or docs screenshot. Usage: /readme-screenshot
---

# Refresh the README screenshot

Capture a presentation-quality shot of the running `desktop/` app mid-playback
and replace `docs/desktop-app.png`, the image the repo-root `README.md` embeds
(`![...](docs/desktop-app.png)` with an italic caption below it). Rob runs this
after big UI changes, not on every release. The shot must be **reviewed and
approved by Rob in-chat before anything in the repo changes**.

`$SCRATCH` below is the session scratchpad directory. Paths are relative to
the repo root unless noted. The `desktop:verify` skill shares this machinery
(`uidrive.swift` lives in this skill directory) and has deeper UI-driving
details (coordinates, drag gotchas, transport-state reading).

## Phase 1 — Preflight & build

1. Working-tree guard: `git status --short docs/desktop-app.png README.md`.
   If either has uncommitted changes, stop and ask Rob before proceeding —
   never clobber unreviewed work.
2. Track check — always use the real reference track (never the synthetic
   `test_audio/` tones; flat waveforms make a boring screenshot):

   ```
   benchmarks/audio/public-corpus/01-Interplanetary_Criminal-Saucers.mp3
   ```

   `benchmarks/audio/` is gitignored, so the file may be absent. If missing,
   fetch it with `scripts/fetch_public_corpus.sh` (idempotent, SHA-256
   verified, pulls from archive.org).
3. Build (in `desktop/`, which is excluded from the workspace):
   `cd desktop && cargo build --release`

## Phase 2 — Launch & stage

1. Launch from `desktop/`:

   ```bash
   RUST_LOG=info target/release/timestretch-desktop \
     ../benchmarks/audio/public-corpus/01-Interplanetary_Criminal-Saucers.mp3 \
     > "$SCRATCH/app.log" 2>&1 &
   ```

   `argv[1]` auto-loads the track (skips the file dialog). The app does NOT
   autoplay — playback needs a synthetic Play click below. Sidecars
   (`.tspeaks`) are gitignored and likely absent, so a cold launch pays the
   full decode + peaks + analysis cost — wait until the log/UI shows analysis
   done (BPM readout populated) before staging, don't assume 4 s is enough.
2. Compile the UI helper from the copy in this skill directory
   (subcommands: `winid <name>` / `list` / `click <x> <y>` /
   `drag <x1> <y1> <x2> <y2> <ms> [holdBeforeMs] [holdAfterMs]`):

   ```bash
   swiftc -O -o "$SCRATCH/uidrive" .claude/skills/readme-screenshot/uidrive.swift
   ```

3. Safety check before driving the pointer — CGEvent clicks land on whatever
   is topmost. Bring the app frontmost, dump the layer-0 window list, and
   confirm the app window is first at the expected position. If another
   window is on top or Rob is clearly active, STOP and hand over to him.

   ```bash
   osascript -e 'tell application "System Events" to set frontmost of first process whose name contains "timestretch" to true'
   "$SCRATCH/uidrive" winid timestretch    # -> "windowID x y w h" (screen points)
   "$SCRATCH/uidrive" list | head -5
   ```

   The window has no size persistence, so it always opens at the default
   1000×632 pt (with titlebar). At (256, 92): Play button ≈ (279, 452),
   overview strip ≈ y 407 spanning x 264–1248. If the origin differs, offset
   the click targets accordingly.
4. Stage for an **engaging** shot: tempo offset at ±8% and the playhead in a
   transient-rich section so the zoomed waveform shows all three colors
   (blue body + orange/white transient bursts) — never the intro.

   a. **Tempo ±8% via the Target BPM field**, not the slider — the slider
      spans far more than ±20%, so a click lands wildly off (observed:
      +36.6%). Click the field, select-all, type `detected BPM × 1.08`
      (one decimal), Enter. **Gotcha: committing the field restarts the
      transport at 0:00**, so always set tempo BEFORE navigating.
   b. **Play**, then navigate with the `+16` bar beat-jump button, which
      keeps playback running (overview-strip seek clicks PAUSE playback —
      don't use them mid-staging). Pick the destination from the overview
      colors: orange/white patches = transient-heavy. For Saucers
      (127 BPM) that's ~bar 100–125 (≈3:10–3:55).
   c. **Gotcha: instantaneous synthetic clicks sometimes vanish** — the UI
      samples held state per frame, so a plain `click` on Jump buttons
      drops ~half the presses. Use press-and-hold clicks and verify the
      bar counter from a probe screenshot between batches:

   ```bash
   # window-relative pt + window origin from winid; targets below assume (256, 92)
   "$SCRATCH/uidrive" click 520 533 && sleep 0.3 && \
   osascript -e 'tell application "System Events" to keystroke "a" using command down' && \
   osascript -e 'tell application "System Events" to keystroke "137.2"' && \
   osascript -e 'tell application "System Events" to key code 36' && sleep 0.5 && \
   "$SCRATCH/uidrive" click 279 452 && sleep 1     # Play
   # then, per batch of jumps (press-and-hold click = drag in place):
   "$SCRATCH/uidrive" drag 411 473 411 473 30 120 120   # +16 bars
   ```

   Let it play a couple of seconds after the last jump so the playhead
   sits mid-view before capturing.

## Phase 3 — Capture

1. Wake the display — `screencapture` silently produces a black/failed
   capture when the display is asleep: `caffeinate -u -t 3`
2. Capture (no shadow, exact 2× point→pixel mapping):

   ```bash
   screencapture -x -o -l <windowID> "$SCRATCH/readme-shot.png"
   ```

   Expect **2000×1264 px** (2× the default window). If the geometry from
   `winid` isn't 1000×632, something resized the window — fix that before
   capturing, don't ship an odd-sized shot.

## Phase 4 — Review gate (Rob approves before any repo change)

1. Read `$SCRATCH/readme-shot.png` so it renders in-chat and sanity-check it:
   transport shows Pause (i.e. playing) with a time readout well past the
   intro, tempo readout shows **±8.0%** with the target BPM, beat grid
   overlay visible on the zoomed waveform with **orange/white transient
   coloring** (not a flat all-blue stretch), keylock deck selected (the
   README alt text mentions both), track name row shows the mp3, sane BPM,
   no dialogs or debug overlays.
2. Ask Rob to approve or request restaging (different track position, loop
   engaged, tempo nudged, etc.). Re-stage and re-capture as needed.
   **Do not overwrite `docs/desktop-app.png` until he approves.**

## Phase 5 — Refresh repo & commit

1. On approval: `cp "$SCRATCH/readme-shot.png" docs/desktop-app.png`
2. Verify the README embed still exists
   (`grep 'docs/desktop-app.png' README.md`). Only touch `README.md` if the
   embed is missing, or if the captured state no longer matches the alt
   text / italic caption under it — then update those to match.
3. Commit the image (plus `README.md` only if touched) — commit signing hangs
   on a 1Password `op-ssh-sign` prompt, so run in **background Bash** and tell
   Rob to approve it. Subject `docs: refresh README screenshot`, a 1–2 line
   body noting what UI change prompted the refresh, ending with the
   `Co-Authored-By` trailer. Do **not** push unless Rob asks.

## Phase 6 — Cleanup & report

1. `pkill -f timestretch-desktop`
2. `grep -ci 'underrun\|error' "$SCRATCH/app.log"` — expect 0; mention
   anything found.
3. Report: committed image dimensions and file size, the commit hash, and
   that the README embed is intact.
