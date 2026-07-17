---
name: readme-screenshot
description: Recapture the README screenshot (docs/desktop-app.png) by launching the desktop app with a real track, playing it, and screenshotting the window. Use when Rob asks to update/refresh the README or docs screenshot.
---

# Updating the README screenshot

Replaces `docs/desktop-app.png` (referenced from the repo-root `README.md`)
with a fresh capture of the desktop app mid-playback.

## Track

Always use the real music track so the waveform looks interesting:

```
benchmarks/audio/public-corpus/01-Interplanetary_Criminal-Saucers.mp3
```

Its `.tsanalysis.json` sidecar is cached next to it, so load/analysis is fast.
Do NOT use the synthetic `test_audio/` tones — they make a boring screenshot.

## Procedure

All paths below are relative to the repo root unless noted.

1. **Build & launch** (in `desktop/`, which is excluded from the workspace):

   ```bash
   cargo build --release
   RUST_LOG=info target/release/timestretch-desktop \
     ../benchmarks/audio/public-corpus/01-Interplanetary_Criminal-Saucers.mp3 \
     > "$SCRATCH/app.log" 2>&1 &
   sleep 4
   ```

2. **Compile the UI helper** from the copy kept in this skill directory
   (`uidrive.swift`, subcommands: `winid <name>` / `list` / `click <x> <y>` /
   `drag <x1> <y1> <x2> <y2> <ms> [holdBeforeMs] [holdAfterMs]`):

   ```bash
   swiftc -O -o "$SCRATCH/uidrive" desktop/.claude/skills/readme-screenshot/uidrive.swift
   ```

3. **Safety check before driving the pointer** — CGEvent clicks land on
   whatever is topmost. Bring the app frontmost, then dump the layer-0
   window list and confirm the app window is first and at the expected
   position. If another window is on top or Rob is clearly active, STOP
   and hand over to him.

   ```bash
   osascript -e 'tell application "System Events" to set frontmost of first process whose name contains "timestretch" to true'
   "$SCRATCH/uidrive" winid timestretch    # -> "windowID x y w h" (screen points)
   "$SCRATCH/uidrive" list | head -5
   ```

   At the default 1000×632 window at (256, 92): Play button ≈ (279, 452),
   overview strip ≈ y 407 spanning x 264–1248.

4. **Seek somewhere interesting, play, capture** in ONE staged command so
   tool-call gaps don't drift the transport. Seek via the overview strip to
   ~40% in (past the intro, into the busy part), then play and let the
   playhead settle mid-view:

   ```bash
   "$SCRATCH/uidrive" click 650 407 && sleep 0.5 && \
   "$SCRATCH/uidrive" click 279 452 && sleep 4 && \
   screencapture -x -o -l <windowID> "$SCRATCH/shot.png"
   ```

   `-o` (no shadow) gives an exact 2× point→pixel mapping; without it the
   shadow margin breaks coordinates and adds transparent padding.

5. **Verify the capture** by reading the image before replacing anything:
   - Transport shows **Pause** (i.e. playing) and a nonzero time readout.
   - Beat grid overlay visible on the zoomed waveform; Keylock deck selected
     (the README alt text mentions both).
   - Track name row shows the mp3, sane BPM detection.
   - Dimensions should be 2000×1264 (2× the default window).

6. **Replace and clean up**:

   ```bash
   cp "$SCRATCH/shot.png" docs/desktop-app.png
   pkill -f timestretch-desktop
   grep -c underrun "$SCRATCH/app.log"   # expect 0
   ```

Commit only if asked. The README alt text lives at the image reference in
`README.md` — update it if the captured state no longer matches it.
