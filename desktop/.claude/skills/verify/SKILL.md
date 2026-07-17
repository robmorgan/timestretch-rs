---
name: verify
description: Build, launch, and drive the desktop deck app on macOS to verify UI/audio changes end-to-end (screenshots + synthesized clicks/drags).
---

# Verifying the desktop deck app

## Build & launch

```bash
cargo build --release          # in desktop/ (excluded from the workspace)
RUST_LOG=info target/release/timestretch-desktop \
  ../test_audio/dense_mastered_128bpm.wav > app.log 2>&1 &
```

`argv[1]` loads a track at startup (skips the file dialog). Test tracks live
in `test_audio/`; `dense_mastered_128bpm.wav` is 12 s / 128 BPM — note it
EOFs (auto-stop, position resets to 0) fast, so re-click Play between slow
steps, and remember Play/Pause is one toggle button (two clicks = pause).

## Driving the UI (TCC is unblocked for this)

No cliclick or python-Quartz on this machine. Compile the CGEvent helper
(source kept in the session scratchpad as `uidrive.swift`; recreate it if
gone — winid/click/drag subcommands posting CGEvents):

```bash
swiftc -O -o uidrive uidrive.swift   # swift JIT is too slow for timing-sensitive drags
./uidrive winid timestretch          # -> "windowID x y w h" (screen points)
./uidrive click <x> <y>
./uidrive drag <x1> <y1> <x2> <y2> <ms> [holdBeforeMs] [holdAfterMs]
osascript -e 'tell application "System Events" to set frontmost of first process whose name contains "timestretch" to true'
```

## Screenshots & coordinates

```bash
screencapture -x -o -l <windowID> shot.png   # -o (no shadow) => exact 2x mapping
```

With `-o`, image px = 2 × window pt: `screen = (win_x + img_x/2, win_y + img_y/2)`.
At the default 1000×632 window at (256, 92): Play button ≈ (279, 452),
zoomed waveform center ≈ (756, 272), overview strip ≈ y 400, time readout
next to Play. Without `-o` the shadow margin breaks the mapping — always use `-o`.

## Safety: check for an active user session FIRST

CGEvent clicks/drags land on whatever window is topmost at those screen
coordinates — **before every driving sequence**, dump the layer-0 window
list front-to-back (`CGWindowListCopyWindowInfo`) and confirm (a) the app
window is frontmost at the click points and (b) its position hasn't moved
(Rob may drag it aside or be actively working — seen 2026-07-17: clicks
went into his Ghostty terminal after he moved the test window off-screen).
If another app is on top or the user is clearly active, STOP driving
pointer events and hand live testing to Rob. AX-element clicks (see the
desktop-ui-verification memory) are safe for buttons regardless of
z-order, but drags have no AX equivalent.

## Timing gotchas

- Stage everything in ONE bash command (`(drag &) && sleep … && screencapture …`);
  gaps between tool calls are seconds long and the 12 s track EOFs meanwhile.
- Verify transport state from the screenshot (Pause label = playing,
  Resume = paused, Play + dim Stop = stopped) before interpreting positions.
- Audio itself can't be observed here — verify positions/transport/log
  (`grep underrun app.log`); audible feel is Rob's call.
