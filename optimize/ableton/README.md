# Ableton Live Reference Generation Pipeline

Generate **Ableton Live 11 Complex Pro** time-stretch references and score
the timestretch-rs library output against them.

## Prerequisites

1. **macOS** with Ableton Live 11 Suite installed at `/Applications/Ableton Live 11 Suite.app`
2. **Accessibility permissions** for Terminal/iTerm:
   System Settings → Privacy & Security → Accessibility
3. **Python 3.11+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) AbletonOSC** for runtime verification:
   - Download from https://github.com/ideoforms/AbletonOSC
   - Copy to `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC/`
   - Enable in Ableton: Preferences → Link/Tempo/MIDI → Control Surface

## One-Time Setup: Create Template

1. Open Ableton Live 11
2. Create a new project with **one audio track**
3. Set: sample rate = 44100 Hz, bit depth = 16-bit
4. Set master tempo to 120 BPM (will be overwritten by scripts)
5. Set warp mode on the audio track to **Complex Pro**
6. Configure export settings: WAV, same sample rate, no normalization, no dithering
7. Save as `template/base_template.als`

To inspect the template XML:
```bash
python scripts/create_als_project.py decompress template/base_template.als template/base_template.xml
```

## Track Catalog

Edit `track_catalog.json` to define source tracks and target BPMs:

```json
[
  {
    "id": "my-track",
    "source": "../samples/my_track.wav",
    "source_bpm": 124.0,
    "targets": [
      {"target_bpm": 115.0, "ratio": 0.9274},
      {"target_bpm": 128.0, "ratio": 1.0323}
    ],
    "stereo": true
  }
]
```

- `source`: Path relative to the catalog file (or absolute)
- `ratio`: `target_bpm / source_bpm`
- `stereo`: Enables stereo-specific scoring metrics

## Usage

### Generate Ableton References (GUI automation)

```bash
make ableton-refs
```

This opens each project in Ableton, exports via GUI automation, and validates
the output. Runs sequentially (one track at a time).

### Generate Library Outputs

```bash
make library-refs
```

Runs `timestretch-cli` in both batch and streaming modes.

### Score Against Ableton

```bash
make score-ableton
```

### Full Pipeline

```bash
make full       # ableton-refs + library-refs + score
make compare    # library-refs + score (skip Ableton rendering)
```

### Dry Run

```bash
make manifest   # Print what would be rendered
```

## Directory Layout

```
optimize/ableton/
├── track_catalog.json        # Input track definitions
├── template/
│   └── base_template.als     # Manually created (gitignored)
├── scripts/
│   ├── create_als_project.py # ALS XML manipulation
│   ├── export_dialog.applescript
│   ├── render_via_ableton.py # Single-track renderer
│   ├── verify_render.py      # Output validation
│   ├── batch_generate_refs.py
│   └── generate_library_refs.py
├── refs/
│   ├── ableton/              # Ableton Complex Pro outputs (gitignored)
│   └── library/              # timestretch-rs outputs (gitignored)
└── logs/                     # Render logs (gitignored)
```

## Scoring Integration

The existing `optimize/scripts/score.py` supports `--ref-source ableton`:

```bash
cd optimize
python scripts/score.py --ref-source ableton \
  --ableton-manifest ableton/ableton_manifest.json \
  --batch ableton/logs/scores.json
```

This uses the same metrics and weights as rubberband scoring but compares
against Ableton Complex Pro output instead.

## Troubleshooting

### AppleScript fails / no export
- Ensure Accessibility permissions are granted
- Increase delays in `export_dialog.applescript` for slower machines
- Try running the export manually first: `Cmd+Shift+R` in Ableton

### Wrong warp mode
Inspect a generated project:
```bash
python scripts/create_als_project.py inspect generated_project.als
```

### Duration mismatch
Validate an output WAV:
```bash
python scripts/verify_render.py --info output.wav
```
