#!/usr/bin/env python3
"""Generate QM Vamp beat-tracker baselines for the BPM corpus.

Runs the qm-barbeattracker plugin (qm-vamp-plugins, GPL) over every
manifest track whose audio is present locally and stores its OUTPUT —
beat and downbeat times — as JSON under `benchmarks/baselines/qm/`.
Only plugin output is stored; qm-dsp source never enters this repo
(it is GPL, this crate is not).

The stored baselines are read by `qa/bpm_accuracy.rs` as an external
reference column: our tracker's beat scores are compared against QM's
on the same hand-corrected annotations (ROADMAP Stage 10 exit
criterion: "no row where the QM baseline wins by more than noise").

Requirements (documented in benchmarks/README.md):
  - `sonic-annotator` on PATH (or $SONIC_ANNOTATOR)
    https://github.com/sonic-visualiser/sonic-annotator/releases
  - qm-vamp-plugins.dylib in ~/Library/Audio/Plug-Ins/Vamp (macOS)
    or the platform Vamp path; build from https://github.com/c4dm

Usage, from the repo root:
  python3 benchmarks/generate_qm_baseline.py [track-id ...]

With no arguments every manifest track with local audio is processed;
missing audio is skipped with a note. Baselines are deterministic for
a given audio file + plugin version, so regeneration is idempotent.
"""

import csv
import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "benchmarks" / "manifest.toml"
AUDIO_BASE = REPO / "benchmarks" / "audio"
OUT_DIR = REPO / "benchmarks" / "baselines" / "qm"
PLUGIN = "vamp:qm-vamp-plugins:qm-barbeattracker"


def find_annotator() -> str:
    exe = os.environ.get("SONIC_ANNOTATOR") or shutil.which("sonic-annotator")
    if not exe:
        sys.exit(
            "sonic-annotator not found on PATH (or $SONIC_ANNOTATOR); "
            "see benchmarks/README.md for install instructions"
        )
    return exe


def annotator_version(exe: str) -> str:
    out = subprocess.run(
        [exe, "--version"], capture_output=True, text=True, check=False
    )
    return (out.stdout + out.stderr).strip().splitlines()[0]


def run_transform(exe: str, output: str, audio: Path) -> list[float]:
    """Runs one plugin output over one file, returns event times (secs)."""
    result = subprocess.run(
        [exe, "-d", f"{PLUGIN}:{output}", "-w", "csv", "--csv-stdout",
         "--csv-force", str(audio)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sonic-annotator failed on {audio.name} ({output}): "
            f"{result.stderr.strip().splitlines()[-1:]}"
        )
    times = []
    for row in csv.reader(io.StringIO(result.stdout)):
        # Rows: [filename-or-empty, time_secs, label...]
        if len(row) >= 2 and row[1]:
            times.append(round(float(row[1]), 6))
    return sorted(times)


def main() -> None:
    exe = find_annotator()
    version = annotator_version(exe)
    with open(MANIFEST, "rb") as f:
        manifest = tomllib.load(f)

    only = set(sys.argv[1:])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0

    for track in manifest.get("track", []):
        track_id = track["id"]
        if only and track_id not in only:
            continue
        audio = AUDIO_BASE / track["original"]
        if not audio.exists():
            print(f"skip {track_id}: audio not present ({audio.name})")
            skipped += 1
            continue

        print(f"analyzing {track_id} ...", flush=True)
        beats = run_transform(exe, "beats", audio)
        downbeats = run_transform(exe, "bars", audio)
        if not beats:
            print(f"skip {track_id}: QM returned no beats")
            skipped += 1
            continue

        out_path = OUT_DIR / f"{track_id}.json"
        payload = {
            "generator": f"{PLUGIN} via {version}",
            "generated": datetime.date.today().isoformat(),
            "note": (
                "External reference output (plugin is GPL; output only). "
                "Regenerate with benchmarks/generate_qm_baseline.py."
            ),
            "beats": beats,
            "downbeats": downbeats,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1)
            f.write("\n")
        print(f"  {len(beats)} beats, {len(downbeats)} downbeats -> {out_path.relative_to(REPO)}")
        written += 1

    print(f"done: {written} written, {skipped} skipped")


if __name__ == "__main__":
    main()
