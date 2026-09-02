#!/usr/bin/env python3
"""Render the Elastique reference matrix headlessly through REAPER.

REAPER ships zplane's élastique 3.3.3 Pro — the Elastique Pro V3 engine —
as a pitch-shift mode, and runs ReaScripts from the command line. This
wrapper turns the manifest into a job list for scripts/reaper_elastique.lua,
launches REAPER, waits for it, verifies each output, and prints the
`[[track.reference]]` rows (with SHA-256) to paste into the manifest.

Output layout (the convention scripts/ab.sh --ref-arm and the benchmarks
README use):

    <out>/<track_stem>/<rate_tag>.wav        e.g. 12247392_MusicSoundsBett/+8pct.wav

where <track_stem> is the source stem restricted to [A-Za-z0-9_-] and cut
to 24 characters, and <rate_tag> is the signed integer percent. Sources
that are not WAV (the public corpus is MP3) also get a `+0pct.wav`, a
plain decode at unity, because the reference-quality harness and
ab_render only read WAV.

Usage:
    scripts/render_elastique.py [--tracks id,id,...] [--rates -8,-4,4,8,-30,-50,30,50]
                                [--out benchmarks/audio/references/elastique]
                                [--force] [--dry-run] [--manifest-out rows.toml]
"""

import argparse
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
REAPER = pathlib.Path("/Applications/REAPER.app/Contents/MacOS/REAPER")
SCRIPT = REPO / "scripts" / "reaper_elastique.lua"
DEFAULT_RATES = "-8,-4,4,8,-30,-50,30,50"
SOFTWARE = "REAPER"
ALGORITHM = "élastique 3.3.3 Pro (Normal)"


def stem_of(path: pathlib.Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", path.stem)[:24]


def rate_tag(pct: int) -> str:
    return f"{'+' if pct >= 0 else '-'}{abs(pct)}pct"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lua_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "benchmarks" / "manifest.toml"))
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "audio" / "references" / "elastique"))
    ap.add_argument("--tracks", help="comma-separated manifest track ids (default: every track whose source exists)")
    ap.add_argument("--rates", default=DEFAULT_RATES, help="signed tempo percents")
    ap.add_argument("--force", action="store_true", help="re-render existing outputs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest-out", help="write the [[track.reference]] rows here instead of stdout")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait for REAPER")
    args = ap.parse_args()

    manifest = tomllib.loads(pathlib.Path(args.manifest).read_bytes().decode())
    audio_base = pathlib.Path(args.manifest).resolve().parent / "audio"
    out_base = pathlib.Path(args.out).resolve()
    wanted = set(args.tracks.split(",")) if args.tracks else None
    pcts = [int(p) for p in args.rates.split(",") if p.strip()]

    jobs = []  # (track, pct, src, out_path)
    skipped = []
    for track in manifest.get("track", []):
        if wanted and track["id"] not in wanted:
            continue
        src = (audio_base / track["original"]).resolve()
        if not src.exists():
            skipped.append(f"{track['id']}: source missing ({src})")
            continue
        stem = stem_of(src)
        rates = list(pcts)
        if src.suffix.lower() != ".wav" and 0 not in rates:
            rates.append(0)
        for pct in rates:
            out_path = out_base / stem / f"{rate_tag(pct)}.wav"
            if out_path.exists() and not args.force:
                continue
            jobs.append((track, pct, src, out_path))

    for s in skipped:
        print(f"skip {s}", file=sys.stderr)
    print(f"{len(jobs)} render(s) to do", file=sys.stderr)
    for track, pct, src, out_path in jobs:
        print(f"  {track['id']} {rate_tag(pct)} -> {out_path.relative_to(out_base)}", file=sys.stderr)

    if jobs and not args.dry_run:
        if not REAPER.exists():
            print(f"REAPER not found at {REAPER}", file=sys.stderr)
            return 1
        work = out_base / ".reaper"
        work.mkdir(parents=True, exist_ok=True)
        jobs_path = work / f"jobs-{int(time.time())}.lua"
        for _, _, _, out_path in jobs:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["return {"]
        for track, pct, src, out_path in jobs:
            lines.append(
                "  { src = %s, out_dir = %s, out_name = %s, rate = %.6f },"
                % (lua_str(str(src)), lua_str(str(out_path.parent)), lua_str(out_path.stem), 1.0 + pct / 100.0)
            )
        lines.append("}")
        jobs_path.write_text("\n".join(lines) + "\n")
        done_path = jobs_path.with_suffix(".lua.done")
        log_path = jobs_path.with_suffix(".lua.log")

        env = dict(os.environ, TIMESTRETCH_REAPER_JOBS=str(jobs_path))
        print(f"launching REAPER for {len(jobs)} job(s); log: {log_path}", file=sys.stderr)
        proc = subprocess.Popen(
            [str(REAPER), "-nosplash", "-ignoreerrors", str(SCRIPT)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # The script writes `.done` after the last render and then asks
        # REAPER to quit; the quit is not always honoured headless, so the
        # marker is the completion signal and the process gets a short grace
        # period before it is killed.
        deadline = time.time() + args.timeout
        while proc.poll() is None and time.time() < deadline and not done_path.exists():
            time.sleep(1.0)
        if proc.poll() is None:
            grace = time.time() + 10.0
            while proc.poll() is None and time.time() < grace:
                time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
            if not done_path.exists():
                print("REAPER timed out; killed", file=sys.stderr)
        if log_path.exists():
            print(log_path.read_text(), file=sys.stderr, end="")
        if not done_path.exists():
            print("REAPER did not finish the job list (no .done marker)", file=sys.stderr)
            return 1
        n_ok, n_fail = (int(x) for x in done_path.read_text().split())
        if n_fail:
            print(f"{n_fail} render(s) failed", file=sys.stderr)
            return 1
        shutil.rmtree(work, ignore_errors=True)

    # Manifest rows for everything present on disk (rendered now or before).
    rows = []
    for track in manifest.get("track", []):
        if wanted and track["id"] not in wanted:
            continue
        src = (audio_base / track["original"]).resolve()
        stem = stem_of(src)
        track_dir = out_base / stem
        if not track_dir.is_dir():
            continue
        is_wav = src.suffix.lower() == ".wav"
        rows.append(f"# {track['id']}" + ("" if is_wav else "  (MP3 source: reference_quality reads WAV originals only; use +0pct.wav)"))
        for f in sorted(track_dir.glob("*pct.wav"), key=lambda p: int(p.stem.replace("pct", ""))):
            pct = int(f.stem.replace("pct", ""))
            if pct == 0:
                continue
            try:
                file_field = f.resolve().relative_to(audio_base).as_posix()
            except ValueError:
                file_field = str(f.resolve())  # outside benchmarks/audio: not manifest-loadable as is
            rows.append("  [[track.reference]]")
            rows.append(f'  file = "{file_field}"')
            rows.append(f'  file_sha256 = "{sha256(f)}"')
            rows.append(f"  target_bpm = {track['bpm'] * (1.0 + pct / 100.0):.4f}")
            rows.append(f'  software = "{SOFTWARE}"')
            rows.append(f'  algorithm = "{ALGORITHM}"')
            rows.append("")
    text = "\n".join(rows) + "\n"
    if args.manifest_out:
        pathlib.Path(args.manifest_out).write_text(text)
        print(f"manifest rows written to {args.manifest_out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
