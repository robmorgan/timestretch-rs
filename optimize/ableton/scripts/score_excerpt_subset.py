#!/usr/bin/env python3
"""
Fast excerpt-based Ableton scoring for iterative streaming quality tuning.

Pipeline per case:
1) Trim a short source excerpt from the original track.
2) Render library output for that excerpt (batch or streaming).
3) Trim the Ableton reference at the time-mapped output region.
4) Score library excerpt against Ableton excerpt with optimize/scripts/score.py --pair.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ABLETON_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = ABLETON_DIR.parent.parent


@dataclass(frozen=True)
class CaseKey:
    track_id: str
    target_bpm: float


def run(cmd: list[str], desc: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{desc} failed\ncommand: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def format_bpm(value: float) -> str:
    return f"{value:.1f}"


def parse_case(text: str) -> CaseKey:
    if "@" not in text:
        raise ValueError(f"invalid case '{text}': expected format track-id@bpm")
    track, bpm_text = text.split("@", 1)
    return CaseKey(track_id=track.strip(), target_bpm=float(bpm_text.strip()))


def find_entry(manifest: list[dict[str, Any]], key: CaseKey) -> dict[str, Any]:
    for entry in manifest:
        if (
            entry.get("track_id") == key.track_id
            and abs(float(entry.get("target_bpm", 0.0)) - key.target_bpm) < 1e-6
        ):
            return entry
    raise KeyError(f"case not found in manifest: {key.track_id}@{format_bpm(key.target_bpm)}")


def resolve_ref_path(manifest_path: Path, entry: dict[str, Any]) -> Path:
    ref_path = Path(entry["ref_path"])
    if ref_path.is_absolute():
        return ref_path
    return manifest_path.parent / ref_path


def ffmpeg_trim(input_path: Path, output_path: Path, start_sec: float, duration_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-i",
        str(input_path),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    run(cmd, f"ffmpeg trim {input_path.name}")


def render_library_excerpt(
    cli_path: Path,
    source_excerpt: Path,
    out_path: Path,
    ratio: float,
    mode: str,
    chunk_size: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli_path),
        str(source_excerpt),
        str(out_path),
        "--ratio",
        f"{ratio:.6f}",
        "--no-normalize",
    ]
    if mode == "streaming":
        cmd.extend(["--streaming", "--chunk-size", str(chunk_size)])
    run(cmd, f"render {out_path.name}")


def score_pair(
    score_script: Path, ref_excerpt: Path, test_excerpt: Path, align_ms: float = 0.0
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(score_script),
        "--pair",
        str(ref_excerpt),
        str(test_excerpt),
    ]
    if align_ms > 0.0:
        cmd.extend(["--align-ms", f"{align_ms:.3f}"])
    proc = run(cmd, "score pair")
    return json.loads(proc.stdout)


def load_baseline_scores(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        rows = json.load(f)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        desc = row.get("description")
        if isinstance(desc, str):
            result[desc] = row
    return result


def summarize_series(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    vmin = float(min(values))
    vmax = float(max(values))
    return {
        "count": int(len(values)),
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "range": float(vmax - vmin),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Ableton subset using short excerpts")
    parser.add_argument(
        "--manifest",
        default=str(ABLETON_DIR / "ableton_manifest.json"),
        help="Path to ableton_manifest.json",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["hot-stuff@100.0", "cold-heart@128.0"],
        help="Cases as track-id@bpm",
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Library render mode",
    )
    parser.add_argument("--start-sec", type=float, default=30.0, help="Source excerpt start (sec)")
    parser.add_argument("--excerpt-sec", type=float, default=45.0, help="Source excerpt duration (sec)")
    parser.add_argument(
        "--preroll-sec",
        type=float,
        default=20.0,
        help="Extra source context rendered before the scored excerpt (sec)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Streaming chunk size (frames)")
    parser.add_argument(
        "--work-dir",
        default=str(ABLETON_DIR / "tmp_excerpt"),
        help="Workspace for generated excerpt files",
    )
    parser.add_argument(
        "--baseline-scores",
        default=str(ABLETON_DIR / "logs" / "scores_verify.json"),
        help="Optional baseline score JSON for delta reporting",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--align-ms",
        type=float,
        default=0.0,
        help="Optional max lag (ms) for alignment-compensated diagnostic scoring",
    )
    parser.add_argument(
        "--reruns",
        type=int,
        default=1,
        help="Number of independent reruns per case for lag/score stability metrics",
    )
    parser.add_argument(
        "--cli-path",
        default=str(REPO_ROOT / "target" / "release" / "timestretch-cli"),
        help="Path to timestretch-cli binary",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    cli_path = Path(args.cli_path).resolve()
    score_script = (REPO_ROOT / "optimize" / "scripts" / "score.py").resolve()
    work_dir = Path(args.work_dir).resolve()
    baseline_map = load_baseline_scores(Path(args.baseline_scores).resolve())

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not cli_path.exists():
        raise FileNotFoundError(f"cli not found: {cli_path}")
    if not score_script.exists():
        raise FileNotFoundError(f"score script not found: {score_script}")
    if args.reruns < 1:
        raise ValueError("--reruns must be >= 1")

    with manifest_path.open("r") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise RuntimeError(f"unexpected manifest format: {manifest_path}")

    started = time.time()
    results: list[dict[str, Any]] = []

    for case_text in args.cases:
        key = parse_case(case_text)
        entry = find_entry(manifest, key)
        ratio = float(entry["ratio"])

        source_path = Path(entry["source"]).resolve()
        ref_full_path = resolve_ref_path(manifest_path, entry).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"source not found: {source_path}")
        if not ref_full_path.exists():
            raise FileNotFoundError(f"reference not found: {ref_full_path}")

        label = f"{entry['track_id']}@{format_bpm(float(entry['target_bpm']))}"
        safe_label = f"{entry['track_id']}_{format_bpm(float(entry['target_bpm']))}bpm_{args.mode}"

        source_excerpt = work_dir / "source" / f"{safe_label}_src.wav"
        ref_excerpt = work_dir / "ref" / f"{safe_label}_ref.wav"
        test_render_first: Path | None = None
        test_excerpt_first: Path | None = None

        source_start = args.start_sec
        source_duration = args.excerpt_sec
        source_trim_start = max(0.0, source_start - args.preroll_sec)
        actual_preroll = source_start - source_trim_start
        source_trim_duration = actual_preroll + source_duration

        output_start = source_start * ratio
        output_duration = max(1.0, source_duration * ratio)
        output_start_in_render = actual_preroll * ratio

        ffmpeg_trim(source_path, source_excerpt, source_trim_start, source_trim_duration)
        ffmpeg_trim(ref_full_path, ref_excerpt, output_start, output_duration)
        score: dict[str, Any] | None = None
        score_aligned: dict[str, Any] | None = None
        run_metrics: list[dict[str, Any]] = []
        for run_idx in range(args.reruns):
            run_suffix = "" if args.reruns == 1 else f"_r{run_idx + 1}"
            test_render = work_dir / "library" / f"{safe_label}{run_suffix}_render.wav"
            test_excerpt = work_dir / "library" / f"{safe_label}{run_suffix}_test.wav"

            render_library_excerpt(
                cli_path, source_excerpt, test_render, ratio, args.mode, args.chunk_size
            )
            ffmpeg_trim(test_render, test_excerpt, output_start_in_render, output_duration)

            run_score = score_pair(score_script, ref_excerpt, test_excerpt)
            run_score_aligned = None
            if args.align_ms > 0.0:
                run_score_aligned = score_pair(
                    score_script, ref_excerpt, test_excerpt, align_ms=args.align_ms
                )

            if score is None:
                score = run_score
                score_aligned = run_score_aligned
                test_render_first = test_render
                test_excerpt_first = test_excerpt

            run_metrics.append(
                {
                    "run": run_idx + 1,
                    "test_render": str(test_render),
                    "test_excerpt": str(test_excerpt),
                    "raw_total_score": float(run_score["total_score"]),
                    "raw_transient_preservation": float(
                        run_score["metrics"]["transient_preservation"]
                    ),
                    "aligned_total_score": None
                    if run_score_aligned is None
                    else float(run_score_aligned["total_score"]),
                    "aligned_transient_preservation": None
                    if run_score_aligned is None
                    else float(run_score_aligned["metrics"]["transient_preservation"]),
                    "aligned_lag_ms": None
                    if run_score_aligned is None
                    else float(run_score_aligned["alignment"]["lag_ms"]),
                }
            )

        if score is None or test_render_first is None or test_excerpt_first is None:
            raise RuntimeError(f"failed to generate score for case: {label}")

        raw_total_values = [r["raw_total_score"] for r in run_metrics]
        raw_tp_values = [r["raw_transient_preservation"] for r in run_metrics]
        aligned_total_values = [
            r["aligned_total_score"]
            for r in run_metrics
            if r["aligned_total_score"] is not None
        ]
        aligned_tp_values = [
            r["aligned_transient_preservation"]
            for r in run_metrics
            if r["aligned_transient_preservation"] is not None
        ]
        lag_values = [r["aligned_lag_ms"] for r in run_metrics if r["aligned_lag_ms"] is not None]
        score_stability = {
            "raw_total_score": summarize_series(raw_total_values),
            "raw_transient_preservation": summarize_series(raw_tp_values),
            "aligned_total_score": summarize_series(aligned_total_values),
            "aligned_transient_preservation": summarize_series(aligned_tp_values),
        }
        lag_stability = summarize_series(lag_values)

        desc = f"{entry['track_id']} @ {format_bpm(float(entry['target_bpm']))} BPM"
        if args.mode == "streaming":
            desc = f"{desc} [streaming]"
        baseline = baseline_map.get(desc)
        baseline_total = baseline.get("total_score") if baseline else None
        baseline_tp = baseline.get("metrics", {}).get("transient_preservation") if baseline else None

        row = {
            "case": label,
            "description": desc,
            "mode": args.mode,
            "ratio": ratio,
            "excerpt": {
                "source_start_sec": source_start,
                "source_duration_sec": source_duration,
                "source_trim_start_sec": source_trim_start,
                "source_trim_duration_sec": source_trim_duration,
                "actual_preroll_sec": actual_preroll,
                "output_start_sec": output_start,
                "output_duration_sec": output_duration,
                "output_start_in_render_sec": output_start_in_render,
            },
            "paths": {
                "source_excerpt": str(source_excerpt),
                "ref_excerpt": str(ref_excerpt),
                "test_render": str(test_render_first),
                "test_excerpt": str(test_excerpt_first),
            },
            "score": score,
            "score_aligned": score_aligned,
            "reruns": args.reruns,
            "run_metrics": run_metrics,
            "score_stability": score_stability,
            "lag_stability": lag_stability,
            "baseline_total_score": baseline_total,
            "baseline_transient_preservation": baseline_tp,
            "delta_total_score": None if baseline_total is None else score["total_score"] - baseline_total,
            "delta_transient_preservation": None
            if baseline_tp is None
            else score["metrics"]["transient_preservation"] - baseline_tp,
        }
        results.append(row)

        delta_total = row["delta_total_score"]
        delta_tp = row["delta_transient_preservation"]
        aligned_suffix = ""
        if score_aligned is not None:
            aligned_suffix = (
                f" aligned_total={score_aligned['total_score']:.2f}"
                f" aligned_tp={score_aligned['metrics']['transient_preservation']:.2f}"
                f" lag_ms={score_aligned['alignment']['lag_ms']:+.2f}"
            )
            if lag_stability is not None and args.reruns > 1:
                aligned_suffix += (
                    f" lag_std={lag_stability['std']:.2f}"
                    f" lag_range={lag_stability['range']:.2f}"
                )
        stability_suffix = ""
        if args.reruns > 1:
            raw_total_std = score_stability["raw_total_score"]["std"]
            raw_tp_std = score_stability["raw_transient_preservation"]["std"]
            stability_suffix = (
                f" runs={args.reruns}"
                f" raw_total_std={raw_total_std:.3f}"
                f" raw_tp_std={raw_tp_std:.2f}"
            )
        print(
            f"{desc}: total={score['total_score']:.2f}"
            f" tp={score['metrics']['transient_preservation']:.2f}"
            + (
                ""
                if delta_total is None
                else f" (delta_total={delta_total:+.2f}, delta_tp={delta_tp:+.2f})"
            )
            + aligned_suffix
            + stability_suffix
        )

    elapsed = time.time() - started
    summary = {
        "mode": args.mode,
        "cases": len(results),
        "elapsed_sec": elapsed,
        "results": results,
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
