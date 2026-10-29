#!/usr/bin/env python3
"""Shared fixed benchmark protocol helpers; CLI runs current session qualification.

Helpers operate on repository-generated signals and current providers.
"""
import argparse, csv, hashlib, html, itertools, json, math, os, platform, random, shutil, subprocess, sys
import statistics

sys.dont_write_bytecode = True
from pathlib import Path
from audio_files import wav_read, wav_write

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["rust", "fft", "math", "apple"]


def executable_name(name):
    return name + (".exe" if platform.system() == "Windows" else "")


def run(args, **kw):
    return subprocess.run([str(x) for x in args], check=True, **kw)

def command(args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()

def read_result(args):
    return json.loads(command(args))

def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def capture(out):
    files = command(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    ).splitlines()
    hashes = {p: digest(Path(p)) for p in files if Path(p).is_file()}
    return {"git_head": command(["git", "rev-parse", "HEAD"]), "files": hashes}

def metadata():
    return dict(platform=platform.platform(), rustc=command(['rustc','-Vv']),
                rustflags=os.environ.get('RUSTFLAGS',''), profile='release opt-level=3 lto=true',
                seed=20261028)

def perf_row(row):
    t = sorted(row["times_ns"])
    n = len(t)
    pct = lambda p: t[round((n - 1) * p)]
    budget = row["callback"] / row["sample_rate"] * 1e9
    row.update(
        budget_ns=budget,
        median_ns=pct(0.5),
        p95_ns=pct(0.95),
        p99_ns=pct(0.99),
        p999_ns=pct(0.999) if n >= 10000 else None,
        max_ns=max(t),
        deadline_misses=sum(x > budget for x in t),
        frames_per_second=n * row["callback"] / (sum(t) / 1e9),
    )
    return row

def save_performance(out, rows):
    write(out / "performance.json", rows)
    flat = [{k: v for k, v in r.items() if k != "times_ns"} for r in rows]
    with (out / "performance.csv").open("w") as f:
        w = csv.DictWriter(f, sorted(set().union(*(r.keys() for r in flat))))
        w.writeheader()
        w.writerows(flat)

def synthetic(path, rate, kind, ch=2):
    n = rate * 2
    rng = random.Random(1701)
    data = []
    for i in range(n):
        t = i / rate
        if kind == "silence":
            x = 0.0
        elif kind == "impulses":
            x = 0.8 if i in [rate // 4, rate // 2, rate, 3 * rate // 2, n - 64] else 0.0
        elif kind == "noise":
            x = rng.uniform(-0.3, 0.3)
        elif kind == "multitone":
            x = sum(0.1 * math.sin(2 * math.pi * f * t) for f in [55, 440, 3000, 9000])
        else:
            x = 0.4 * math.sin(2 * math.pi * 440 * t)
        data.append(x)
        if ch == 2:
            data.append(
                -x
                if kind == "antiphase"
                else 0.7 * x
                if kind != "stereo"
                else 0.3 * math.sin(2 * math.pi * 660 * t)
            )
    wav_write(path, ch, rate, data)

def first_difference(left, right):
    # Ordered, exact scheduler-boundary observations. Hashes identify a stage,
    # not a within-stage instruction or a causal explanation.
    from itertools import zip_longest

    first = {}
    with left.open() as left_file, right.open() as right_file:
        for i, (a, b) in enumerate(zip_longest(left_file, right_file)):
            if a is None or b is None:
                first.setdefault("trace_length", {"step": i, "stage": "trace_length"})
                break
            a = json.loads(a)
            b = json.loads(b)
            # Later oracle revisions record the synthesis/phase histories too.
            # Compare every recorded stage, including output hashes; matching
            # analysis alone must not be reported as an exact complete trace.
            for record in (a,b):
                record.update({f'stages.{name}':value for name,value in record.get('stages',{}).items()})
            for key in [
                "consumed",
                "output",
                "frame",
                "marks",
                "spectrum_hash",
                "magnitude_hash",
                "classifier",
                "splitter_window",
                "stages.main_history",
                "stages.aux_history",
                "stages.input_phase",
                "stages.stored",
                "stages.output_phase",
                "stages.output_ola",
                "stages.aux_ola",
                "output_hash",
            ]:
                if key in first or a.get(key) == b.get(key):
                    continue
                result = {"step": i, "stage": key}
                if isinstance(a.get(key), list) and isinstance(b.get(key), list):
                    for j, (x, y) in enumerate(zip_longest(a[key], b[key])):
                        if x != y:
                            result.update(index=j, reference=x, candidate=y)
                            break
                else:
                    result.update(reference=a.get(key), candidate=b.get(key))
                first[key] = result
            # A mark's floating scores can differ before its selected position
            # does. Keep the first actual position change separately.
            if 'mark_positions' not in first:
                positions_a=[m[0] for m in a['marks']]
                positions_b=[m[0] for m in b['marks']]
                for j,(x,y) in enumerate(zip_longest(positions_a,positions_b)):
                    if x != y:
                        first['mark_positions']={'step':i,'stage':'mark_positions','index':j,'reference':x,'candidate':y}
                        break
    if not first:
        return None
    result = min(first.values(), key=lambda r: r["step"]).copy()
    result["first_by_stage"] = first
    return result

if __name__ == '__main__':
    from qualify import main
    main()
