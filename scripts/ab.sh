#!/usr/bin/env bash
# Blind A/B listening harness for algorithm changes.
#
# The house quality gate is the owner's ears; this makes the loop cheap:
#
#   scripts/ab.sh render <name> --base <git-ref> \
#       --rates 0.92,1.08 [--rb] <wav[:start_secs]>...
#       Renders <wav> excerpts through the CURRENT working tree and
#       through <git-ref> (built in a temporary worktree), optionally a
#       Rubber Band reference arm, level-matches all arms per condition
#       (RMS to source + common no-clip trim), and shuffles them into
#       target/ab/<name>/blind/<track>/<rate>/arm_{A,B,...}.wav with a
#       sealed key. Listen, note verdicts per letter, then:
#
#   scripts/ab.sh unblind <name>
#       Prints the letter->arm key. (Don't peek before your notes are
#       written down — the whole point is that you can't.)
#
# Example:
#   scripts/ab.sh render cadence-tweak --base main --rates 0.92,1.08 \
#       "benchmarks/audio/bpm-corpus/12247392_Music Sounds Better With You_(Original Mix).wav:90"
set -euo pipefail
cd "$(dirname "$0")/.."

cmd=${1:?usage: ab.sh render|unblind <name> ...}
name=${2:?a comparison name is required}
out="target/ab/$name"

if [ "$cmd" = "unblind" ]; then
    python3 - "$out/BLIND_KEY.json" <<'EOF'
import json, sys
key = json.load(open(sys.argv[1]))
for cond, arms in key.items():
    print(f"{cond}: " + "  ".join(f"{l}={a}" for l, a in sorted(arms.items())))
EOF
    exit 0
fi
[ "$cmd" = "render" ] || { echo "unknown command: $cmd" >&2; exit 2; }
shift 2

base_ref=""
rates="0.92,1.08"
want_rb=0
tracks=()
while [ $# -gt 0 ]; do
    case "$1" in
        --base) base_ref=$2; shift 2 ;;
        --rates) rates=$2; shift 2 ;;
        --rb) want_rb=1; shift ;;
        *) tracks+=("$1"); shift ;;
    esac
done
[ ${#tracks[@]} -gt 0 ] || { echo "at least one wav path required" >&2; exit 2; }

rm -rf "$out"
mkdir -p "$out/raw"

echo "== rendering arm 'current' (working tree) =="
cargo run --release --example ab_render -- current "$out/raw" "$rates" "${tracks[@]}"

if [ -n "$base_ref" ]; then
    echo "== rendering arm 'base' ($base_ref) =="
    abs_out="$(pwd)/$out/raw"
    abs_tracks=()
    for t in "${tracks[@]}"; do
        case "$t" in
            /*) abs_tracks+=("$t") ;;
            *) abs_tracks+=("$(pwd)/$t") ;;
        esac
    done
    wt=$(mktemp -d)/ab-base
    git worktree add --detach -q "$wt" "$base_ref"
    # The baseline may predate ab_render; carry it in.
    cp examples/ab_render.rs "$wt/examples/ab_render.rs"
    (cd "$wt" && cargo run --release --example ab_render -- base "$abs_out" "$rates" "${abs_tracks[@]}")
    git worktree remove --force "$wt"
fi

if [ "$want_rb" = 1 ]; then
    echo "== rendering arm 'rubberband' =="
    cli=$(command -v rubberband-r3 || command -v rubberband || true)
    [ -n "$cli" ] || { echo "rubberband CLI not found" >&2; exit 1; }
    fine=""; [ "$(basename "$cli")" = "rubberband" ] && fine="--fine"
    for src in "$out"/raw/*/*/source.wav; do
        dir=$(dirname "$src")
        rate_tag=$(basename "$dir")
        pct=${rate_tag/pct/}
        ratio=$(python3 -c "print(1.0/(1.0+($pct)/100.0))")
        "$cli" $fine --time "$ratio" "$src" "$dir/rubberband.wav" >/dev/null 2>&1
    done
fi

echo "== level-matching and blinding =="
python3 - "$out" <<'EOF'
import struct, pathlib, math, json, random, hashlib, sys

def read_wav(p):
    d = pathlib.Path(p).read_bytes()
    fmt = d.find(b'fmt '); ch = struct.unpack('<H', d[fmt+10:fmt+12])[0]
    sr = struct.unpack('<I', d[fmt+12:fmt+16])[0]
    bits = struct.unpack('<H', d[fmt+22:fmt+24])[0]
    afmt = struct.unpack('<H', d[fmt+8:fmt+10])[0]
    i = d.find(b'data'); n = struct.unpack('<I', d[i+4:i+8])[0]
    raw = d[i+8:i+8+n]
    if afmt == 3 or bits == 32:
        x = list(struct.unpack(f'<{n//4}f', raw))
    elif bits == 16:
        x = [v/32768.0 for v in struct.unpack(f'<{n//2}h', raw)]
    else:
        raise ValueError(f"unsupported wav: fmt {afmt} bits {bits}")
    return x, sr, ch

def write_wav(p, x, sr, ch):
    data = struct.pack(f'<{len(x)}f', *x)
    hdr = (b'RIFF' + struct.pack('<I', 36+len(data)) + b'WAVEfmt '
           + struct.pack('<IHHIIHH', 16, 3, ch, sr, sr*ch*4, ch*4, 32)
           + b'data' + struct.pack('<I', len(data)))
    pathlib.Path(p).write_bytes(hdr + data)

def rms(x): return math.sqrt(sum(v*v for v in x)/max(len(x),1))

out = pathlib.Path(sys.argv[1])
raw = out/"raw"
key = {}
for cond in sorted(raw.glob("*/*")):
    if not cond.is_dir(): continue
    track, rate = cond.parent.name, cond.name
    src, sr, ch = read_wav(cond/"source.wav")
    target = rms(src)
    arms = {}
    for f in sorted(cond.glob("*.wav")):
        if f.stem == "source": continue
        x, s2, c2 = read_wav(f)
        g = target/max(rms(x), 1e-12)
        arms[f.stem] = [v*g for v in x]
    if not arms: continue
    n = min(len(x) for x in arms.values())
    for a in arms: arms[a] = arms[a][:n]
    peak = max(max(abs(v) for v in x) for x in arms.values())
    trim = 0.98/peak if peak > 0.98 else 1.0
    blind = out/"blind"/track/rate
    blind.mkdir(parents=True, exist_ok=True)
    rng = random.Random(hashlib.sha256(f"{out.name}/{track}/{rate}".encode()).hexdigest())
    order = sorted(arms); rng.shuffle(order)
    letters = [chr(ord('A')+i) for i in range(len(order))]
    key[f"{track}/{rate}"] = dict(zip(letters, order))
    for letter, arm in zip(letters, order):
        write_wav(blind/f"arm_{letter}.wav", [v*trim for v in arms[arm]], sr, ch)
    write_wav(blind/"source.wav", [v*trim for v in src], sr, ch)
(out/"BLIND_KEY.json").write_text(json.dumps(key, indent=1))
print(f"blind set: {len(key)} conditions -> {out}/blind")
print(f"key sealed in {out}/BLIND_KEY.json — run scripts/ab.sh unblind after noting verdicts")
EOF
