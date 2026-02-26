#!/bin/bash
set -euo pipefail

# generate_references.sh - Generates reference audio files using CLI tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST_PATH="$REPO_ROOT/test_manifest.json"
CONFIG_PATH="$REPO_ROOT/config.toml"
REF_DIR="$REPO_ROOT/references"

# Function to parse TOML (simple grep/sed for basic needs)
get_config() {
    local key=$1
    grep "^$key" "$CONFIG_PATH" | cut -d'=' -f2 | tr -d ' ",' | xargs
}

# Detection of tools
detect_tool() {
    if command -v rubberband >/dev/null 2>&1; then
        echo "rubberband"
    elif command -v soundstretch >/dev/null 2>&1; then
        echo "soundstretch"
    elif command -v sox >/dev/null 2>&1; then
        echo "sox"
    else
        echo "none"
    fi
}

TOOL=$(detect_tool)

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --tool TOOL          Force a specific tool (rubberband, soundstretch, sox)"
    echo "  --generate-manifest  Create a starter test_manifest.json"
    echo "  --validate           Check all reference files exist and are valid"
    echo "  --help               Show this help"
}

generate_manifest() {
    echo "Generating starter manifest at $MANIFEST_PATH..."
    cat > "$MANIFEST_PATH" <<EOF
[
  {"source": "samples/sine_440.wav", "ratio": 0.5, "description": "sine wave slowdown"},
  {"source": "samples/sine_440.wav", "ratio": 2.0, "description": "sine wave speedup"},
  {"source": "samples/drums_break.wav", "ratio": 0.75, "description": "transient-heavy slowdown"},
  {"source": "samples/drums_break.wav", "ratio": 1.5, "description": "transient-heavy speedup"},
  {"source": "samples/vocal_phrase.wav", "ratio": 0.8, "description": "vocal slowdown"},
  {"source": "samples/vocal_phrase.wav", "ratio": 1.25, "description": "vocal speedup"},
  {"source": "samples/full_mix.wav", "ratio": 0.9, "description": "complex mix slight stretch"},
  {"source": "samples/silence_transient.wav", "ratio": 1.0, "description": "silence to transient edge case"}
]
EOF
    echo "Done. Please ensure these sample files exist in optimize/samples/"
}

validate_refs() {
    echo "Validating reference files..."
    if [ ! -f "$MANIFEST_PATH" ]; then
        echo "Error: Manifest not found at $MANIFEST_PATH"
        exit 1
    fi

    # Use python to parse json manifest
    python3 -c "import json, os, subprocess;
manifest = json.load(open('$MANIFEST_PATH'))
for item in manifest:
    ref_name = os.path.basename(item['source']).replace('.wav', '') + f'_ref_{item["ratio"]}.wav'
    ref_path = os.path.join('$REF_DIR', ref_name)
    if not os.path.exists(ref_path):
        print(f'Missing: {ref_path}')
        continue
    # Check wav header
    try:
        subprocess.check_output(['soxi', ref_path] if subprocess.run(['command', '-v', 'soxi'], capture_output=True).returncode == 0 else ['ffprobe', ref_path], stderr=subprocess.STDOUT)
        print(f'Valid: {ref_path}')
    except:
        print(f'Invalid WAV: {ref_path}')
"
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --tool) TOOL="$2"; shift 2 ;;
        --generate-manifest) generate_manifest; exit 0 ;;
        --validate) validate_refs; exit 0 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ "$TOOL" == "none" ]; then
    echo "Error: No reference tools found (rubberband, soundstretch, or sox)."
    exit 1
fi

echo "Selected tool: $TOOL"
if [ "$TOOL" == "rubberband" ]; then
    rubberband --version | head -n 1
elif [ "$TOOL" == "soundstretch" ]; then
    soundstretch -help 2>&1 | head -n 1
fi

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "Manifest not found. Use --generate-manifest to create one."
    exit 1
fi

# Run generation
mkdir -p "$REF_DIR"
python3 -c "import json, os, subprocess;
manifest = json.load(open('$MANIFEST_PATH'))
tool = '$TOOL'
repo_root = '$REPO_ROOT'
ref_dir = '$REF_DIR'

for item in manifest:
    source = os.path.join(repo_root, item['source'])
    ratio = item['ratio']
    ref_name = os.path.basename(item['source']).replace('.wav', '') + f'_ref_{ratio}.wav'
    output = os.path.join(ref_dir, ref_name)
    
    if not os.path.exists(source):
        print(f'Warning: Source file not found: {source}')
        continue

    print(f'Generating reference for {item["description"]} (ratio: {ratio})...')
    
    if tool == 'rubberband':
        # High quality
        cmd = ['rubberband', '-3', '-F', '--pitch-hq', '-t', str(ratio), source, output]
        subprocess.run(cmd, check=True)
        # Medium quality if requested (implied by requirements)
        output_med = output.replace('.wav', '_med.wav')
        cmd_med = ['rubberband', '-2', '-t', str(ratio), source, output_med]
        subprocess.run(cmd_med, check=True)
    elif tool == 'soundstretch':
        # soundstretch takes percentage change: (ratio - 1) * 100
        tempo = (ratio - 1) * 100
        cmd = ['soundstretch', source, output, f'-tempo={tempo}']
        subprocess.run(cmd, check=True)
    elif tool == 'sox':
        cmd = ['sox', source, output, 'stretch', str(ratio)]
        subprocess.run(cmd, check=True)
"
