#!/bin/bash
set -euo pipefail

MODE="start"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROADMAP_PATH="$REPO_ROOT/ROADMAP.md"
MODEL="${MODEL:-gpt-5.4}"
ITERATIONS="${ITERATIONS:-5}"
STATE_DIR_INPUT="${STATE_DIR:-.codex-loop}"
TEST_CMD="${TEST_CMD:-cargo test --test dual_plane_rt --test realtime_allocations --test realtime_dj_conditions}"
SMOKE_CMD="$TEST_CMD"
PROMPT_SCOPE="full"
FAILURE_POLICY="stop"
STALL_LIMIT=0
STAGE_TESTS_MODE=""
STATUS_HEADINGS=0

if [[ $# -gt 0 ]]; then
    case "$1" in
        start|resume|status)
            MODE="$1"
            shift
            ;;
        --help|-h|help)
            MODE="help"
            shift
            ;;
    esac
fi

if [[ "$STATE_DIR_INPUT" = /* ]]; then
    STATE_DIR="$STATE_DIR_INPUT"
else
    STATE_DIR="$REPO_ROOT/$STATE_DIR_INPUT"
fi

STATE_DIR_PREFIX=""
case "$STATE_DIR" in
    "$REPO_ROOT"/*)
        STATE_DIR_PREFIX="${STATE_DIR#$REPO_ROOT/}"
        ;;
esac

ITERATIONS_DIR="$STATE_DIR/iterations"
STATE_FILE="$STATE_DIR/state.json"
MANAGED_UNTRACKED_FILE="$STATE_DIR/managed_untracked.txt"

usage() {
    cat <<'EOF'
Usage: scripts/roadmap_loop.sh [start|resume|status] [options]

Modes:
  start   Create a new bounded roadmap execution session
  resume  Continue a prior session from state.json
  status  Print current roadmap/session status

Options:
  --prompt-scope <full|active-stage>
      full         Include the entire roadmap in the prompt (legacy behavior)
      active-stage Include Goal, Principles, and the current active stage only

  --failure-policy <stop|continue>
      stop      Stop immediately on agent or test failure (legacy behavior)
      continue  Record failure context, retain changes, and continue iterating

  --stall-limit <N>
      Stop early after N consecutive stalls (`agent_failed`, `test_failed`,
      or `no_changes`) on the same active stage. `0` disables stall stopping.

  --smoke-cmd "<cmd>"
      Override the smoke-check command run after each iteration.

  --stage-tests <auto|cmd>
      auto  Select stage-specific verification commands from the active stage
      cmd   Run the provided shell command after the smoke command

  --status-headings
      Enable machine-readable roadmap stage parsing using `## [ ] Stage N: ...`
      headings and `Automation: auto|manual` markers.

  --help, -h
      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt-scope)
            PROMPT_SCOPE="$2"
            shift 2
            ;;
        --failure-policy)
            FAILURE_POLICY="$2"
            shift 2
            ;;
        --stall-limit)
            STALL_LIMIT="$2"
            shift 2
            ;;
        --smoke-cmd)
            SMOKE_CMD="$2"
            shift 2
            ;;
        --stage-tests)
            STAGE_TESTS_MODE="$2"
            shift 2
            ;;
        --status-headings)
            STATUS_HEADINGS=1
            shift
            ;;
        --help|-h)
            MODE="help"
            shift
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_tool() {
    local tool=$1
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool not found: $tool" >&2
        exit 1
    fi
}

tracked_worktree_dirty() {
    if ! git diff --quiet --ignore-submodules --; then
        return 0
    fi
    if ! git diff --cached --quiet --ignore-submodules --; then
        return 0
    fi
    return 1
}

repo_snapshot() {
    python3 - "$REPO_ROOT" "$STATE_DIR_PREFIX" <<'PY'
import hashlib
import pathlib
import subprocess
import sys

repo_root = sys.argv[1]
prefix = sys.argv[2]
digest = hashlib.sha256()

for command in (
    ["git", "diff", "--no-ext-diff", "--binary", "--cached", "--"],
    ["git", "diff", "--no-ext-diff", "--binary", "--"],
):
    digest.update(subprocess.check_output(command, cwd=repo_root))

paths = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard"],
    cwd=repo_root,
    text=True,
).splitlines()

for path in sorted(paths):
    if prefix and (path == prefix or path.startswith(prefix + "/")):
        continue
    full_path = pathlib.Path(repo_root, path)
    digest.update(path.encode("utf-8", "surrogateescape"))
    digest.update(b"\0")
    if full_path.is_file():
        digest.update(hashlib.sha256(full_path.read_bytes()).digest())
    else:
        digest.update(b"<non-file>")
    digest.update(b"\0")

print(digest.hexdigest())
PY
}

write_untracked_list() {
    local output_file=$1
    git ls-files --others --exclude-standard | python3 -c '
import sys

prefix = sys.argv[1]
paths = []
for raw_line in sys.stdin:
    path = raw_line.rstrip("\n")
    if prefix and (path == prefix or path.startswith(prefix + "/")):
        continue
    paths.append(path)
for path in sorted(paths):
    print(path)
' "$STATE_DIR_PREFIX" > "$output_file"
}

append_managed_untracked() {
    local before_file=$1
    local after_file=$2
    local delta_file
    delta_file="$(mktemp)"
    comm -13 "$before_file" "$after_file" > "$delta_file" || true
    if [[ -s "$delta_file" ]]; then
        cat "$delta_file" >> "$MANAGED_UNTRACKED_FILE"
        sort -u "$MANAGED_UNTRACKED_FILE" -o "$MANAGED_UNTRACKED_FILE"
    fi
    rm -f "$delta_file"
}

stage_managed_changes() {
    git add -u --
    if [[ ! -f "$MANAGED_UNTRACKED_FILE" ]]; then
        return 0
    fi
    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        if [[ -e "$REPO_ROOT/$path" ]]; then
            git add -- "$path"
        fi
    done < "$MANAGED_UNTRACKED_FILE"
}

write_state() {
    local session_id=$1
    local branch=$2
    local start_sha=$3
    local next_iteration=$4
    local last_result=$5
    local last_failure_path=$6
    local active_stage_number=$7
    local active_stage_title=$8
    local active_stage_status=$9
    local consecutive_stalls=${10}
    local success_count=${11}
    local failure_count=${12}

    python3 - "$STATE_FILE" "$session_id" "$branch" "$start_sha" "$next_iteration" \
        "$last_result" "$last_failure_path" "$MANAGED_UNTRACKED_FILE" "$active_stage_number" \
        "$active_stage_title" "$active_stage_status" "$consecutive_stalls" \
        "$success_count" "$failure_count" <<'PY'
import json
import sys

(
    state_path,
    session_id,
    branch,
    start_sha,
    next_iteration,
    last_result,
    last_failure_path,
    managed_untracked,
    active_stage_number,
    active_stage_title,
    active_stage_status,
    consecutive_stalls,
    success_count,
    failure_count,
) = sys.argv[1:15]

payload = {
    "session_id": session_id,
    "branch": branch,
    "start_sha": start_sha,
    "next_iteration": int(next_iteration),
    "last_result": last_result,
    "last_failure_path": last_failure_path,
    "managed_untracked_path": managed_untracked,
    "active_stage_number": int(active_stage_number) if active_stage_number else None,
    "active_stage_title": active_stage_title,
    "active_stage_status": active_stage_status,
    "consecutive_stalls": int(consecutive_stalls),
    "success_count": int(success_count),
    "failure_count": int(failure_count),
}

with open(state_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

read_state_field() {
    local field=$1
    python3 - "$STATE_FILE" "$field" <<'PY'
import json
import sys

state_path, field = sys.argv[1:3]
with open(state_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data.get(field, "")
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

read_state_int_field() {
    local field=$1
    if [[ ! -f "$STATE_FILE" ]]; then
        echo "0"
        return
    fi
    local value
    value="$(read_state_field "$field" 2>/dev/null || true)"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value"
    else
        echo "0"
    fi
}

ensure_summary_file() {
    local summary_file=$1
    if [[ ! -s "$summary_file" ]]; then
        cat > "$summary_file" <<'EOF'
Codex did not write a final summary message for this iteration.
EOF
    fi
}

extract_commit_subject_from_summary() {
    local summary_file=$1
    python3 - "$summary_file" <<'PY'
import pathlib
import re
import sys
import textwrap

summary_path = pathlib.Path(sys.argv[1])
text = summary_path.read_text(encoding="utf-8", errors="replace")
lines = [line.strip() for line in text.splitlines() if line.strip()]

candidate = ""
for line in lines:
    match = re.match(r"^(?:commit message|commit|subject)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        break

if not candidate:
    for line in lines:
        lowered = line.lower()
        if lowered in {"files changed:", "checks run:", "remaining risk:"}:
            continue
        if lowered.startswith(("files changed:", "checks run:", "remaining risk:")):
            continue
        candidate = line
        break

if candidate:
    first_sentence = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)[0].strip()
    if first_sentence:
        candidate = first_sentence

    prefix, separator, suffix = candidate.partition(": ")
    if separator and any(
        token in prefix.lower()
        for token in (
            "implemented",
            "added",
            "updated",
            "fixed",
            "refactored",
            "improved",
            "stage",
            "slice",
        )
    ):
        candidate = suffix.strip() or candidate

candidate = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", candidate)
candidate = candidate.replace("`", "")
candidate = re.sub(r"^\s*[-*]\s*", "", candidate)
candidate = re.sub(r"\s+", " ", candidate).strip().strip(".")
candidate = re.sub(r"^(?:plan-loop|loop)\s*:\s*", "", candidate, flags=re.IGNORECASE)

if not candidate or candidate == "Codex did not write a final summary message for this iteration":
    print("")
    raise SystemExit

lowered = candidate.lower()
if re.fullmatch(r"stage \d+ iteration \d+", lowered):
    print("")
    raise SystemExit
if re.fullmatch(r"roadmap iteration \d+", lowered):
    print("")
    raise SystemExit

if len(candidate) > 120:
    candidate = textwrap.shorten(candidate, width=120, placeholder="...")

print(candidate)
PY
}

build_fallback_commit_subject() {
    local stage_number=$1
    local stage_title=$2
    python3 - "$stage_number" "$stage_title" <<'PY'
import re
import subprocess
import sys
import textwrap

stage_number, stage_title = sys.argv[1:3]
paths = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD", "--"],
    text=True,
).splitlines()
paths = [path.strip() for path in paths if path.strip()]

if paths:
    if len(paths) == 1:
        subject = f"update {paths[0]}"
    elif len(paths) == 2:
        subject = f"update {paths[0]} and {paths[1]}"
    else:
        subject = f"update {paths[0]}, {paths[1]}, and {len(paths) - 2} more files"
elif stage_number and stage_title:
    clean_title = re.sub(r"\s+", " ", stage_title.strip())
    subject = f"advance stage {stage_number}: {clean_title}" if clean_title else f"advance stage {stage_number}"
else:
    subject = "update roadmap loop changes"

if len(subject) > 120:
    subject = textwrap.shorten(subject, width=120, placeholder="...")

print(subject)
PY
}

build_iteration_commit_message() {
    local summary_file=$1
    local stage_number=$2
    local stage_title=$3
    local subject

    subject="$(extract_commit_subject_from_summary "$summary_file")"
    if [[ -z "$subject" ]]; then
        subject="$(build_fallback_commit_subject "$stage_number" "$stage_title")"
    fi

    if [[ "$subject" =~ ^(plan-loop|loop): ]]; then
        printf '%s\n' "$subject"
    else
        printf 'plan-loop: %s\n' "$subject"
    fi
}

write_failure_summary() {
    local source_log=$1
    local output_file=$2
    python3 - "$source_log" "$output_file" <<'PY'
import pathlib
import sys

source_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
important = []
for line in lines:
    lowered = line.lower()
    if any(token in lowered for token in ("error:", "failed", "failure", "panicked at", "test result:")):
        important.append(line)
if not important:
    important = lines[-40:]
else:
    important = important[-20:]
output_path.write_text("\n".join(important).strip() + "\n", encoding="utf-8")
PY
}

roadmap_write_info() {
    local output_file=$1
    python3 - "$ROADMAP_PATH" "$output_file" <<'PY'
import json
import pathlib
import re
import sys

roadmap_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
lines = roadmap_path.read_text(encoding="utf-8").splitlines()
stage_re = re.compile(r"^## \[( |~|x)\] Stage (\d+): (.+)$")

def extract_block(title: str) -> str:
    heading = f"## {title}"
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            end = len(lines)
            for j in range(idx + 1, len(lines)):
                if lines[j].startswith("## "):
                    end = j
                    break
            block = "\n".join(lines[idx:end]).strip()
            return block + "\n" if block else ""
    return ""

stage_starts = []
for idx, line in enumerate(lines):
    match = stage_re.match(line)
    if match:
        stage_starts.append((idx, match))

stages = []
for pos, (start_idx, match) in enumerate(stage_starts):
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if lines[j].startswith("## "):
            end_idx = j
            break
    block_lines = lines[start_idx:end_idx]
    status_char = match.group(1)
    status = "[ ]" if status_char == " " else f"[{status_char}]"
    automation = ""
    for line in block_lines[1:]:
        if line.startswith("Automation:"):
            automation = line.split(":", 1)[1].strip()
            break
    stages.append(
        {
            "number": int(match.group(2)),
            "title": match.group(3),
            "status": status,
            "automation": automation,
            "block": "\n".join(block_lines).strip() + "\n",
        }
    )

active_stage = next((stage for stage in stages if stage["status"] != "[x]"), None)
payload = {
    "goal_block": extract_block("Goal"),
    "principles_block": extract_block("Principles"),
    "stages": stages,
    "stage_count": len(stages),
    "active_stage": active_stage,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

roadmap_info_field() {
    local info_file=$1
    local field_path=$2
    python3 - "$info_file" "$field_path" <<'PY'
import json
import sys

info_path, field_path = sys.argv[1:3]
with open(info_path, "r", encoding="utf-8") as handle:
    value = json.load(handle)

for part in field_path.split("."):
    if not part:
        continue
    if isinstance(value, dict):
        value = value.get(part, "")
    else:
        value = ""
        break

if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

validate_machine_readable_roadmap() {
    local info_file=$1
    local stage_count
    stage_count="$(roadmap_info_field "$info_file" "stage_count")"
    if ! [[ "$stage_count" =~ ^[0-9]+$ ]] || [[ "$stage_count" -lt 1 ]]; then
        echo "error: roadmap parsing enabled but no machine-readable stage headings were found in $ROADMAP_PATH" >&2
        exit 1
    fi
    python3 - "$info_file" "$ROADMAP_PATH" <<'PY'
import json
import sys

info_path, roadmap_path = sys.argv[1:3]
with open(info_path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

invalid = []
for stage in data["stages"]:
    automation = stage.get("automation", "")
    if automation not in {"auto", "manual"}:
        invalid.append((stage["number"], automation or "missing"))

if invalid:
    problems = ", ".join(
        f"Stage {number} has invalid Automation value '{value}'"
        for number, value in invalid
    )
    print(
        f"error: machine-readable roadmap stages in {roadmap_path} must define "
        f"Automation: auto|manual immediately under each heading; {problems}",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

print_status() {
    if (( STATUS_HEADINGS )); then
        local roadmap_info
        roadmap_info="$(mktemp)"
        roadmap_write_info "$roadmap_info"
        validate_machine_readable_roadmap "$roadmap_info"

        local active_number active_title active_status active_automation
        active_number="$(roadmap_info_field "$roadmap_info" "active_stage.number")"
        active_title="$(roadmap_info_field "$roadmap_info" "active_stage.title")"
        active_status="$(roadmap_info_field "$roadmap_info" "active_stage.status")"
        active_automation="$(roadmap_info_field "$roadmap_info" "active_stage.automation")"

        if [[ -n "$active_number" ]]; then
            echo "Roadmap active stage: ${active_status} Stage ${active_number}: ${active_title} (automation: ${active_automation:-missing})"
        else
            echo "Roadmap active stage: all machine-readable stages are marked complete"
        fi
        echo
        echo "Roadmap stages:"
        python3 - "$roadmap_info" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)

for stage in data["stages"]:
    automation = stage["automation"] or "missing"
    print(f"  {stage['status']} Stage {stage['number']}: {stage['title']} (automation: {automation})")
PY
        rm -f "$roadmap_info"
        echo
    fi

    if [[ ! -f "$STATE_FILE" ]]; then
        echo "No saved session state at $STATE_FILE"
        return 0
    fi

    echo "Saved session state:"
    echo "  session_id: $(read_state_field session_id)"
    echo "  branch: $(read_state_field branch)"
    echo "  start_sha: $(read_state_field start_sha)"
    echo "  next_iteration: $(read_state_field next_iteration)"
    echo "  last_result: $(read_state_field last_result)"
    echo "  last_failure_path: $(read_state_field last_failure_path)"
    echo "  active_stage_number: $(read_state_field active_stage_number)"
    echo "  active_stage_title: $(read_state_field active_stage_title)"
    echo "  active_stage_status: $(read_state_field active_stage_status)"
    echo "  consecutive_stalls: $(read_state_field consecutive_stalls)"
    echo "  success_count: $(read_state_field success_count)"
    echo "  failure_count: $(read_state_field failure_count)"
}

build_prompt() {
    local prompt_file=$1
    local branch=$2
    local head_sha=$3
    local previous_summary_file=$4
    local failure_summary_file=$5
    local iteration=$6
    local roadmap_info_file=$7

    if [[ "$PROMPT_SCOPE" == "active-stage" ]]; then
        local goal_block principles_block active_stage_block active_stage_number active_stage_title
        goal_block="$(roadmap_info_field "$roadmap_info_file" "goal_block")"
        principles_block="$(roadmap_info_field "$roadmap_info_file" "principles_block")"
        active_stage_block="$(roadmap_info_field "$roadmap_info_file" "active_stage.block")"
        active_stage_number="$(roadmap_info_field "$roadmap_info_file" "active_stage.number")"
        active_stage_title="$(roadmap_info_field "$roadmap_info_file" "active_stage.title")"

        {
            cat <<EOF
You are working in the Git repository at $REPO_ROOT on branch $branch at commit $head_sha.

Execute exactly one small slice from the active roadmap stage only.

Requirements:
- Implement exactly one small slice inside the active stage.
- Do not jump to later stages.
- Run the most relevant local checks for the files you touch.
- Update the active stage status from [ ] to [~] when work begins.
- Update [~] to [x] only if the stage exit criteria are satisfied and the dedicated stage verification passes.
- Do not edit $STATE_DIR.
- Leave the repository in a state where the outer loop can run its smoke and stage-specific tests.
- If the active stage is Stage 1, do not spend the iteration on coverage-only, test-only, or refactor-only work.
- For Stage 1, the slice must include a runtime behavior change intended to reduce the fast-modulation artifact, not just more instrumentation or guards around it.
- For Stage 1, only add tests or QA coverage when they are directly paired with the runtime fix in the same slice.
- For Stage 1, prefer work that reruns quality_gate_dual_plane_fast_modulation_artifacts after the change or materially increases the chance that gate will improve on the next iteration.
- End with a Commit message: line containing a single git subject that states what actually changed.
- Then include a concise summary of files changed, checks run, and remaining risk.

This is loop iteration $iteration.
This is the active roadmap stage: Stage $active_stage_number - $active_stage_title

EOF
            echo "$goal_block"
            echo
            echo "$principles_block"
            echo
            echo "## Active Stage"
            echo "$active_stage_block"
            echo
            echo "## Previous Iteration Summary"
            if [[ -f "$previous_summary_file" ]]; then
                cat "$previous_summary_file"
            else
                echo "No prior iteration summary is available."
            fi
            echo
            echo "## Last Failure Context"
            if [[ -f "$failure_summary_file" ]]; then
                cat "$failure_summary_file"
            else
                echo "No prior failure context is recorded."
            fi
        } > "$prompt_file"
        return
    fi

    {
        cat <<EOF
You are working in the Git repository at $REPO_ROOT on branch $branch at commit $head_sha.

Execute exactly one small roadmap slice from ROADMAP.md. Favor the smallest production-grade improvement that materially advances the roadmap, preferably in the current or next active phase.

Requirements:
- Implement one coherent slice only.
- Run the most relevant local checks for the files you touch.
- Update ROADMAP.md only if it materially needs a status or progress adjustment based on the work you complete.
- Do not edit $STATE_DIR.
- Leave the repository in a state where the outer loop can run its fast test suite.
- End with a Commit message: line containing a single git subject that states what actually changed.
- Then include a concise summary of files changed, checks run, and remaining risk.

This is loop iteration $iteration.

EOF
        echo "## Roadmap"
        cat "$ROADMAP_PATH"
        echo
        echo "## Previous Iteration Summary"
        if [[ -f "$previous_summary_file" ]]; then
            cat "$previous_summary_file"
        else
            echo "No prior iteration summary is available."
        fi
        echo
        echo "## Last Failure Context"
        if [[ -f "$failure_summary_file" ]]; then
            cat "$failure_summary_file"
        else
            echo "No prior failure context is recorded."
        fi
    } > "$prompt_file"
}

stage_test_commands_auto() {
    local stage_number=$1
    case "$stage_number" in
        1)
            cat <<'EOF'
cargo test --features qa-harnesses --release --test quality_gates quality_gate_dual_plane_fast_modulation_artifacts -- --nocapture
EOF
            ;;
        2)
            cat <<'EOF'
cargo test --features qa-harnesses --test quality_gates quality_gate_batch_vs_stream_hybrid_subset -- --nocapture
cargo test --test hybrid_streaming --test streaming_batch_parity --test stretch_quality_regressions
EOF
            ;;
        3)
            cat <<'EOF'
cargo test --test preanalysis_pipeline --test spectral_quality --test bpm_stretch
EOF
            ;;
        4)
            cat <<'EOF'
cargo test --test band_split --test spectral_quality --test quality
EOF
            ;;
        5)
            cat <<'EOF'
cargo test --test stretch_quality_regressions --test quality --test spectral_quality
EOF
            ;;
        6)
            cat <<'EOF'
cargo test --features qa-harnesses --release --test quality_gates quality_gate_streaming_callback_budget_tempo_and_pitch_modulation -- --nocapture
cargo test --test fixed_buffer_streaming --test pitch_shift --test stretch_quality_regressions
EOF
            ;;
        7)
            cat <<'EOF'
cargo test --test edge_cases --test algorithm_edge_cases --test buffer_workflows --test wav_error_paths
EOF
            ;;
        8)
            cat <<'EOF'
cargo test --features qa-harnesses --test quality_benchmark -- --ignored --nocapture
cargo test --features qa-harnesses --test reference_quality -- --nocapture
cargo test --features qa-harnesses --test rubberband_comparison -- --nocapture
EOF
            ;;
    esac
}

resolved_stage_test_commands() {
    local mode=$1
    local stage_number=$2
    if [[ -z "$mode" ]]; then
        return 0
    fi
    if [[ "$mode" == "auto" ]]; then
        stage_test_commands_auto "$stage_number"
    else
        printf '%s\n' "$mode"
    fi
}

run_command_series() {
    local log_file=$1
    shift
    : > "$log_file"
    local command
    for command in "$@"; do
        [[ -n "$command" ]] || continue
        echo "==> running: $command" | tee -a "$log_file"
        if ! (
            cd "$REPO_ROOT"
            bash -lc "$command"
        ) 2>&1 | tee -a "$log_file"; then
            return 1
        fi
    done
    return 0
}

start_session() {
    if [[ -f "$STATE_FILE" ]]; then
        echo "error: existing loop state found at $STATE_FILE" >&2
        echo "Use the matching resume target or remove the old session state first." >&2
        exit 1
    fi
    if tracked_worktree_dirty; then
        echo "error: tracked worktree is dirty; commit or stash tracked changes before starting the loop." >&2
        exit 1
    fi

    mkdir -p "$ITERATIONS_DIR"
    : > "$MANAGED_UNTRACKED_FILE"

    local session_id branch start_sha active_stage_number active_stage_title active_stage_status
    session_id="roadmap-loop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    branch="$(git rev-parse --abbrev-ref HEAD)"
    start_sha="$(git rev-parse HEAD)"
    active_stage_number=""
    active_stage_title=""
    active_stage_status=""

    if (( STATUS_HEADINGS )); then
        local roadmap_info
        roadmap_info="$(mktemp)"
        roadmap_write_info "$roadmap_info"
        validate_machine_readable_roadmap "$roadmap_info"
        active_stage_number="$(roadmap_info_field "$roadmap_info" "active_stage.number")"
        active_stage_title="$(roadmap_info_field "$roadmap_info" "active_stage.title")"
        active_stage_status="$(roadmap_info_field "$roadmap_info" "active_stage.status")"
        rm -f "$roadmap_info"
    fi

    write_state "$session_id" "$branch" "$start_sha" "1" "initialized" "" \
        "$active_stage_number" "$active_stage_title" "$active_stage_status" "0" "0" "0"
}

resume_session() {
    if [[ ! -f "$STATE_FILE" ]]; then
        echo "error: no saved loop state found at $STATE_FILE" >&2
        exit 1
    fi

    local expected_branch current_branch last_result
    expected_branch="$(read_state_field branch)"
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "$current_branch" != "$expected_branch" ]]; then
        echo "error: loop state expects branch '$expected_branch' but current branch is '$current_branch'" >&2
        exit 1
    fi

    last_result="$(read_state_field last_result)"
    if tracked_worktree_dirty && [[ "$last_result" != "test_failed" ]] && [[ "$last_result" != "agent_failed" ]]; then
        echo "error: tracked worktree is dirty; resume is only allowed from a failed iteration with pending changes." >&2
        exit 1
    fi

    mkdir -p "$ITERATIONS_DIR"
    touch "$MANAGED_UNTRACKED_FILE"
}

run_iteration() {
    local session_id=$1
    local branch=$2
    local start_sha=$3
    local iteration=$4

    local iteration_dir prompt_file summary_file codex_log test_log failure_summary_file previous_summary_file
    local pre_snapshot post_snapshot head_sha last_failure_path agent_status tests_status
    local before_untracked after_untracked roadmap_info pre_stage_number pre_stage_title pre_stage_status
    local pre_stage_automation post_stage_number post_stage_title post_stage_status
    local current_stalls success_count failure_count stage_advanced new_stalls commit_message

    current_stalls="$(read_state_int_field consecutive_stalls)"
    success_count="$(read_state_int_field success_count)"
    failure_count="$(read_state_int_field failure_count)"

    iteration_dir="$ITERATIONS_DIR/$iteration"
    mkdir -p "$iteration_dir"

    prompt_file="$iteration_dir/prompt.md"
    summary_file="$iteration_dir/summary.md"
    codex_log="$iteration_dir/codex.log"
    test_log="$iteration_dir/test.log"
    failure_summary_file="$iteration_dir/failure_summary.txt"

    if (( iteration > 1 )); then
        previous_summary_file="$ITERATIONS_DIR/$((iteration - 1))/summary.md"
    else
        previous_summary_file=""
    fi

    roadmap_info=""
    pre_stage_number=""
    pre_stage_title=""
    pre_stage_status=""
    pre_stage_automation=""
    if (( STATUS_HEADINGS )); then
        roadmap_info="$(mktemp)"
        roadmap_write_info "$roadmap_info"
        validate_machine_readable_roadmap "$roadmap_info"

        pre_stage_number="$(roadmap_info_field "$roadmap_info" "active_stage.number")"
        pre_stage_title="$(roadmap_info_field "$roadmap_info" "active_stage.title")"
        pre_stage_status="$(roadmap_info_field "$roadmap_info" "active_stage.status")"
        pre_stage_automation="$(roadmap_info_field "$roadmap_info" "active_stage.automation")"

        if [[ -z "$pre_stage_number" ]]; then
            write_state "$session_id" "$branch" "$start_sha" "$iteration" "complete" "" \
                "" "" "" "0" "$success_count" "$failure_count"
            echo "All machine-readable roadmap stages are marked complete; stopping."
            rm -f "$roadmap_info"
            return 10
        fi

        if [[ "$pre_stage_automation" == "manual" ]]; then
            write_state "$session_id" "$branch" "$start_sha" "$iteration" "manual_stage" "" \
                "$pre_stage_number" "$pre_stage_title" "$pre_stage_status" "$current_stalls" \
                "$success_count" "$failure_count"
            echo "Active roadmap stage ${pre_stage_status} Stage ${pre_stage_number}: ${pre_stage_title} is marked Automation: manual; stopping."
            rm -f "$roadmap_info"
            return 10
        fi
    fi

    last_failure_path="$(read_state_field last_failure_path)"
    head_sha="$(git rev-parse HEAD)"
    build_prompt "$prompt_file" "$branch" "$head_sha" "$previous_summary_file" "$last_failure_path" "$iteration" "$roadmap_info"

    pre_snapshot="$(repo_snapshot)"
    before_untracked="$(mktemp)"
    after_untracked="$(mktemp)"
    write_untracked_list "$before_untracked"

    echo "==> Iteration $iteration: codex exec"
    if codex exec --full-auto --model "$MODEL" -C "$REPO_ROOT" --output-last-message "$summary_file" - < "$prompt_file" 2>&1 | tee "$codex_log"; then
        agent_status="ok"
    else
        agent_status="failed"
    fi
    ensure_summary_file "$summary_file"

    post_snapshot="$(repo_snapshot)"
    write_untracked_list "$after_untracked"
    append_managed_untracked "$before_untracked" "$after_untracked"
    rm -f "$before_untracked" "$after_untracked"

    if [[ "$FAILURE_POLICY" == "stop" ]] && [[ "$pre_snapshot" == "$post_snapshot" ]]; then
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "no_changes" "" \
            "$pre_stage_number" "$pre_stage_title" "$pre_stage_status" "$current_stalls" \
            "$success_count" "$failure_count"
        echo "No repository changes detected in iteration $iteration; stopping early."
        rm -f "$roadmap_info"
        return 10
    fi

    if [[ "$FAILURE_POLICY" == "continue" ]] && [[ "$agent_status" == "ok" ]] && [[ "$pre_snapshot" == "$post_snapshot" ]]; then
        new_stalls=$((current_stalls + 1))
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "no_changes" "" \
            "$pre_stage_number" "$pre_stage_title" "$pre_stage_status" "$new_stalls" \
            "$success_count" "$failure_count"
        echo "No repository changes detected in iteration $iteration; continuing."
        rm -f "$roadmap_info"
        if (( STALL_LIMIT > 0 && new_stalls >= STALL_LIMIT )); then
            echo "Stall limit reached after $new_stalls consecutive stalls on the active stage; stopping."
            return 10
        fi
        return 0
    fi

    if [[ "$agent_status" != "ok" ]]; then
        write_failure_summary "$codex_log" "$failure_summary_file"
        post_stage_number="$pre_stage_number"
        post_stage_title="$pre_stage_title"
        post_stage_status="$pre_stage_status"
        stage_advanced=0
        if (( STATUS_HEADINGS )); then
            roadmap_write_info "$roadmap_info"
            post_stage_number="$(roadmap_info_field "$roadmap_info" "active_stage.number")"
            post_stage_title="$(roadmap_info_field "$roadmap_info" "active_stage.title")"
            post_stage_status="$(roadmap_info_field "$roadmap_info" "active_stage.status")"
            if [[ "$post_stage_number" != "$pre_stage_number" ]]; then
                stage_advanced=1
            fi
        fi
        failure_count=$((failure_count + 1))
        if (( stage_advanced )); then
            new_stalls=0
        else
            new_stalls=$((current_stalls + 1))
        fi
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "agent_failed" "$failure_summary_file" \
            "$post_stage_number" "$post_stage_title" "$post_stage_status" "$new_stalls" \
            "$success_count" "$failure_count"
        echo "Codex failed during iteration $iteration; recorded failure context at $failure_summary_file"
        rm -f "$roadmap_info"
        if [[ "$FAILURE_POLICY" == "stop" ]]; then
            return 20
        fi
        if (( STALL_LIMIT > 0 && new_stalls >= STALL_LIMIT )); then
            echo "Stall limit reached after $new_stalls consecutive stalls on the active stage; stopping."
            return 10
        fi
        return 0
    fi

    local -a check_commands=()
    if [[ -n "$SMOKE_CMD" ]]; then
        check_commands+=("$SMOKE_CMD")
    fi
    if [[ -n "$STAGE_TESTS_MODE" ]]; then
        while IFS= read -r command; do
            [[ -n "$command" ]] || continue
            check_commands+=("$command")
        done < <(resolved_stage_test_commands "$STAGE_TESTS_MODE" "$pre_stage_number")
    fi

    echo "==> Iteration $iteration: running checks"
    if run_command_series "$test_log" "${check_commands[@]}"; then
        tests_status="ok"
    else
        tests_status="failed"
    fi

    if [[ "$tests_status" != "ok" ]]; then
        write_failure_summary "$test_log" "$failure_summary_file"
        post_stage_number="$pre_stage_number"
        post_stage_title="$pre_stage_title"
        post_stage_status="$pre_stage_status"
        stage_advanced=0
        if (( STATUS_HEADINGS )); then
            roadmap_write_info "$roadmap_info"
            post_stage_number="$(roadmap_info_field "$roadmap_info" "active_stage.number")"
            post_stage_title="$(roadmap_info_field "$roadmap_info" "active_stage.title")"
            post_stage_status="$(roadmap_info_field "$roadmap_info" "active_stage.status")"
            if [[ "$post_stage_number" != "$pre_stage_number" ]]; then
                stage_advanced=1
            fi
        fi
        failure_count=$((failure_count + 1))
        if (( stage_advanced )); then
            new_stalls=0
        else
            new_stalls=$((current_stalls + 1))
        fi
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "test_failed" "$failure_summary_file" \
            "$post_stage_number" "$post_stage_title" "$post_stage_status" "$new_stalls" \
            "$success_count" "$failure_count"
        echo "Checks failed during iteration $iteration; recorded failure context at $failure_summary_file"
        rm -f "$roadmap_info"
        if [[ "$FAILURE_POLICY" == "stop" ]]; then
            return 30
        fi
        if (( STALL_LIMIT > 0 && new_stalls >= STALL_LIMIT )); then
            echo "Stall limit reached after $new_stalls consecutive stalls on the active stage; stopping."
            return 10
        fi
        return 0
    fi

    stage_managed_changes
    if git diff --cached --quiet --ignore-submodules --; then
        if [[ "$FAILURE_POLICY" == "stop" ]]; then
            write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "no_changes" "" \
                "$pre_stage_number" "$pre_stage_title" "$pre_stage_status" "$current_stalls" \
                "$success_count" "$failure_count"
            echo "No committable changes remained after staging; stopping early."
            rm -f "$roadmap_info"
            return 10
        fi
        new_stalls=$((current_stalls + 1))
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "no_changes" "" \
            "$pre_stage_number" "$pre_stage_title" "$pre_stage_status" "$new_stalls" \
            "$success_count" "$failure_count"
        echo "No committable changes remained after staging; continuing."
        rm -f "$roadmap_info"
        if (( STALL_LIMIT > 0 && new_stalls >= STALL_LIMIT )); then
            echo "Stall limit reached after $new_stalls consecutive stalls on the active stage; stopping."
            return 10
        fi
        return 0
    fi

    if (( STATUS_HEADINGS )); then
        roadmap_write_info "$roadmap_info"
        post_stage_number="$(roadmap_info_field "$roadmap_info" "active_stage.number")"
        post_stage_title="$(roadmap_info_field "$roadmap_info" "active_stage.title")"
        post_stage_status="$(roadmap_info_field "$roadmap_info" "active_stage.status")"
    else
        post_stage_number=""
        post_stage_title=""
        post_stage_status=""
    fi

    commit_message="$(build_iteration_commit_message "$summary_file" "$pre_stage_number" "$pre_stage_title")"
    git commit -m "$commit_message" >/dev/null

    success_count=$((success_count + 1))
    write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "success" "" \
        "$post_stage_number" "$post_stage_title" "$post_stage_status" "0" \
        "$success_count" "$failure_count"
    echo "Committed iteration $iteration: $commit_message"
    rm -f "$roadmap_info"
    return 0
}

main() {
    case "$MODE" in
        start|resume|status)
            ;;
        help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac

    if [[ "$PROMPT_SCOPE" != "full" ]] && [[ "$PROMPT_SCOPE" != "active-stage" ]]; then
        echo "error: --prompt-scope must be 'full' or 'active-stage'" >&2
        exit 1
    fi
    if [[ "$FAILURE_POLICY" != "stop" ]] && [[ "$FAILURE_POLICY" != "continue" ]]; then
        echo "error: --failure-policy must be 'stop' or 'continue'" >&2
        exit 1
    fi
    if ! [[ "$STALL_LIMIT" =~ ^[0-9]+$ ]]; then
        echo "error: --stall-limit must be a non-negative integer" >&2
        exit 1
    fi
    if [[ "$PROMPT_SCOPE" == "active-stage" ]] && (( ! STATUS_HEADINGS )); then
        echo "error: --prompt-scope active-stage requires --status-headings" >&2
        exit 1
    fi
    if [[ "$STAGE_TESTS_MODE" == "auto" ]] && (( ! STATUS_HEADINGS )); then
        echo "error: --stage-tests auto requires --status-headings" >&2
        exit 1
    fi

    require_tool python3

    if [[ ! -f "$ROADMAP_PATH" ]]; then
        echo "error: roadmap file not found at $ROADMAP_PATH" >&2
        exit 1
    fi

    if [[ "$MODE" == "status" ]]; then
        print_status
        exit 0
    fi

    require_tool git
    require_tool codex

    if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
        echo "error: ITERATIONS must be a positive integer" >&2
        exit 1
    fi

    cd "$REPO_ROOT"

    if [[ "$MODE" == "start" ]]; then
        start_session
    else
        resume_session
    fi

    local session_id branch start_sha next_iteration iteration stop_reason
    session_id="$(read_state_field session_id)"
    branch="$(read_state_field branch)"
    start_sha="$(read_state_field start_sha)"
    next_iteration="$(read_state_field next_iteration)"
    stop_reason=0

    for (( iteration=next_iteration; iteration<next_iteration+ITERATIONS; iteration++ )); do
        if run_iteration "$session_id" "$branch" "$start_sha" "$iteration"; then
            continue
        else
            stop_reason=$?
        fi
        case "$stop_reason" in
            10)
                break
                ;;
            20|30)
                exit "$stop_reason"
                ;;
            *)
                exit "$stop_reason"
                ;;
        esac
    done
}

main "$@"
