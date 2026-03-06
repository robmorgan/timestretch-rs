#!/bin/bash
set -euo pipefail

MODE=${1:-start}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROADMAP_PATH="$REPO_ROOT/ROADMAP.md"
MODEL="${MODEL:-gpt-5.4}"
ITERATIONS="${ITERATIONS:-5}"
STATE_DIR_INPUT="${STATE_DIR:-.codex-loop}"
TEST_CMD="${TEST_CMD:-cargo test --test dual_plane_rt --test realtime_allocations --test realtime_dj_conditions}"

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
Usage: scripts/roadmap_loop.sh [start|resume]

Modes:
  start   Create a new bounded roadmap execution session
  resume  Continue a prior session from state.json
EOF
}

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

    python3 - "$STATE_FILE" "$session_id" "$branch" "$start_sha" "$next_iteration" "$last_result" "$last_failure_path" "$MANAGED_UNTRACKED_FILE" <<'PY'
import json
import sys

state_path, session_id, branch, start_sha, next_iteration, last_result, last_failure_path, managed_untracked = sys.argv[1:9]
payload = {
    "session_id": session_id,
    "branch": branch,
    "start_sha": start_sha,
    "next_iteration": int(next_iteration),
    "last_result": last_result,
    "last_failure_path": last_failure_path,
    "managed_untracked_path": managed_untracked,
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

ensure_summary_file() {
    local summary_file=$1
    if [[ ! -f "$summary_file" ]]; then
        cat > "$summary_file" <<'EOF'
Codex did not write a final summary message for this iteration.
EOF
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

build_prompt() {
    local prompt_file=$1
    local branch=$2
    local head_sha=$3
    local previous_summary_file=$4
    local failure_summary_file=$5
    local iteration=$6

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
- End with a concise summary of files changed, checks run, and remaining risk.

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

start_session() {
    if [[ -f "$STATE_FILE" ]]; then
        echo "error: existing loop state found at $STATE_FILE" >&2
        echo "Use 'make resume' to continue or 'make clean-loop' to remove the old session." >&2
        exit 1
    fi
    if tracked_worktree_dirty; then
        echo "error: tracked worktree is dirty; commit or stash tracked changes before starting the loop." >&2
        exit 1
    fi

    mkdir -p "$ITERATIONS_DIR"
    : > "$MANAGED_UNTRACKED_FILE"

    local session_id branch start_sha
    session_id="roadmap-loop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    branch="$(git rev-parse --abbrev-ref HEAD)"
    start_sha="$(git rev-parse HEAD)"
    write_state "$session_id" "$branch" "$start_sha" "1" "initialized" ""
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
    local before_untracked after_untracked

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

    last_failure_path="$(read_state_field last_failure_path)"
    head_sha="$(git rev-parse HEAD)"
    build_prompt "$prompt_file" "$branch" "$head_sha" "$previous_summary_file" "$last_failure_path" "$iteration"

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

    if [[ "$pre_snapshot" == "$post_snapshot" ]]; then
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "no_changes" ""
        echo "No repository changes detected in iteration $iteration; stopping early."
        return 10
    fi

    if [[ "$agent_status" != "ok" ]]; then
        write_failure_summary "$codex_log" "$failure_summary_file"
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "agent_failed" "$failure_summary_file"
        echo "Codex failed during iteration $iteration; recorded failure context at $failure_summary_file"
        return 20
    fi

    echo "==> Iteration $iteration: running fast tests"
    if (
        cd "$REPO_ROOT"
        bash -lc "$TEST_CMD"
    ) 2>&1 | tee "$test_log"; then
        tests_status="ok"
    else
        tests_status="failed"
    fi

    if [[ "$tests_status" != "ok" ]]; then
        write_failure_summary "$test_log" "$failure_summary_file"
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "test_failed" "$failure_summary_file"
        echo "Fast tests failed during iteration $iteration; recorded failure context at $failure_summary_file"
        return 30
    fi

    stage_managed_changes
    if git diff --cached --quiet --ignore-submodules --; then
        write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "no_changes" ""
        echo "No committable changes remained after staging; stopping early."
        return 10
    fi

    git commit -m "loop: roadmap iteration $iteration" >/dev/null
    write_state "$session_id" "$branch" "$start_sha" "$((iteration + 1))" "success" ""
    echo "Committed iteration $iteration as 'loop: roadmap iteration $iteration'"
    return 0
}

main() {
    case "$MODE" in
        start|resume)
            ;;
        --help|-h|help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 1
            ;;
    esac

    require_tool git
    require_tool codex
    require_tool python3

    if [[ ! -f "$ROADMAP_PATH" ]]; then
        echo "error: roadmap file not found at $ROADMAP_PATH" >&2
        exit 1
    fi

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
