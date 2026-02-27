#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: maintain.sh [--mode fast|quality|full] [--update-deps] [--dry-run]

Options:
  --mode <tier>   Select check tier. Default: fast.
  --update-deps   Run conservative dependency update (`cargo update`) first.
  --dry-run       Print maintenance actions without executing commands.
  -h, --help      Show this help message.
EOF
}

mode="fast"
update_deps=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --mode" >&2
        usage >&2
        exit 2
      fi
      mode="$2"
      shift 2
      ;;
    --update-deps)
      update_deps=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$mode" in
  fast|quality|full) ;;
  *)
    echo "Invalid mode: $mode (expected fast|quality|full)" >&2
    exit 2
    ;;
esac

need_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Required command not found: $cmd" >&2
    exit 1
  }
}

run_step() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  echo "+ $*"
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
check_script="${script_dir}/check_tiers.sh"

need_cmd git
need_cmd cargo
if [[ ! -f "$check_script" ]]; then
  echo "Tier runner not found: $check_script" >&2
  exit 1
fi

timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
report=()

echo "Starting timestretch-rs maintenance run..."
echo "Timestamp (UTC): ${timestamp}"
echo "Mode: ${mode}"
echo "Update dependencies: $([[ "$update_deps" -eq 1 ]] && echo yes || echo no)"
echo "Dry run: $([[ "$dry_run" -eq 1 ]] && echo yes || echo no)"

status_lines="$(git status --porcelain || true)"
if [[ -n "$status_lines" ]]; then
  dirty_count="$(printf '%s\n' "$status_lines" | sed '/^$/d' | wc -l | tr -d ' ')"
  echo
  echo "Worktree is dirty (${dirty_count} paths). Preserving existing changes."
  printf '%s\n' "$status_lines" | sed 's/^/  /'
  report+=("Worktree dirty (${dirty_count} paths); preserved without reverting.")
else
  report+=("Worktree clean at start.")
fi

if [[ "$update_deps" -eq 1 ]]; then
  run_step "Dependency update (conservative semver within Cargo constraints)" cargo update
  report+=("Dependency update step completed.")
else
  report+=("Dependency update skipped.")
fi

run_step "Tier checks (${mode})" bash "$check_script" "$mode"
report+=("Tier checks '${mode}' completed.")

echo
echo "=== Maintenance Report ==="
echo "Timestamp (UTC): ${timestamp}"
echo "Mode: ${mode}"
echo "Result: success"
for line in "${report[@]}"; do
  echo "- ${line}"
done
