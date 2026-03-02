#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SKILL_DIR}/../.." && pwd)"
REPORT_ROOT="${REPO_ROOT}/target/arch_perf_review"

TIER="${1:-runtime}"
case "${TIER}" in
  static|runtime|deep) ;;
  *)
    echo "Usage: $0 [static|runtime|deep]" >&2
    exit 2
    ;;
esac

mkdir -p "${REPORT_ROOT}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="${REPORT_ROOT}/${TIMESTAMP}_${TIER}"
STATUS_FILE="${RUN_DIR}/command_status.tsv"
REPORT_FILE="${RUN_DIR}/report.md"
mkdir -p "${RUN_DIR}"

printf "name\tstatus\texit_code\tduration_sec\tcommand\tnote\n" > "${STATUS_FILE}"

cat > "${RUN_DIR}/metadata.env" <<EOF
TIMESTAMP=${TIMESTAMP}
TIER=${TIER}
REPO_ROOT=${REPO_ROOT}
SKILL_DIR=${SKILL_DIR}
EOF

run_cmd() {
  local name="$1"
  local command_text="$2"
  local required_bin="${3:-}"
  local timeout_secs="${TIMESTRETCH_REVIEW_CMD_TIMEOUT_SECS:-0}"
  local log_file="${RUN_DIR}/${name}.log"
  local start_ts
  local end_ts
  local duration_sec
  local exit_code
  local status
  local note=""

  if [[ -n "${required_bin}" ]] && ! command -v "${required_bin}" >/dev/null 2>&1; then
    echo "Skipping: missing dependency '${required_bin}'" > "${log_file}"
    printf "%s\tSKIP\t-\t0\t%s\tmissing dependency: %s\n" \
      "${name}" "${command_text}" "${required_bin}" >> "${STATUS_FILE}"
    echo "[SKIP] ${name} (missing dependency: ${required_bin})"
    return
  fi

  start_ts="$(date +%s)"
  if [[ "${timeout_secs}" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN { exit !(${timeout_secs} > 0) }"; then
    if command -v python3 >/dev/null 2>&1; then
      python3 - "${timeout_secs}" "${REPO_ROOT}" "${command_text}" > "${log_file}" 2>&1 <<'PY'
import subprocess
import sys

timeout = float(sys.argv[1])
repo_root = sys.argv[2]
command_text = sys.argv[3]
cmd = ["bash", "-lc", f'cd "{repo_root}" && {command_text}']
try:
    completed = subprocess.run(cmd, check=False, timeout=timeout)
except subprocess.TimeoutExpired:
    print(f"ERROR: command timed out after {timeout:.0f}s", file=sys.stderr)
    sys.exit(124)
sys.exit(completed.returncode)
PY
      exit_code="$?"
      if [[ "${exit_code}" -eq 124 ]]; then
        note="timed out after ${timeout_secs}s"
      fi
    else
      bash -lc "cd \"${REPO_ROOT}\" && ${command_text}" > "${log_file}" 2>&1
      exit_code="$?"
    fi
  else
    bash -lc "cd \"${REPO_ROOT}\" && ${command_text}" > "${log_file}" 2>&1
    exit_code="$?"
  fi
  end_ts="$(date +%s)"
  duration_sec="$((end_ts - start_ts))"

  if [[ "${exit_code}" -eq 0 ]]; then
    status="PASS"
  else
    status="FAIL"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${name}" "${status}" "${exit_code}" "${duration_sec}" "${command_text}" "${note}" >> "${STATUS_FILE}"
  if [[ -n "${note}" ]]; then
    echo "[${status}] ${name} (exit=${exit_code}, ${duration_sec}s, ${note})"
  else
    echo "[${status}] ${name} (exit=${exit_code}, ${duration_sec}s)"
  fi
}

skip_cmd() {
  local name="$1"
  local command_text="$2"
  local reason="$3"
  local log_file="${RUN_DIR}/${name}.log"

  echo "Skipped: ${reason}" > "${log_file}"
  printf "%s\tSKIP\t-\t0\t%s\t%s\n" "${name}" "${command_text}" "${reason}" >> "${STATUS_FILE}"
  echo "[SKIP] ${name} (${reason})"
}

run_static_tier() {
  run_cmd "src_and_tests_file_map" "rg --files src tests benchmarks" "rg"
  run_cmd "architecture_doc_snapshot" "sed -n '1,220p' ARCHITECTURE.md" "sed"
  run_cmd "readme_snapshot" "sed -n '1,260p' README.md" "sed"
  run_cmd "realtime_pattern_scan" "rg -n '(realtime|latency|callback|allocation|hybrid|phase_vocoder|wsola|budget|parity)' src tests README.md ARCHITECTURE.md" "rg"
}

run_runtime_tier() {
  run_cmd "realtime_dj_conditions" \
    "cargo test --release --test realtime_dj_conditions -- --nocapture" \
    "cargo"
  run_cmd "realtime_allocations" \
    "cargo test --release --test realtime_allocations -- --nocapture" \
    "cargo"
  run_cmd "callback_budget_gate" \
    "TIMESTRETCH_STRICT_CALLBACK_BUDGET=1 cargo test --release --test quality_gates quality_gate_streaming_worst_case_callback_budget -- --nocapture" \
    "cargo"
  run_cmd "streaming_batch_parity" \
    "cargo test --release --test streaming_batch_parity -- --nocapture" \
    "cargo"
  run_cmd "bench_streaming" \
    "cargo test --release --test benchmarks bench_streaming -- --nocapture" \
    "cargo"
}

run_deep_tier() {
  local baseline_script="${REPO_ROOT}/benchmarks/run_m0_baseline.sh"
  local corpus_dir="${TIMESTRETCH_REVIEW_CORPUS_DIR:-${REPO_ROOT}/benchmarks/audio}"

  run_cmd "hybrid_subset_gate" \
    "cargo test --release --test quality_gates quality_gate_batch_vs_stream_hybrid_subset -- --nocapture" \
    "cargo"

  if [[ ! -x "${baseline_script}" ]]; then
    skip_cmd "m0_baseline" "./benchmarks/run_m0_baseline.sh" "missing executable: benchmarks/run_m0_baseline.sh"
    return
  fi

  if [[ ! -d "${corpus_dir}" ]]; then
    skip_cmd "m0_baseline" "./benchmarks/run_m0_baseline.sh" "missing external corpus directory: ${corpus_dir}"
    return
  fi

  if ! ls "${corpus_dir}"/*.wav >/dev/null 2>&1; then
    skip_cmd "m0_baseline" "./benchmarks/run_m0_baseline.sh" "missing external corpus audio (*.wav) in ${corpus_dir}"
    return
  fi

  run_cmd "m0_baseline" "./benchmarks/run_m0_baseline.sh" "bash"
}

case "${TIER}" in
  static) run_static_tier ;;
  runtime) run_runtime_tier ;;
  deep) run_deep_tier ;;
esac

bash "${SCRIPT_DIR}/render_report.sh" "${RUN_DIR}" "${TIER}" "${REPO_ROOT}" "${SKILL_DIR}"
echo "Report written to: ${REPORT_FILE}"
echo "Run directory: ${RUN_DIR}"
