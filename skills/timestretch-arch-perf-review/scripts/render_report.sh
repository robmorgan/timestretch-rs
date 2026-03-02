#!/usr/bin/env bash
set -u -o pipefail

RUN_DIR="${1:-}"
TIER_ARG="${2:-}"
REPO_ROOT_ARG="${3:-}"
SKILL_DIR_ARG="${4:-}"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Usage: $0 <run_dir> [tier] [repo_root] [skill_dir]" >&2
  exit 2
fi

STATUS_FILE="${RUN_DIR}/command_status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
  echo "Missing status file: ${STATUS_FILE}" >&2
  exit 2
fi

TIER="${TIER_ARG}"
REPO_ROOT="${REPO_ROOT_ARG}"
SKILL_DIR="${SKILL_DIR_ARG}"
if [[ -f "${RUN_DIR}/metadata.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/metadata.env"
fi
TIER="${TIER:-${TIER_ARG:-unknown}}"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_ARG:-$(pwd)}}"
SKILL_DIR="${SKILL_DIR:-${SKILL_DIR_ARG:-${REPO_ROOT}/skills/timestretch-arch-perf-review}}"

REPORT_FILE="${RUN_DIR}/report.md"
ISSUES_FILE="${RUN_DIR}/issues.tsv"
: > "${ISSUES_FILE}"

severity_for() {
  local name="$1"
  local status="$2"
  if [[ "${status}" == "SKIP" ]]; then
    echo "P3"
    return
  fi

  case "${name}" in
    realtime_allocations|callback_budget_gate) echo "P0" ;;
    realtime_dj_conditions|streaming_batch_parity) echo "P1" ;;
    bench_streaming|hybrid_subset_gate|m0_baseline) echo "P2" ;;
    *) echo "P3" ;;
  esac
}

file_ref_for() {
  local name="$1"
  case "${name}" in
    realtime_dj_conditions) echo "tests/realtime_dj_conditions.rs:1" ;;
    realtime_allocations) echo "tests/realtime_allocations.rs:1" ;;
    callback_budget_gate|hybrid_subset_gate) echo "tests/quality_gates.rs:1" ;;
    streaming_batch_parity) echo "tests/streaming_batch_parity.rs:1" ;;
    bench_streaming) echo "tests/benchmarks.rs:1" ;;
    m0_baseline) echo "benchmarks/run_m0_baseline.sh:1" ;;
    src_and_tests_file_map|realtime_pattern_scan) echo "src/lib.rs:1" ;;
    architecture_doc_snapshot) echo "ARCHITECTURE.md:1" ;;
    readme_snapshot) echo "README.md:1" ;;
    *) echo "N/A" ;;
  esac
}

impact_for() {
  local name="$1"
  local status="$2"
  if [[ "${status}" == "SKIP" ]]; then
    echo "review signal is incomplete for this check"
    return
  fi

  case "${name}" in
    realtime_allocations) echo "allocation in realtime paths can cause callback jitter/dropouts" ;;
    callback_budget_gate) echo "callback budget failure can break live playback guarantees" ;;
    realtime_dj_conditions) echo "realtime DJ conditions regression may impact live behavior" ;;
    streaming_batch_parity) echo "stream-vs-batch drift can cause audible inconsistency" ;;
    bench_streaming) echo "streaming throughput regression reduces CPU headroom" ;;
    hybrid_subset_gate) echo "hybrid subset quality mismatch indicates algorithmic divergence risk" ;;
    m0_baseline) echo "reference baseline unavailable reduces confidence in deep quality tracking" ;;
    src_and_tests_file_map|architecture_doc_snapshot|readme_snapshot|realtime_pattern_scan)
      echo "static architecture signal could not be fully captured"
      ;;
    *)
      echo "command-level review failure requires follow-up"
      ;;
  esac
}

remediation_for() {
  local name="$1"
  case "${name}" in
    realtime_allocations) echo "isolate callback path allocations and move buffers to preallocated state" ;;
    callback_budget_gate) echo "profile callback hotspots and reduce FFT/hop cost or work per callback" ;;
    realtime_dj_conditions) echo "inspect realtime path state transitions and verify DjBeatmatch preset assumptions" ;;
    streaming_batch_parity) echo "compare stream processor overlap/add state against batch implementation" ;;
    bench_streaming) echo "capture CPU profile and optimize critical loops before live deployment" ;;
    hybrid_subset_gate) echo "audit hybrid path transient handling and parity with non-hybrid stream mode" ;;
    m0_baseline) echo "provision reference corpus/dependencies and rerun baseline archive script" ;;
    src_and_tests_file_map|architecture_doc_snapshot|readme_snapshot|realtime_pattern_scan)
      echo "rerun static tier after fixing environment command availability"
      ;;
    *)
      echo "rerun the failing command with focused diagnostics"
      ;;
  esac
}

status_for() {
  local target="$1"
  awk -F'\t' -v t="${target}" 'NR > 1 && $1 == t { print $2; exit }' "${STATUS_FILE}"
}

status_or_not_run() {
  local target="$1"
  local value
  value="$(status_for "${target}")"
  if [[ -z "${value}" ]]; then
    echo "NOT_RUN"
  else
    echo "${value}"
  fi
}

command_for() {
  local target="$1"
  awk -F'\t' -v t="${target}" 'NR > 1 && $1 == t { print $5; exit }' "${STATUS_FILE}"
}

pass_count="$(awk -F'\t' 'NR > 1 && $2 == "PASS" { c++ } END { print c + 0 }' "${STATUS_FILE}")"
fail_count="$(awk -F'\t' 'NR > 1 && $2 == "FAIL" { c++ } END { print c + 0 }' "${STATUS_FILE}")"
skip_count="$(awk -F'\t' 'NR > 1 && $2 == "SKIP" { c++ } END { print c + 0 }' "${STATUS_FILE}")"
total_count="$(awk -F'\t' 'NR > 1 { c++ } END { print c + 0 }' "${STATUS_FILE}")"

while IFS=$'\t' read -r name status exit_code duration_sec command_text note; do
  [[ "${name}" == "name" ]] && continue
  if [[ "${status}" == "FAIL" || "${status}" == "SKIP" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$(severity_for "${name}" "${status}")" \
      "${name}" \
      "$(file_ref_for "${name}")" \
      "$(impact_for "${name}" "${status}")" \
      "${command_text}" \
      "$(remediation_for "${name}")" \
      "${note}" >> "${ISSUES_FILE}"
  fi
done < "${STATUS_FILE}"

src_count="$(find "${REPO_ROOT}/src" -type f 2>/dev/null | wc -l | tr -d ' ')"
tests_count="$(find "${REPO_ROOT}/tests" -type f 2>/dev/null | wc -l | tr -d ' ')"

callback_summary=""
if [[ -f "${RUN_DIR}/callback_budget_gate.log" ]]; then
  callback_summary="$(grep -E 'callbacks=.*max_ratio=.*budget_ms=' "${RUN_DIR}/callback_budget_gate.log" | tail -n 1 || true)"
fi

benchmark_excerpt=""
if [[ -f "${RUN_DIR}/bench_streaming.log" ]]; then
  benchmark_excerpt="$(grep -E 'x realtime' "${RUN_DIR}/bench_streaming.log" | tail -n 8 || true)"
fi

{
  echo "# Timestretch Architecture/Performance Review"
  echo
  echo "- Timestamp (UTC): $(date -u +"%Y-%m-%d %H:%M:%S")"
  echo "- Tier: \`${TIER}\`"
  echo "- Repository: \`${REPO_ROOT}\`"
  echo "- Skill Path: \`${SKILL_DIR}\`"
  echo "- Command Totals: total=${total_count}, pass=${pass_count}, fail=${fail_count}, skip=${skip_count}"
  echo
  echo "## Findings (severity-ordered)"
  if [[ -s "${ISSUES_FILE}" ]]; then
    for severity in P0 P1 P2 P3; do
      while IFS=$'\t' read -r sev name file_ref impact command_text remediation note; do
        [[ "${sev}" != "${severity}" ]] && continue
        if [[ -n "${note}" ]]; then
          echo "- ${sev} | file: ${file_ref} | impact: ${impact} (${note}) | evidence: \`${command_text}\` | remediation: ${remediation}"
        else
          echo "- ${sev} | file: ${file_ref} | impact: ${impact} | evidence: \`${command_text}\` | remediation: ${remediation}"
        fi
      done < "${ISSUES_FILE}"
    done
  else
    echo "- P3 | file: N/A | impact: no command-level failures or skips detected in this run | evidence: \`${STATUS_FILE}\` | remediation: maintain periodic runtime reviews and add manual code-level sampling when needed"
  fi
  echo
  echo "## Architecture Notes"
  echo "- Tier executed: \`${TIER}\`"
  echo "- Source files in scope: ${src_count}"
  echo "- Test files in scope: ${tests_count}"
  echo "- Command matrix: \`skills/timestretch-arch-perf-review/references/command-matrix.md\`"
  echo "- Rubric: \`skills/timestretch-arch-perf-review/references/review-rubric.md\`"
  echo
  echo "## Realtime Budget"
  echo "- \`realtime_dj_conditions\`: $(status_or_not_run realtime_dj_conditions)"
  echo "- \`realtime_allocations\`: $(status_or_not_run realtime_allocations)"
  echo "- \`callback_budget_gate\`: $(status_or_not_run callback_budget_gate)"
  if [[ -n "${callback_summary}" ]]; then
    echo "- Callback gate summary: \`${callback_summary}\`"
  else
    echo "- Callback gate summary: unavailable in this tier/run."
  fi
  echo
  echo "## Benchmark Snapshot"
  echo "- \`bench_streaming\`: $(status_or_not_run bench_streaming)"
  echo "- \`m0_baseline\`: $(status_or_not_run m0_baseline)"
  if [[ -n "${benchmark_excerpt}" ]]; then
    echo "- Recent throughput lines:"
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      echo "  - ${line}"
    done <<< "${benchmark_excerpt}"
  else
    echo "- Throughput excerpt: unavailable in this tier/run."
  fi
  echo
  echo "## Risks"
  if [[ "${fail_count}" -gt 0 ]]; then
    echo "- ${fail_count} command(s) failed; live-readiness confidence is reduced."
  else
    echo "- No failing commands in this run."
  fi
  if [[ "${skip_count}" -gt 0 ]]; then
    echo "- ${skip_count} command(s) skipped; coverage is incomplete until skipped checks run."
  else
    echo "- No skipped commands in this run."
  fi
  if [[ "${TIER}" != "deep" ]]; then
    echo "- Deep quality/baseline checks were not executed in this tier."
  fi
  echo
  echo "## Recommended Next Actions"
  action_index=1
  if [[ "$(status_for realtime_allocations)" == "FAIL" ]]; then
    echo "${action_index}. Remove realtime allocations and rerun \`realtime_allocations\`."
    action_index=$((action_index + 1))
  fi
  if [[ "$(status_for callback_budget_gate)" == "FAIL" ]]; then
    echo "${action_index}. Reduce callback work and rerun strict callback budget gate."
    action_index=$((action_index + 1))
  fi
  if [[ "$(status_for streaming_batch_parity)" == "FAIL" ]]; then
    echo "${action_index}. Investigate stream/batch state parity and rerun parity tests."
    action_index=$((action_index + 1))
  fi
  if [[ "$(status_for bench_streaming)" == "FAIL" ]]; then
    echo "${action_index}. Profile hot paths and rerun bench_streaming for headroom verification."
    action_index=$((action_index + 1))
  fi
  if [[ "$(status_for hybrid_subset_gate)" == "FAIL" ]]; then
    echo "${action_index}. Resolve hybrid subset quality divergence before trusting deep-tier parity conclusions."
    action_index=$((action_index + 1))
  fi
  if [[ "$(status_for m0_baseline)" == "FAIL" ]]; then
    echo "${action_index}. Investigate reference baseline failure (including timeout/dependency conditions) and rerun deep tier."
    action_index=$((action_index + 1))
  fi
  if [[ "$(status_for m0_baseline)" == "SKIP" ]]; then
    echo "${action_index}. Provision external corpus/dependencies and rerun deep tier baseline."
    action_index=$((action_index + 1))
  fi
  if [[ "${action_index}" -eq 1 ]]; then
    echo "1. Keep \`runtime\` tier as default for live-readiness checks; run \`deep\` only on explicit request."
    echo "2. Archive this report directory for longitudinal trend comparison."
  fi
} > "${REPORT_FILE}"

echo "${REPORT_FILE}"
