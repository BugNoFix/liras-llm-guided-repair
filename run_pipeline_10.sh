#!/usr/bin/env bash

set -u

CONFIG_PATH="${1:-config.json}"
RUNS="${RUNS:-20}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENARIOS=(
  "NL_Specification_1.txt"
  "NL_Specification_2.txt"
  "NL_Specification_3.txt"
)

successes=0
failures=0
total_runs=$((RUNS * ${#SCENARIOS[@]}))
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/liras_pipeline_batch.XXXXXX")"

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

echo "[RUN_BATCH] config=${CONFIG_PATH} scenarios=${#SCENARIOS[@]} runs_per_scenario=${RUNS} total_runs=${total_runs}"

for scenario in "${SCENARIOS[@]}"; do
  scenario_config="${tmp_dir}/${scenario%.txt}.config.json"
  "${PYTHON_BIN}" - "${CONFIG_PATH}" "${scenario}" "${scenario_config}" <<'PY'
import json
import sys

config_path, scenario, out_path = sys.argv[1:4]
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
config["scenario"] = scenario
if scenario == "NL_Specification_3.txt":
    config["lira_cli_jar"] = "liras-cli-layout2.jar"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
PY
  lira_cli_jar="$("${PYTHON_BIN}" - "${scenario_config}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("lira_cli_jar"))
PY
)"
  echo "[RUN_BATCH] scenario=${scenario} lira_cli_jar=${lira_cli_jar}"

  for i in $(seq 1 "${RUNS}"); do
    echo
    echo "=== PIPELINE RUN scenario=${scenario} run=${i}/${RUNS} ==="
    "${PYTHON_BIN}" pipeline_runner.py --config "${scenario_config}"
    exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
      successes=$((successes + 1))
      echo "[RUN_BATCH] scenario=${scenario} run=${i} status=ok"
    else
      failures=$((failures + 1))
      echo "[RUN_BATCH] scenario=${scenario} run=${i} status=failed exit_code=${exit_code}"
    fi
  done
done

echo
echo "[RUN_BATCH] completed total=${total_runs} successes=${successes} failures=${failures}"

if [ "${failures}" -gt 0 ]; then
  exit 1
fi
