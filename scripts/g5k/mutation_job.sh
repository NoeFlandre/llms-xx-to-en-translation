#!/usr/bin/env bash
set -uo pipefail
WORK="/tmp/xx2en-mut-${OAR_JOB_ID:-local}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export UV_CACHE_DIR="$WORK/uv"
export UV_PROJECT_ENVIRONMENT="$WORK/venv"
python3 -m pip install -q --target "$WORK/bin" uv || { echo "uv install failed"; exit 1; }
export PATH="$WORK/bin/bin:$PATH"
cd "$HOME/src/llms-xx-to-en-translation" || exit 1
if [ "${FRESH:-0}" = "1" ]; then rm -rf .mutmut-cache mutants; fi
echo "node: $(hostname)  start: $(date)"
uv sync --frozen --group dev --no-install-project || { echo "uv sync failed"; exit 1; }
timeout 2640 uv run mutmut run
rc=$?
echo "mutmut exit: $rc at $(date)"
uv run mutmut results > "$HOME/mutmut-results-${OAR_JOB_ID:-local}.txt" 2>&1
echo "results saved"
if [ "$rc" = "124" ]; then
  ssh -o BatchMode=yes nancy "oarsub -n xx2en-mutmut -p 'gpu_count=0' -l nodes=1,walltime=1:00 'bash \$HOME/mutation_job.sh'" | grep -E "OAR_JOB_ID|Error" || echo "resubmit failed"
fi
