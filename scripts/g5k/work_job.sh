#!/usr/bin/env bash
# One tiny job (1 h walltime): translate the pending shards of one model and benchmark, then score them.
# Every shard is uploaded to the results dataset when it finishes, so a killed job loses at most one
# shard and the next job resumes from what exists. Float16 jobs (Turing GPUs) skip scoring, which
# needs bf16; a bf16 node scores those shards later.
#
#   oarsub -n xx2en-<model> -p "cluster='granite'" -l gpu=1,walltime=1:00 \
#     "MODEL=translategemma-4b-it BENCHMARK=flores_plus_devtest DTYPE=bfloat16 bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/work_job.sh"
set -euo pipefail

SRC="${SRC:-$HOME/src/llms-xx-to-en-translation}"
GEN="${GEN:-$HOME/venvs/xx2en}"
MET="${MET:-$HOME/venvs/metrics}"
MODEL="${MODEL:?set MODEL}"
BENCHMARK="${BENCHMARK:-}"
BUDGET_MIN="${BUDGET_MIN:-30}"
SCORE_BUDGET_MIN="${SCORE_BUDGET_MIN:-20}"
DTYPE="${DTYPE:-bfloat16}"
REPO="${RESULTS_REPO:-NoeFlandre/llms-xx-to-en-translation}"
JOB="${OAR_JOB_ID:-local}"
WORK="/tmp/xx2en-${JOB}"

BENCH_ARGS=()
if [ -n "$BENCHMARK" ]; then BENCH_ARGS=(--benchmark "$BENCHMARK"); fi

export HF_HOME="${WORK}/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

cd "$SRC"
echo "node: $(hostname)  model: $MODEL  benchmark: ${BENCHMARK:-all}  dtype: $DTYPE  budget: ${BUDGET_MIN} min"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$GEN/bin/xx2en" work --model "$MODEL" --results-repo "$REPO" --work "$WORK" \
  --budget-min "$BUDGET_MIN" --dtype "$DTYPE" "${BENCH_ARGS[@]}"

if [ "$DTYPE" = "float16" ]; then
  echo "float16 GPU: scoring needs bf16 and is left for a bf16 node"
elif [ -x "$MET/bin/xx2en" ]; then
  "$MET/bin/xx2en" score --model "$MODEL" --results-repo "$REPO" --work "$WORK" \
    --metricx-repo "$HOME/src/metricx" --budget-min "$SCORE_BUDGET_MIN" "${BENCH_ARGS[@]}"
else
  echo "metrics environment missing at $MET: scoring skipped"
fi
