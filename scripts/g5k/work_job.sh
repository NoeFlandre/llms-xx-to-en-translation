#!/usr/bin/env bash
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
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ARGS=(--shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT")

export HF_HOME="${WORK}/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT="${WORK}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${WORK}/inductor"
export TRITON_CACHE_DIR="${WORK}/triton"
export XDG_CACHE_HOME="${WORK}/xdg"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

cd "$SRC"
echo "node: $(hostname)  model: $MODEL  benchmark: ${BENCHMARK:-all}  dtype: $DTYPE  budget: ${BUDGET_MIN} min"
export GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
export NODE_NAME="$(hostname)"
export CODE_COMMIT="$(git -C "$SRC" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$GEN/bin/xx2en" work --model "$MODEL" --results-repo "$REPO" --work "$WORK" \
  --budget-min "$BUDGET_MIN" --dtype "$DTYPE" "${BENCH_ARGS[@]}" "${SHARD_ARGS[@]}"

if [ "$DTYPE" = "float16" ]; then
  echo "float16 GPU: scoring needs bf16 and is left for a bf16 node"
elif [ -x "$MET/bin/xx2en" ]; then
  "$MET/bin/xx2en" score --model "$MODEL" --results-repo "$REPO" --work "$WORK" \
    --metricx-repo "$HOME/src/metricx" --budget-min "$SCORE_BUDGET_MIN" "${BENCH_ARGS[@]}" "${SHARD_ARGS[@]}"
else
  echo "metrics environment missing at $MET: scoring skipped"
fi
