#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-$HOME/src/llms-xx-to-en-translation}"
VENV="${VENV:-$HOME/venvs/xx2en}"
MODEL="${MODEL:?set MODEL}"
LANG_CODE="${LANG_CODE:-${LANG:?set LANG_CODE or LANG}}"
LIMIT="${LIMIT:-20}"
BENCH="${BENCH:-flores_plus_devtest}"
JOB="${OAR_JOB_ID:-local}"
WORK="/tmp/xx2en-${JOB}"

export HF_HOME="${WORK}/hf"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /tmp | tail -1

"$VENV/bin/xx2en" run \
  --model "$MODEL" \
  --benchmarks "$BENCH" \
  --langs "$LANG_CODE" \
  --limit "$LIMIT" \
  --backend vllm \
  --configs "$SRC/configs" \
  --inventory "$SRC/data/benchmark_inventory.json" \
  --out "$WORK/out" \
  --cache "$WORK/cache"

RESULTS="$HOME/results/${JOB}"
mkdir -p "$RESULTS"
cp -r "$WORK/out" "$RESULTS/"
echo "results kept in $RESULTS (small files only); /tmp scratch removed"
