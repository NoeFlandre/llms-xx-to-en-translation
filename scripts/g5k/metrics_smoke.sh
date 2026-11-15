#!/usr/bin/env bash
set -euo pipefail
JOB="${OAR_JOB_ID:-local}"
export WORK="/tmp/xx2en-smoke-${JOB}"
export HF_HOME="${WORK}/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export METRICX_REPO="$HOME/src/metricx"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$HOME/venvs/metrics/bin/xx2en" smoke-metrics
