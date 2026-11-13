#!/usr/bin/env bash
# Tiny metrics smoke job (30 min): one GPU, two toy segments through COMET-22, MetricX-24 and GlotLID.
#   oarsub -n xx2en-metrics-smoke -p "cluster='granite'" -l gpu=1,walltime=0:30 "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/metrics_smoke.sh"
set -euo pipefail
JOB="${OAR_JOB_ID:-local}"
export WORK="/tmp/xx2en-smoke-${JOB}"
export HF_HOME="${WORK}/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export METRICX_REPO="$HOME/src/metricx"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$HOME/venvs/metrics/bin/python" "$HOME/src/llms-xx-to-en-translation/scripts/g5k/metrics_smoke.py"
