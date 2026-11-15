#!/usr/bin/env bash
set -euo pipefail
WORK="/tmp/xx2en-card-${OAR_JOB_ID:-local}"
export HF_HOME="$WORK/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export HF_HUB_DISABLE_XET=1
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
cd "$HOME/src/llms-xx-to-en-translation"
echo "node: $(hostname)  start: $(date)"
"$HOME/venvs/xx2en/bin/xx2en" card
echo "done: $(date)"
