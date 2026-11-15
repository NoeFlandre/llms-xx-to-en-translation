#!/usr/bin/env bash
set -uo pipefail
WORK="/tmp/xx2en-check-${OAR_JOB_ID:-local}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
export HF_HOME="$WORK/hf"
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
export HF_HUB_DISABLE_XET=1
cd "$HOME/src/llms-xx-to-en-translation" || exit 1
echo "node: $(hostname)  start: $(date)"
"$HOME/venvs/xx2en/bin/xx2en" check-shards
echo "done: $(date)"
