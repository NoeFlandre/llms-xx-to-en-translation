#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/src/llms-xx-to-en-translation" || exit 1
echo "node: $(hostname)  start: $(date)"
for s in nancy sophia rennes grenoble; do
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$s" true || echo "ssh to $s failed"
done
timeout 2820 "$HOME/venvs/xx2en/bin/xx2en" dispatch >> "$HOME/dispatch.log" 2>&1
echo "dispatcher stopped: $(date)"
if tail -n 3 "$HOME/dispatch.log" | grep -q "all shards and scores are in the dataset"; then
  echo "dataset complete; not resubmitting"
  exit 0
fi
for try in $(seq 1 15); do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 nancy "oarsub -n xx2en-dispatch -p 'gpu_count=0' -l nodes=1,walltime=1:00 'bash \$HOME/dispatch_job.sh'" 2>&1)
  if echo "$out" | grep -q OAR_JOB_ID; then
    echo "resubmitted: $(echo "$out" | grep OAR_JOB_ID)"
    exit 0
  fi
  echo "resubmit try $try failed: $(echo "$out" | tail -n 1)"
  sleep 30
done
echo "resubmit gave up"
