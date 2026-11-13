#!/usr/bin/env bash
# Create or refresh the project virtualenv on a Grid'5000 node.
# Run inside an OAR job (not on a frontend): installing vLLM is heavy.
#
#   oarsub -n xx2en-setup -l nodes=1,walltime=0:45 "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/setup_env.sh"
set -euo pipefail

SRC="${SRC:-$HOME/src/llms-xx-to-en-translation}"
VENV="${VENV:-$HOME/venvs/xx2en}"
export PIP_CACHE_DIR="/tmp/pip-cache-${USER}"

echo "node: $(hostname)  python: $(python3 --version)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "$SRC[gpu]"
"$VENV/bin/pip" freeze > "$VENV/freeze.txt"
echo "installed: $(wc -l < "$VENV/freeze.txt") packages; record in dataset card: $VENV/freeze.txt"
"$VENV/bin/python" -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
