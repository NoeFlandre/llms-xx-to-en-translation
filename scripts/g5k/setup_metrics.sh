#!/usr/bin/env bash
# Metrics environment for COMET-22, MetricX-24 and GlotLID.
# Kept separate from the generation venv: MetricX pins transformers 4.30.2, which vLLM cannot run with.
# Run inside an OAR job, not on a frontend:
#
#   oarsub -n xx2en-metrics-setup -l nodes=1,walltime=0:45 \
#     "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/setup_metrics.sh"
set -euo pipefail

SRC="${SRC:-$HOME/src/llms-xx-to-en-translation}"
VENV="${VENV:-$HOME/venvs/metrics}"
export PIP_CACHE_DIR="/tmp/pip-cache-${USER}"

echo "node: $(hostname)  python: $(python3 --version)"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
  "unbabel-comet==2.2.7" \
  "fasttext-turbo==0.1.1" \
  "huggingface_hub" \
  "transformers==4.30.2" \
  "sentencepiece==0.1.99" \
  "datasets==2.13.1"
"$VENV/bin/pip" install --quiet -e "$SRC"
"$VENV/bin/pip" freeze > "$VENV/freeze.txt"
echo "installed: $(wc -l < "$VENV/freeze.txt") packages; freeze at $VENV/freeze.txt"
"$VENV/bin/python" -c "import comet, transformers, torch; print('comet ok; transformers', transformers.__version__, 'torch', torch.__version__)"
