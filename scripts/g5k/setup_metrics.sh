#!/usr/bin/env bash
# Metrics environment for COMET-22, MetricX-24 and GlotLID, on Python 3.12.
# unbabel-comet 2.2.7 pins numpy<2, and numpy 1.x has no Python 3.13 wheels, so this venv uses 3.12.
# uv downloads that interpreter. The generation venv (vLLM, Python 3.13) stays separate.
# Run inside an OAR job, not on a frontend:
#
#   oarsub -n xx2en-metrics-setup -l nodes=1,walltime=0:50 \
#     "bash $HOME/src/llms-xx-to-en-translation/scripts/g5k/setup_metrics.sh"
set -euo pipefail

SRC="${SRC:-$HOME/src/llms-xx-to-en-translation}"
VENV="${VENV:-$HOME/venvs/metrics}"
BOOT="${BOOT:-$HOME/venvs/uv-bootstrap}"
export UV_CACHE_DIR="/tmp/uv-cache-${USER}"

echo "node: $(hostname)  python: $(python3 --version)"
if [ ! -x "$BOOT/bin/uv" ]; then
  python3 -m venv "$BOOT"
  "$BOOT/bin/pip" install --quiet uv
fi
UV="$BOOT/bin/uv"

# Rebuild from scratch: this venv only ever holds metric packages, and an old 3.13 one must not survive.
rm -rf "$VENV"
"$UV" venv --python 3.12 "$VENV"

CONSTRAINTS="$VENV/constraints.txt"
printf 'setuptools<80\npyarrow>=16,<18\n' > "$CONSTRAINTS"
# torchmetrics (pinned by unbabel-comet) imports pkg_resources; setuptools<80 keeps it available.
"$UV" pip install --python "$VENV/bin/python" -c "$CONSTRAINTS" \
  "unbabel-comet==2.2.7" \
  "fasttext-turbo==0.1.1" \
  "huggingface_hub" \
  "datasets>=2.19,<3" \
  "sentencepiece" \
  "setuptools<80" \
  "pyarrow>=16,<18" \
  "accelerate>=0.26"  # transformers Trainer (used by MetricX predict) requires it
"$UV" pip install --python "$VENV/bin/python" -e "$SRC"
"$UV" pip freeze --python "$VENV/bin/python" > "$VENV/freeze.txt"
echo "installed: $(wc -l < "$VENV/freeze.txt") packages; freeze at $VENV/freeze.txt"
"$VENV/bin/python" -c "import comet, transformers, torch, numpy; print('comet ok; transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)"
