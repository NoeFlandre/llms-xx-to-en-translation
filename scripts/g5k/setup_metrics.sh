#!/usr/bin/env bash
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

rm -rf "$VENV"
"$UV" venv --python 3.12 "$VENV"

CONSTRAINTS="$VENV/constraints.txt"
printf 'setuptools<80\npyarrow>=16,<18\n' > "$CONSTRAINTS"
"$UV" pip install --python "$VENV/bin/python" -c "$CONSTRAINTS" \
  "unbabel-comet==2.2.7" \
  "fasttext-turbo==0.1.1" \
  "huggingface_hub" \
  "datasets>=2.19,<3" \
  "sentencepiece" \
  "setuptools<80" \
  "pyarrow>=16,<18" \
  "accelerate>=0.26"
"$UV" pip install --python "$VENV/bin/python" -e "$SRC"
"$UV" pip freeze --python "$VENV/bin/python" > "$VENV/freeze.txt"
echo "installed: $(wc -l < "$VENV/freeze.txt") packages; freeze at $VENV/freeze.txt"
"$VENV/bin/python" -c "import comet, transformers, torch, numpy; print('comet ok; transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)"
