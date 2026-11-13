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
# Constraint applies to build-isolated sdist builds too: some sdists import pkg_resources at build time.
printf 'setuptools<80\npyarrow>=14,<17\n' > "$VENV/constraints.txt"
# MetricX's requirements.txt pins transformers 4.30.2 and sentencepiece 0.1.99, which conflict with
# COMET 2.2.7. The resolver's versions are used instead; the smoke scoring run verifies MetricX.
"$VENV/bin/pip" install --quiet -c "$VENV/constraints.txt" \
  "unbabel-comet==2.2.7" \
  "fasttext-turbo==0.1.1" \
  huggingface_hub \
  datasets \
  sentencepiece \
  "setuptools<80" \
  "pyarrow>=14,<17" \
  "datasets>=2.19,<3"  # numpy stays 1.x (torchmetrics 0.10.3 pins it); pyarrow must match
"$VENV/bin/pip" install --quiet -e "$SRC"
"$VENV/bin/pip" freeze > "$VENV/freeze.txt"
echo "installed: $(wc -l < "$VENV/freeze.txt") packages; freeze at $VENV/freeze.txt"
"$VENV/bin/python" -c "import comet, transformers, torch; print('comet ok; transformers', transformers.__version__, 'torch', torch.__version__)"
