#!/bin/bash
# One-time setup on the cluster's login node, after `bash cluster/sync.sh push` from the laptop:
#
#   bash ~/ABS/MafiaLab/cluster/setup.sh
#
# Installs uv and the Python environment, pulls the Ollama image, and downloads every Ollama
# model that configs/tournament.yaml and configs/experiments.yaml use (the GPU nodes are offline).
# Everything goes under ~/ABS. Safe to run again: finished steps are skipped, new models pulled.
set -euo pipefail

ABS=${ABS:-$HOME/ABS}
OLLAMA_VERSION=0.40.12
SIF=$ABS/containers/ollama-$OLLAMA_VERSION.sif
mkdir -p "$ABS"/{uv,containers,ollama-home,ollama-models,results,tmp/mafialab-setup}

# uv, its Python and its cache stay in ~/ABS, and the shell profile is left alone
export UV_INSTALL_DIR=$ABS/uv/bin UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR=$ABS/uv/cache UV_PYTHON_INSTALL_DIR=$ABS/uv/python
if [[ ! -x $ABS/uv/bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd "$ABS/MafiaLab"
[[ -x .venv/bin/python ]] || "$ABS/uv/bin/uv" venv --python 3.14 .venv
"$ABS/uv/bin/uv" pip install --python .venv/bin/python -e .

if [[ ! -f $SIF ]]; then
  export SINGULARITY_CACHEDIR=$ABS/tmp/mafialab-setup SINGULARITY_TMPDIR=$ABS/tmp/mafialab-setup
  singularity pull "$SIF" "docker://ollama/ollama:$OLLAMA_VERSION"
fi

# the Ollama model ids behind every alias the configs use
mapfile -t MODELS < <(.venv/bin/python - <<'EOF'
from mafialab.experiments import load_config as experiments
from mafialab.llm.registry import load_models
from mafialab.tournament import load_config as tournament

t, e, m = tournament(), experiments(), load_models()
aliases = {*t["models"], *t["backgrounds"], *(a for block in e.values() for a in block["models"])}
print("\n".join(sorted({m.get(a).model for a in aliases if m.get(a).provider == "ollama"})))
EOF
)

# a private Ollama server on the login node, only to download the models
export OLLAMA_HOST=127.0.0.1:$((20000 + RANDOM % 10000)) OLLAMA_MODELS=$ABS/ollama-models
singularity exec --home "$ABS/ollama-home" "$SIF" ollama serve > "$ABS/results/setup-ollama.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 60); do curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null && break; sleep 2; done
for model in "${MODELS[@]}"; do
  singularity exec --home "$ABS/ollama-home" "$SIF" ollama pull "$model"
done
# compare these IDs with `ollama list` on the laptop; a different ID is a different build
singularity exec --home "$ABS/ollama-home" "$SIF" ollama list | tee "$ABS/results/ollama_models.txt"

rm -rf "$ABS/tmp/mafialab-setup"
echo "Setup done. Models: ${MODELS[*]}"
