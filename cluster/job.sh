#!/bin/bash
# The SLURM job: starts Ollama on the GPU, then runs any mafialab command. Submit from ~/ABS:
#
#   cd ~/ABS && sbatch --dependency=singleton MafiaLab/cluster/job.sh experiment run chat
#   cd ~/ABS && sbatch --dependency=singleton MafiaLab/cluster/job.sh tournament --players 8 --games 30
#
# `tournament` and `experiment run` get --workers $WORKERS (default 16) unless the command sets
# --workers itself; Ollama serves that many requests at a time. Saved games are skipped, so a job
# stopped by the 1-day limit is simply submitted again. Every 10 minutes the log shows how much
# of each loaded model is on the GPU; if a model falls back to the CPU the job stops.
#SBATCH --job-name=mafialab
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/mafialab-%j.log
set -euo pipefail

ABS=${ABS:-$HOME/ABS}
WORKERS=${WORKERS:-16}
SIF=$ABS/containers/ollama-0.40.12.sif
JOB=${SLURM_JOB_ID:-local}
cd "$ABS/MafiaLab"

args=("$@")
if [[ ${1:-} == tournament || "${1:-} ${2:-}" == "experiment run" ]] \
  && [[ " $* " != *" --workers "* ]]; then
  args+=(--workers "$WORKERS")
fi

# a port of its own, so two jobs on one node don't share a server
export OLLAMA_HOST=127.0.0.1:$((20000 + ${SLURM_JOB_ID:-0} % 10000))
export OLLAMA_MODELS=$ABS/ollama-models OLLAMA_NUM_PARALLEL=$WORKERS
export OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=1h
export OLLAMA_BASE_URL=http://$OLLAMA_HOST
# the image sets its own OLLAMA_HOST; this prefix makes ours win inside the container
export SINGULARITYENV_OLLAMA_HOST=$OLLAMA_HOST
singularity exec --nv "$SIF" ollama serve \
  > "$ABS/results/ollama-$JOB.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 90); do curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null && break; sleep 2; done
curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null || { echo "Ollama did not start"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "mafialab ${args[*]} (code $(cat COMMIT 2>/dev/null || echo unknown))"

.venv/bin/mafialab "${args[@]}" &
MAIN=$!
# prints "<model> <share>% GPU" for every loaded model; exits 1 if a model is partly on the CPU
gpu_check() {
  .venv/bin/python - "$OLLAMA_BASE_URL/api/ps" <<'EOF'
import json, sys, urllib.request

models = json.load(urllib.request.urlopen(sys.argv[1]))["models"]
for m in models:
    print(f"{m['name']} {m['size_vram'] / m['size']:.0%} GPU", flush=True)
sys.exit(any(m["size_vram"] < m["size"] for m in models))
EOF
}
while kill -0 "$MAIN" 2>/dev/null; do
  # 10 minutes in 30 s steps, so a finished run doesn't keep the GPU waiting
  for _ in $(seq 20); do kill -0 "$MAIN" 2>/dev/null || break 2; sleep 30; done
  if ! gpu_check; then
    echo "A model runs partly on the CPU; stopping (the saved games are kept)."
    kill "$MAIN"
  fi
done
wait "$MAIN"
