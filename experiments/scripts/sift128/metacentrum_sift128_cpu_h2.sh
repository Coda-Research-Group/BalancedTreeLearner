#!/bin/bash
#PBS -N batl_sift128_cpu_h2
#PBS -l select=1:ncpus=6:mem=16gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail
trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT

H=2
NAME="sift1m_h${H}_paper"
THREADS=32

cd "$SCRATCHDIR"
echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"

cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/sift
cp /storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5 data/sift/

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON=/storage/brno2/home/jozefsprlak/conda/batl/bin/python

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export PYTHONPATH="$SCRATCHDIR/BATL"

CONFIG="$SCRATCHDIR/configs/${NAME}_cpu.yaml"
RESULT_DIR="$SCRATCHDIR/results/${NAME}_cpu"
mkdir -p "$(dirname "$CONFIG")" "$RESULT_DIR"

NAME=$NAME DATA="$SCRATCHDIR/data/sift/sift-128-euclidean.hdf5" \
SRC="$SCRATCHDIR/BATL/experiments/configs/sift1m/${NAME}.yaml" \
OUT="$CONFIG" RES="$SCRATCHDIR/results/${NAME}_cpu" \
$PYTHON -u - <<'PY'
import os, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())
cfg["experiment"]["name"] = f"{os.environ['NAME']}_cpu_metacentrum"
cfg["experiment"]["output_dir"] = os.environ["RES"]
cfg["dataset"]["path"] = os.environ["DATA"]
cfg["dataset"]["metric"] = "euclidean"
cfg["dataset"]["storage_mode"] = "preload"
cfg["training"]["batch_size"] = 8192
cfg["training"]["device"] = "cpu"
cfg["training"]["neighbor_search_backend"] = "faiss_cpu"
cfg["training"]["tree_update_cache_embeddings"] = False
cfg["evaluation"]["rerank_backend"] = "numpy_cpu"
Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cd "$SCRATCHDIR/BATL"
INDEX="$RESULT_DIR/index_confidence.pkl"

$PYTHON -u build.py "$CONFIG" \
    --log --index-path "$INDEX" --result-dir "$RESULT_DIR"

$PYTHON -u search.py "$CONFIG" \
    --log --index-path "$INDEX" --result-dir "$RESULT_DIR" \
    --num-leaves 10 20 40 80 100 --n-queries 10000 --batch-search 100

TS=$(date +%Y%m%d_%H%M)
OUT="/storage/brno2/home/jozefsprlak/results/batl_sift128_cpu_h${H}_${TS}"
mkdir -p "$OUT"
cp -r "$RESULT_DIR" "$OUT/"
cp "$CONFIG" "$OUT/"
echo "Done at $(date). Results: $OUT"
