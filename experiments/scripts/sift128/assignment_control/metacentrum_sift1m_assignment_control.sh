#!/bin/bash
#PBS -N batl_sift1m_assignment_control
#PBS -l select=1:ncpus=6:mem=24gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/sift1m_assignment_control.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail
: "${SOURCE_INDEX:?Submit with -v SOURCE_INDEX=/storage/.../index_confidence.pkl}"
if [ ! -f "$SOURCE_INDEX" ]; then
    echo "Source round-confidence index not found: $SOURCE_INDEX" >&2
    exit 1
fi

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Source index: $SOURCE_INDEX"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/sift input results configs
cp /storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5 data/sift/
cp "$SOURCE_INDEX" input/index_confidence.pkl

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON_EXEC=/storage/brno2/home/jozefsprlak/conda/batl/bin/python
# Must match the run that produced SOURCE_INDEX, not the ncpus request. The
# harness requires round-confidence to reproduce the cached tree bit-for-bit,
# and BLAS reduction order depends on the thread count: with a different one,
# probabilities differ in the last bits and assignments flip wherever two
# branches are near-tied at a capacity boundary. metacentrum_sift128_cpu_h2.sh
# uses 32 (oversubscribed against its 6 cpus), so this must too.
THREADS=32
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export PYTHONPATH="$SCRATCHDIR/BATL"

BASE_CONFIG="$SCRATCHDIR/configs/sift1m_assignment_control.yaml"
CONTROL_DIR="$SCRATCHDIR/results/sift1m_assignment_control"
SOURCE_CONFIG="$SCRATCHDIR/BATL/experiments/configs/sift1m/sift1m_h2_paper.yaml"

SRC="$SOURCE_CONFIG" OUT="$BASE_CONFIG" RES="$CONTROL_DIR" \
DATA="$SCRATCHDIR/data/sift/sift-128-euclidean.hdf5" \
"$PYTHON_EXEC" -u - <<'PY'
import os
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())
cfg["experiment"]["name"] = "sift1m_assignment_control"
cfg["experiment"]["output_dir"] = os.environ["RES"]
cfg["dataset"]["path"] = os.environ["DATA"]
cfg["dataset"]["metric"] = "euclidean"
cfg["dataset"]["storage_mode"] = "preload"
cfg["training"]["device"] = "cpu"
cfg["training"]["neighbor_search_backend"] = "faiss_cpu"
cfg["training"]["tree_update_cache_embeddings"] = False
cfg["training"]["tree_update_top_r"] = None
cfg["evaluation"]["rerank_backend"] = "numpy_cpu"
cfg["evaluation"]["tree_assignment_mode"] = "round"
cfg["evaluation"]["tree_assignment_order"] = "confidence"
Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cd "$SCRATCHDIR/BATL"
"$PYTHON_EXEC" -u experiments/scripts/sift128/run_assignment_control.py \
    "$BASE_CONFIG" \
    --log \
    --source-index "$SCRATCHDIR/input/index_confidence.pkl" \
    --output-dir "$CONTROL_DIR"

ARMS=(
    "round_confidence:round:confidence"
    "round_input:round:input"
    "sequential_input:sequential:input"
    "sequential_confidence:sequential:confidence"
)

for ARM_SPEC in "${ARMS[@]}"; do
    IFS=: read -r ARM MODE ORDER <<< "$ARM_SPEC"
    ARM_CONFIG="$SCRATCHDIR/configs/${ARM}.yaml"
    ARM_RESULT_DIR="$CONTROL_DIR/$ARM/search"
    if [ "$MODE" = "round" ]; then
        ARM_INDEX="$CONTROL_DIR/$ARM/index_${ORDER}.pkl"
    else
        ARM_INDEX="$CONTROL_DIR/$ARM/index_${MODE}_${ORDER}.pkl"
    fi

    SRC="$BASE_CONFIG" OUT="$ARM_CONFIG" RES="$ARM_RESULT_DIR" \
    ARM="$ARM" MODE="$MODE" ORDER="$ORDER" \
    "$PYTHON_EXEC" -u - <<'PY'
import os
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())
cfg["experiment"]["name"] = f"sift1m_assignment_{os.environ['ARM']}"
cfg["experiment"]["output_dir"] = os.environ["RES"]
cfg["evaluation"]["tree_assignment_mode"] = os.environ["MODE"]
cfg["evaluation"]["tree_assignment_order"] = os.environ["ORDER"]
Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

    "$PYTHON_EXEC" -u search.py "$ARM_CONFIG" \
        --log \
        --index-path "$ARM_INDEX" \
        --result-dir "$ARM_RESULT_DIR" \
        --num-leaves 10 20 40 80 100 \
        --n-queries 10000 \
        --batch-search 100
done

TIMESTAMP=$(date +%Y%m%d_%H%M)
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_sift1m_assignment_control_${TIMESTAMP}"
mkdir -p "$STORAGE_DIR"
cp -r "$CONTROL_DIR/." "$STORAGE_DIR/"
cp "$BASE_CONFIG" "$STORAGE_DIR/"
# One glob, not one per mode. Under `set -e` a glob that matches nothing is a
# fatal cp error, so the per-mode versions killed the script whenever the ARMS
# list was subsetted to re-run a single arm — after the results had been copied,
# so the job looked failed while its output was actually fine.
cp "$SCRATCHDIR"/configs/*.yaml "$STORAGE_DIR/"
echo "Done at $(date). Results: $STORAGE_DIR"
