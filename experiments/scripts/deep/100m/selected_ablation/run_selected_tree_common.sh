#!/bin/bash
# Build one independently seeded tree for one selected Deep100M ablation arm.
# Required: RESULT_NAME, TREE_INDEX, NUM_TREES, BRANCHING_FACTOR,
# CONVERGENCE_PATIENCE, TOP_K_NEIGHBORS, NEIGHBOR_SEARCH_SUBSET.

set -u

: "${RESULT_NAME:?RESULT_NAME is required}"
: "${TREE_INDEX:?TREE_INDEX is required}"
: "${NUM_TREES:?NUM_TREES is required}"
: "${BRANCHING_FACTOR:?BRANCHING_FACTOR is required}"
: "${CONVERGENCE_PATIENCE:?CONVERGENCE_PATIENCE is required}"
: "${TOP_K_NEIGHBORS:?TOP_K_NEIGHBORS is required}"
: "${NEIGHBOR_SEARCH_SUBSET:?NEIGHBOR_SEARCH_SUBSET is required}"

RUN_STATUS=0
MIN_TREES=2
BEAM_SIZE=100
NUM_LEAVES=100
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_trees"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Building ${RESULT_NAME}, tree ${TREE_INDEX}/${NUM_TREES}"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "results/$RESULT_NAME" "$STORAGE_TREE_DIR"

SOURCE_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
cp "$SOURCE_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SOURCE_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SOURCE_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing ablation build.")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing ablation build.")
print(torch.cuda.get_device_name(0))
PY

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/selected_ablation/write_selected_config.sh"
write_selected_config

cd "$SCRATCHDIR/BATL"
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
TREE_PATH="$RESULT_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
nvidia-smi

"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$RESULT_DIR" \
    --tree-index "$TREE_INDEX" \
    --batch-train 16384 \
    --batch-tree-update 8191 \
    --cycle-diagnostics \
    --cycle-diagnostics-queries 10000 \
    --cycle-diagnostics-loss-pairs 100000
RUN_STATUS=$?

cd "$SCRATCHDIR"
if [ "$RUN_STATUS" -eq 0 ] && [ -f "$TREE_PATH" ]; then
    cp "$TREE_PATH" "$STORAGE_TREE_DIR/"
    cp "$CONFIG_PATH" "$STORAGE_TREE_DIR/${RESULT_NAME}_tree_${TREE_INDEX}.yaml"
    mkdir -p "$STORAGE_TREE_DIR/tree_${TREE_INDEX}_artifacts"
    cp -r "$RESULT_DIR/." "$STORAGE_TREE_DIR/tree_${TREE_INDEX}_artifacts/" 2>/dev/null || true
else
    echo "Build failed or expected tree is missing: $TREE_PATH" >&2
    RUN_STATUS=1
fi

echo "Done at $(date)."
exit "$RUN_STATUS"
