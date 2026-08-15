#!/bin/bash
# Build phase of the Deep100M label ablation (SPEC_performance C4).
#
# Build only. The index is copied to persistent storage and search runs as a
# separate job, so a search-side failure never costs a 20-hour build and the
# sweep can be re-run without rebuilding.
#
# Required from wrapper: ARM_NAME, NEIGHBOR_SEARCH_SUBSET

set -u

RUN_STATUS=0
RESULT_NAME="deep100m_label_${ARM_NAME}"
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "results/$RESULT_NAME" "$STORAGE_DIR"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

# See the 2026-05-24 REVIEW in experiments/scripts/discussion.md.
export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python GPU stack..."
"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing label-ablation build.")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing label-ablation build.")
PY

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
write_label_config

cd BATL
INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"
echo "Building ${ARM_NAME} (neighbor_search_subset=${NEIGHBOR_SEARCH_SUBSET})..."
nvidia-smi

"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
    --batch-tree-update 32768 \
    --cycle-diagnostics \
    --cycle-diagnostics-queries 10000 \
    --cycle-diagnostics-loss-pairs 100000
RUN_STATUS=$?

cd "$SCRATCHDIR"
if [ "$RUN_STATUS" -eq 0 ] && [ -f "$INDEX_PATH" ]; then
    echo "Copying index and build artifacts to $STORAGE_DIR"
    cp "$INDEX_PATH" "$STORAGE_DIR/index_confidence.pkl"
    cp -r "results/${RESULT_NAME}/." "$STORAGE_DIR/" 2>/dev/null || true
    cp "$CONFIG_PATH" "$STORAGE_DIR/"
else
    echo "Build failed or index missing; nothing copied out." >&2
    RUN_STATUS=${RUN_STATUS:-1}
fi
exit $RUN_STATUS
