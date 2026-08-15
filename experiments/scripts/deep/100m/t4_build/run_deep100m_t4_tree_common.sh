#!/bin/bash
# Build ONE tree of the Deep100M T=4 headline ensemble.
#
# Why this exists. The label-ablation arms are `num_trees: 1` so a 100M build
# fits in one job, but the >=2-of-4 frequency filter is inactive at T=1, so
# their recall is not comparable to the paper's Table 1 — the July T=4 baseline
# reached 0.908 at 125k candidates where the T=1 exact arm reaches 0.7475 at
# 153k. Ablation numbers, not headline numbers. This build produces an index
# that can carry a headline claim, and can serve as the source for a T=4
# assignment control.
#
# Why per-tree. Four trees in one job is 4x a build that already took 3.7h for
# the 1% arm, and a walltime kill would cost all of it. `build.py --tree-index N`
# seeds with `cfg.seed + N` and calls set_seed before constructing the model,
# exactly as the sequential loop does, so the trees are the ones a single job
# would have produced. A tree that runs out of walltime costs one tree.
#
# Labels are the paper-faithful 1% subset: BLISS §4.2 states the convention,
# BATL §3.2.3 adopts it by reference, and C4 measured it as the best of the
# three subsets at Deep100M. It is also by far the cheapest — the exact arm
# spent 9.2h of its 16.1h build on mining alone.
#
# Required from wrapper: TREE_INDEX

set -u

RUN_STATUS=0
RESULT_NAME="deep100m_t4"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_trees"

# The headline ensemble, and the label subset the paper's convention implies.
NUM_TREES=4
NEIGHBOR_SEARCH_SUBSET=1000000

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Building Deep100M T=4 tree ${TREE_INDEX} of ${NUM_TREES}"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "results/$RESULT_NAME" "$STORAGE_TREE_DIR"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Copying Deep100M data..."
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
    raise SystemExit("CUDA is not visible to PyTorch; refusing T=4 build.")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing T=4 build.")
PY

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
write_label_config

cd BATL
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
TREE_PATH="$RESULT_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
nvidia-smi

"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$RESULT_DIR" \
    --tree-index "$TREE_INDEX" \
    --batch-tree-update 8191 \
    --cycle-diagnostics \
    --cycle-diagnostics-queries 10000 \
    --cycle-diagnostics-loss-pairs 100000
RUN_STATUS=$?

cd "$SCRATCHDIR"
if [ "$RUN_STATUS" -eq 0 ] && [ -f "$TREE_PATH" ]; then
    echo "Copying tree ${TREE_INDEX} to $STORAGE_TREE_DIR"
    cp "$TREE_PATH" "$STORAGE_TREE_DIR/"
    cp "$CONFIG_PATH" "$STORAGE_TREE_DIR/"
    cp -r "results/${RESULT_NAME}/." "$STORAGE_TREE_DIR/tree_${TREE_INDEX}_artifacts/" 2>/dev/null || true
else
    echo "Tree ${TREE_INDEX} build failed or index missing; nothing copied out." >&2
    RUN_STATUS=${RUN_STATUS:-1}
fi
echo "Done at $(date)."
exit $RUN_STATUS
