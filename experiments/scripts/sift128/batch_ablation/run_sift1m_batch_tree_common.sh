#!/bin/bash
# Build ONE ensemble tree of a SIFT1M batch-size arm (tree-quality suspect S0).
#
# Why per-tree. At batch 256 there are 3,907 optimizer steps per epoch against
# 123 at 8,192 — 31.8x more — and small batches use the CPU far less
# efficiently, so a four-tree build in one job is the shape that already died
# on walltime once (job 22764327, killed during tree 3 of 4). `build.py
# --tree-index N` seeds with `cfg.seed + N` and calls `set_seed` before the
# model is created, exactly as the sequential loop does, so splitting is not an
# approximation: the trees are the same ones the single job would have built.
#
# A tree that runs out of walltime costs one tree, not the run.
#
# Required from wrapper: TREE_INDEX, BATCH_SIZE, ARM_NAME

set -u

RUN_STATUS=0
RESULT_NAME="sift1m_h2_${ARM_NAME}"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_trees"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Arm ${ARM_NAME}, tree ${TREE_INDEX}, batch_size ${BATCH_SIZE}"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/sift "results/$RESULT_NAME" "$STORAGE_TREE_DIR"
cp /storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5 data/sift/

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl/bin/python"

# Matches run_sift1m_ablation_common.sh, which produced the batch-8192 arm
# this is compared against.
THREADS=6
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export PYTHONPATH="$SCRATCHDIR/BATL"

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
DATA_PATH="$SCRATCHDIR/data/sift/sift-128-euclidean.hdf5"
SRC_CONFIG="$SCRATCHDIR/BATL/experiments/configs/sift1m/sift1m_h2_paper.yaml"
# Same writer as the dropout arms, so the only difference between this arm and
# the completed batch-8192 one is BATCH_SIZE.
DROPOUT=0.0

source "$SCRATCHDIR/BATL/experiments/scripts/sift128/ablation/sift1m_ablation_config.sh"
write_ablation_config

cd BATL
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
TREE_PATH="$RESULT_DIR/index_confidence_tree_${TREE_INDEX}.pkl"

"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$RESULT_DIR" \
    --tree-index "$TREE_INDEX" \
    --cycle-diagnostics \
    --cycle-diagnostics-queries 10000 \
    --cycle-diagnostics-loss-pairs 100000
RUN_STATUS=$?

cd "$SCRATCHDIR"
if [ "$RUN_STATUS" -eq 0 ] && [ -f "$TREE_PATH" ]; then
    echo "Copying tree ${TREE_INDEX} to $STORAGE_TREE_DIR"
    cp "$TREE_PATH" "$STORAGE_TREE_DIR/"
    cp "$CONFIG_PATH" "$STORAGE_TREE_DIR/"
else
    echo "Tree ${TREE_INDEX} build failed or index missing; nothing copied out." >&2
    RUN_STATUS=${RUN_STATUS:-1}
fi
echo "Done at $(date)."
exit $RUN_STATUS
