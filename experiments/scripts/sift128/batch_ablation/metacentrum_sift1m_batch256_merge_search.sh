#!/bin/bash
#PBS -N batl_sift1m_b256_merge
#PBS -l select=1:ncpus=6:mem=16gb:scratch_local=50gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/sift1m_batch256_merge_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Merge the four per-tree indexes and sweep. Run after all four tree jobs
# finish; it refuses rather than merging a partial ensemble, because a
# three-tree index would silently change the >=2-of-4 frequency filter and
# produce numbers that look valid and are not comparable to anything.

set -u

RUN_STATUS=0
ARM_NAME="batch256"
BATCH_SIZE=256
DROPOUT=0.0
RESULT_NAME="sift1m_h2_${ARM_NAME}"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_trees"
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
for TREE_INDEX in 0 1 2 3; do
    TREE_PATH="$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
    if [ ! -f "$TREE_PATH" ]; then
        echo "Missing tree index: $TREE_PATH" >&2
        echo "All four per-tree jobs must finish before the merge." >&2
        exit 2
    fi
done

cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/sift "results/$RESULT_NAME" "$STORAGE_DIR"
cp /storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5 data/sift/

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl/bin/python"

THREADS=6
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export PYTHONPATH="$SCRATCHDIR/BATL"

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
DATA_PATH="$SCRATCHDIR/data/sift/sift-128-euclidean.hdf5"
SRC_CONFIG="$SCRATCHDIR/BATL/experiments/configs/sift1m/sift1m_h2_paper.yaml"

source "$SCRATCHDIR/BATL/experiments/scripts/sift128/ablation/sift1m_ablation_config.sh"
write_ablation_config

cd BATL
MERGED_INDEX="$RESULT_DIR/index_confidence.pkl"

"$PYTHON_EXEC" -u merge_index.py \
    --output "$MERGED_INDEX" \
    "$STORAGE_TREE_DIR/index_confidence_tree_0.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_1.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_2.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_3.pkl"
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ]; then
    echo "Searching ${ARM_NAME}..."
    "$PYTHON_EXEC" -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$MERGED_INDEX" \
        --result-dir "$RESULT_DIR" \
        --num-leaves 10 20 40 80 100 \
        --n-queries 10000 \
        --batch-search 100
    RUN_STATUS=$?
fi

cd "$SCRATCHDIR"
if [ "$RUN_STATUS" -eq 0 ]; then
    echo "Copying results to $STORAGE_DIR"
    cp -r "results/${RESULT_NAME}/." "$STORAGE_DIR/" 2>/dev/null || true
    cp "$CONFIG_PATH" "$STORAGE_DIR/"
else
    echo "Merge or search failed; nothing copied out." >&2
    RUN_STATUS=${RUN_STATUS:-1}
fi
echo "Done at $(date)."
exit $RUN_STATUS
