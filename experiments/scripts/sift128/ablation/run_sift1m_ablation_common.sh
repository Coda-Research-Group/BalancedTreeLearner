#!/bin/bash
# Build + search for one arm of a SIFT1M one-knob ablation.
#
# SIFT1M builds in well under an hour on CPU, so unlike the Deep100M ablation
# there is nothing to gain from splitting build and search into separate jobs.
#
# Required from wrapper: ARM_NAME, DROPOUT, BATCH_SIZE

set -u

RUN_STATUS=0
RESULT_NAME="sift1m_h2_${ARM_NAME}"
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/sift "results/$RESULT_NAME" "$STORAGE_DIR"
cp /storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5 data/sift/

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl/bin/python"

# Threads match the ncpus the wrapper requests. The older
# metacentrum_sift128_cpu_h*.sh scripts ask for 6 cpus and then set 32
# threads; that oversubscription does not change recall, but it makes wall
# times meaningless, and these arms are compared on recall-per-bucket.
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
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
echo "Building ${ARM_NAME} (dropout=${DROPOUT}, batch_size=${BATCH_SIZE})..."

"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$RESULT_DIR" \
    --cycle-diagnostics \
    --cycle-diagnostics-queries 10000 \
    --cycle-diagnostics-loss-pairs 100000
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ] && [ -f "$INDEX_PATH" ]; then
    echo "Searching ${ARM_NAME}..."
    "$PYTHON_EXEC" -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
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
    echo "Run failed; nothing copied out." >&2
    RUN_STATUS=${RUN_STATUS:-1}
fi
echo "Done at $(date)."
exit $RUN_STATUS
