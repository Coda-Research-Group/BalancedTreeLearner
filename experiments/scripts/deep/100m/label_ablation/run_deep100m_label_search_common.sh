#!/bin/bash
# Search phase of the Deep100M label ablation (SPEC_performance C4).
#
# Reads the index the build job persisted. Cheap and re-runnable: a failed or
# reconfigured sweep never costs a rebuild.
#
# Required from wrapper: ARM_NAME, NEIGHBOR_SEARCH_SUBSET

set -u

RUN_STATUS=0
RESULT_NAME="deep100m_label_${ARM_NAME}"
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}"
CACHED_INDEX="$STORAGE_DIR/index_confidence.pkl"

if [ ! -f "$CACHED_INDEX" ]; then
    echo "Missing index from the build job: $CACHED_INDEX" >&2
    exit 2
fi

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "results/$RESULT_NAME"
cp "$CACHED_INDEX" "results/${RESULT_NAME}/index_confidence.pkl"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
write_label_config

cd BATL
echo "Searching ${ARM_NAME}..."
nvidia-smi
"$PYTHON_EXEC" -u search.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl" \
    --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
    --num-leaves 10 40 80 100 \
    --n-queries 10000 \
    --batch-search 2000
RUN_STATUS=$?

cd "$SCRATCHDIR"
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="${STORAGE_DIR}_search_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"
echo "Copying search results to $OUTPUT_DIR"
cp -r "results/${RESULT_NAME}/." "$OUTPUT_DIR/" 2>/dev/null || true
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
