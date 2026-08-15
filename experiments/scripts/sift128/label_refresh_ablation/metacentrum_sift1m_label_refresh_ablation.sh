#!/bin/bash
#PBS -N batl_sift1m_label_refresh
#PBS -l select=1:ncpus=6:mem=24gb:scratch_local=50gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/sift1m_label_refresh_ablation.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Same-node screening A/B for C5. Both single-tree arms share the paper seed,
# optimizer, fixed ten-cycle schedule, data, CPU thread count, and evaluation
# sweep. Only training.label_refresh differs.

set -u

RUN_STATUS=0
TIMESTAMP=$(date +%Y%m%d_%H%M)
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_sift1m_label_refresh_${TIMESTAMP}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Copying BATL repo and SIFT1M data..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/sift configs results "$STORAGE_DIR"
cp /storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5 data/sift/

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl/bin/python"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export PYTHONPATH="$SCRATCHDIR/BATL"

SOURCE_CONFIG="$SCRATCHDIR/BATL/experiments/configs/sift1m/sift1m_h2_paper.yaml"
DATASET_PATH="$SCRATCHDIR/data/sift/sift-128-euclidean.hdf5"
CONFIG_DIR="$SCRATCHDIR/configs"
RESULT_ROOT="$SCRATCHDIR/results/sift1m_label_refresh_ablation"

"$PYTHON_EXEC" -u \
    "$SCRATCHDIR/BATL/experiments/scripts/sift128/label_refresh_ablation/write_configs.py" \
    --source-config "$SOURCE_CONFIG" \
    --dataset-path "$DATASET_PATH" \
    --output-root "$RESULT_ROOT" \
    --config-dir "$CONFIG_DIR"
CONFIG_STATUS=$?
if [ "$CONFIG_STATUS" -ne 0 ]; then
    echo "Config generation failed with status $CONFIG_STATUS" >&2
    exit "$CONFIG_STATUS"
fi

cd "$SCRATCHDIR/BATL"
ARMS=("per_cycle" "once")
for ARM in "${ARMS[@]}"; do
    BUILD_CONFIG="$CONFIG_DIR/${ARM}_build.yaml"
    SEARCH_CONFIG="$CONFIG_DIR/${ARM}_search.yaml"
    BUILD_DIR="$RESULT_ROOT/$ARM/build"
    SEARCH_DIR="$RESULT_ROOT/$ARM/search"
    INDEX_PATH="$RESULT_ROOT/$ARM/index_confidence.pkl"
    mkdir -p "$BUILD_DIR" "$SEARCH_DIR"

    echo "Building label_refresh=$ARM at $(date)"
    "$PYTHON_EXEC" -u build.py "$BUILD_CONFIG" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$BUILD_DIR" \
        --cycle-diagnostics \
        --cycle-diagnostics-queries 1000
    BUILD_STATUS=$?
    if [ "$BUILD_STATUS" -ne 0 ]; then
        echo "Build arm $ARM failed with status $BUILD_STATUS" >&2
        RUN_STATUS=$BUILD_STATUS
        break
    fi

    echo "Searching label_refresh=$ARM at $(date)"
    "$PYTHON_EXEC" -u search.py "$SEARCH_CONFIG" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SEARCH_DIR" \
        --num-leaves 10 20 40 80 100 \
        --n-queries 10000 \
        --batch-search 100
    SEARCH_STATUS=$?
    if [ "$SEARCH_STATUS" -ne 0 ]; then
        echo "Search arm $ARM failed with status $SEARCH_STATUS" >&2
        RUN_STATUS=$SEARCH_STATUS
        break
    fi
done

cp -r "$RESULT_ROOT/." "$STORAGE_DIR/" 2>/dev/null || true
cp "$CONFIG_DIR"/*.yaml "$STORAGE_DIR/" 2>/dev/null || true
echo "Done at $(date). Results: $STORAGE_DIR"
exit "$RUN_STATUS"
