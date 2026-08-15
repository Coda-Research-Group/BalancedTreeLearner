#!/bin/bash
# Merge the required ordered tree prefix and evaluate one selected arm.
# Required: RESULT_NAME, SOURCE_RESULT_NAME, NUM_TREES, MIN_TREES,
# BRANCHING_FACTOR, CONVERGENCE_PATIENCE, TOP_K_NEIGHBORS,
# NEIGHBOR_SEARCH_SUBSET.

set -u

: "${RESULT_NAME:?RESULT_NAME is required}"
: "${SOURCE_RESULT_NAME:?SOURCE_RESULT_NAME is required}"
: "${NUM_TREES:?NUM_TREES is required}"
: "${MIN_TREES:?MIN_TREES is required}"
: "${BRANCHING_FACTOR:?BRANCHING_FACTOR is required}"
: "${CONVERGENCE_PATIENCE:?CONVERGENCE_PATIENCE is required}"
: "${TOP_K_NEIGHBORS:?TOP_K_NEIGHBORS is required}"
: "${NEIGHBOR_SEARCH_SUBSET:?NEIGHBOR_SEARCH_SUBSET is required}"

RUN_STATUS=0
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${SOURCE_RESULT_NAME}_trees"
STORAGE_SEARCH_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_search"
TREE_PATHS=()

for ((TREE_INDEX = 0; TREE_INDEX < NUM_TREES; TREE_INDEX++)); do
    TREE_PATH="$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
    if [ ! -f "$TREE_PATH" ]; then
        echo "Missing required tree: $TREE_PATH" >&2
        echo "Refusing to evaluate a partial ensemble." >&2
        exit 2
    fi
    TREE_PATHS+=("$TREE_PATH")
done

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Searching ${RESULT_NAME} from ${SOURCE_RESULT_NAME}: T=${NUM_TREES}, min_trees=${MIN_TREES}"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "results/$RESULT_NAME" "$STORAGE_SEARCH_DIR"

SOURCE_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
cp "$SOURCE_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SOURCE_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SOURCE_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$SCRATCHDIR/BATL"

cd "$SCRATCHDIR/BATL"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
MERGED_INDEX="$RESULT_DIR/index_confidence.pkl"
nvidia-smi

"$PYTHON_EXEC" -u merge_index.py --output "$MERGED_INDEX" "${TREE_PATHS[@]}"
RUN_STATUS=$?

SEARCH_POINTS=(10 20 40 60 80 100 150 200 250 300)
SEARCH_RESULT_NAME="$RESULT_NAME"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/selected_ablation/write_selected_config.sh"

if [ "$RUN_STATUS" -eq 0 ]; then
    for POINT in "${SEARCH_POINTS[@]}"; do
        POINT_CONFIG="$SCRATCHDIR/${SEARCH_RESULT_NAME}_beam_${POINT}.yaml"
        POINT_DIR="$RESULT_DIR/beam_${POINT}"
        mkdir -p "$POINT_DIR" "$STORAGE_SEARCH_DIR/beam_${POINT}"

        RESULT_NAME="${SEARCH_RESULT_NAME}_beam_${POINT}" \
        CONFIG_PATH="$POINT_CONFIG" \
        BEAM_SIZE="$POINT" \
        NUM_LEAVES="$POINT" \
        MIN_TREES="$MIN_TREES" \
            write_selected_config
        cp "$POINT_CONFIG" "$POINT_DIR/config.yaml"

        echo "beam_size=${POINT}, num_leaves=${POINT}, min_trees=${MIN_TREES}"
        "$PYTHON_EXEC" -u search.py \
            "$POINT_CONFIG" \
            --log \
            --index-path "$MERGED_INDEX" \
            --result-dir "$POINT_DIR" \
            --batch-search 25
        POINT_STATUS=$?
        if [ "$POINT_STATUS" -ne 0 ]; then
            RUN_STATUS=$POINT_STATUS
            echo "Search point ${POINT} failed with status ${POINT_STATUS}." >&2
        fi
        cp -r "$POINT_DIR/." "$STORAGE_SEARCH_DIR/beam_${POINT}/" 2>/dev/null || true
    done
fi

echo "Done at $(date). Results: $STORAGE_SEARCH_DIR"
exit "$RUN_STATUS"
