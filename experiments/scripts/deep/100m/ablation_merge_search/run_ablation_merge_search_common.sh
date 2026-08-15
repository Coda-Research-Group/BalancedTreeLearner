#!/bin/bash
# Common body for Deep100M ablation merge-search PBS wrappers.
#
# Required variables from wrapper:
#   ABLATION_NAME, RESULT_NAME, STORAGE_TREE_DIR, BRANCHING_FACTOR,
#   EMBED_DIM, ALPHA, SEARCH_POINTS.

set -u

RUN_STATUS=0

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m results/"$RESULT_NAME"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load gcc/13
module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python stack..."
"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

print(f"faiss imported from: {faiss.__file__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is not visible to PyTorch; refusing search run.")
PY

cd "$SCRATCHDIR/BATL"
MERGED_INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"

for TREE_INDEX in 0 1 2 3; do
    TREE_PATH="$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
    if [ ! -f "$TREE_PATH" ]; then
        echo "Missing required tree index: $TREE_PATH" >&2
        exit 2
    fi
done

nvidia-smi
"$PYTHON_EXEC" -u merge_index.py \
    --output "$MERGED_INDEX_PATH" \
    "$STORAGE_TREE_DIR/index_confidence_tree_0.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_1.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_2.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_3.pkl"
RUN_STATUS=$?

write_point_config() {
    local point="$1"
    local point_result_name="${RESULT_NAME}_beam_${point}"
    local point_result_dir="$SCRATCHDIR/results/${RESULT_NAME}/beam_${point}"
    local config_path="$SCRATCHDIR/${point_result_name}.yaml"

    mkdir -p "$point_result_dir"
    cat > "$config_path" <<EOF
experiment:
  name: ${point_result_name}
  seed: 42
  output_dir: ${point_result_dir}
  tree_assignment_order: confidence
dataset:
  name: deep1b
  path: ${SCRATCHDIR}/data/deep100m
  base_path: ${SCRATCHDIR}/data/deep100m/base.fbin
  query_path: ${SCRATCHDIR}/data/deep100m/query.fbin
  ground_truth_path: ${SCRATCHDIR}/data/deep100m/groundtruth.ivecs
  source_name: Deep1B
  source_url: yandex
  metric: euclidean
  split: train
  subset_size: 100000000
  storage_mode: memmap
model:
  branching_factor: ${BRANCHING_FACTOR}
  tree_height: 2
  embedding_dim: 96
  encoder_hidden: 1024
  embed_dim: ${EMBED_DIM}
  num_decoder_layers: 1
  num_heads: 8
  ff_dim: 1024
  dropout: 0.1
  alpha: ${ALPHA}
  num_trees: 4
training:
  batch_size: 32768
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  alternating_interval: 2
  convergence_patience: 2
  convergence_min_delta: 0.005
  top_k_neighbors: 100
  neighbor_search_subset: 1000000
  neighbor_search_mode: random_subset
  neighbor_search_chunk_size: 1000000
  neighbor_search_backend: faiss_gpu
  tree_update_cache_embeddings: false
  device: cuda
evaluation:
  recall_at: [10]
  num_queries: 10000
  beam_size: ${point}
  num_leaves: [${point}]
  rerank_backend: numpy_cpu
EOF
    echo "$config_path"
}

if [ "$RUN_STATUS" -eq 0 ]; then
    for POINT in "${SEARCH_POINTS[@]}"; do
        CONFIG_PATH=$(write_point_config "$POINT")
        echo "Running ${RESULT_NAME}: beam_size=${POINT}, num_leaves=${POINT}"
        "$PYTHON_EXEC" -u search.py \
            "$CONFIG_PATH" \
            --log \
            --index-path "$MERGED_INDEX_PATH" \
            --result-dir "$SCRATCHDIR/results/${RESULT_NAME}/beam_${POINT}" \
            --num-leaves "$POINT" \
            --n-queries 10000 \
            --batch-search 0
        RUN_STATUS=$?
        if [ "$RUN_STATUS" -ne 0 ]; then
            echo "Search failed for ${RESULT_NAME} at beam_size=${POINT}" >&2
            break
        fi
    done
fi

cd "$SCRATCHDIR"
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying merged search results back to $OUTPUT_DIR"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$OUTPUT_DIR/"
fi
cp "$SCRATCHDIR"/*.yaml "$OUTPUT_DIR/" 2>/dev/null || true
exit "$RUN_STATUS"
