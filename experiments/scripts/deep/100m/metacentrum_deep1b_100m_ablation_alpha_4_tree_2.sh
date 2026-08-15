#!/bin/bash
#PBS -N batl_deep100m_alpha_4_t2
#PBS -l select=1:ncpus=4:ngpus=1:gpu_mem=16gb:mem=250gb:scratch_ssd=250gb
#PBS -l walltime=16:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_alpha_4_tree_2.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u
RUN_STATUS=0
TREE_INDEX=2
ABLATION_NAME="deep100m_ablation_alpha_4"
RUN_NAME="${ABLATION_NAME}_tree_${TREE_INDEX}"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${ABLATION_NAME}_parallel_trees"
BRANCHING_FACTOR=256
EMBED_DIM=256
ALPHA=4.0

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "$STORAGE_TREE_DIR"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python GPU stack..."
"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is not visible to PyTorch; refusing ablation run.")

print(f"faiss_gpu_available: {hasattr(faiss, 'StandardGpuResources')}")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing ablation run.")
PY

CONFIG_PATH="$SCRATCHDIR/${RUN_NAME}.yaml"
RESULT_DIR="$SCRATCHDIR/results/${RUN_NAME}"
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
mkdir -p "$RESULT_DIR"

cat > "$CONFIG_PATH" <<EOF
experiment:
  name: ${RUN_NAME}
  seed: 42
  output_dir: ${RESULT_DIR}
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
  beam_size: 100
  num_leaves: [10, 40, 80, 100]
  rerank_backend: numpy_cpu
EOF

cd "$SCRATCHDIR/BATL"
INDEX_PATH="$RESULT_DIR/index_confidence.pkl"
TREE_INDEX_PATH="$RESULT_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
echo "Running ${RUN_NAME}: tree_index=${TREE_INDEX}, K=${BRANCHING_FACTOR}, embed_dim=${EMBED_DIM}, alpha=${ALPHA}"
nvidia-smi

"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$RESULT_DIR" \
    --batch-train 32768 \
    --batch-tree-update 8191 \
    --tree-index "$TREE_INDEX"
RUN_STATUS=$?

cd "$SCRATCHDIR"
echo "Copying ${RUN_NAME} artifacts to $STORAGE_TREE_DIR"
if [ -f "$TREE_INDEX_PATH" ]; then
    cp "$TREE_INDEX_PATH" "$STORAGE_TREE_DIR/"
else
    echo "Missing expected tree index: $TREE_INDEX_PATH"
fi
cp "$CONFIG_PATH" "$STORAGE_TREE_DIR/${RUN_NAME}.yaml" 2>/dev/null || true
if [ -d "results/${RUN_NAME}" ]; then
    cp -r "results/${RUN_NAME}" "$STORAGE_TREE_DIR/"
fi
exit "$RUN_STATUS"
