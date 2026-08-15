#!/bin/bash
#PBS -N batl_deep100m_t1_top1_bucket
# Build one standard Deep100M BATL tree, then evaluate recall using only the
# model's top-1 predicted bucket. This is a diagnostic T=1 run, not the
# paper-faithful T=4 ensemble.
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=18:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_t1_top1_bucket.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

RESULT_NAME="deep100m_t1_top1_bucket"
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "$STORAGE_DIR"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
cat > "$CONFIG_PATH" <<EOF
experiment:
  name: ${RESULT_NAME}
  seed: 42
  output_dir: $SCRATCHDIR/results/${RESULT_NAME}
  tree_assignment_order: confidence
dataset:
  name: deep1b
  path: $SCRATCHDIR/data/deep100m
  base_path: $SCRATCHDIR/data/deep100m/base.fbin
  query_path: $SCRATCHDIR/data/deep100m/query.fbin
  ground_truth_path: $SCRATCHDIR/data/deep100m/groundtruth.ivecs
  source_name: Deep1B
  source_url: yandex
  metric: euclidean
  split: train
  subset_size: 100000000
  storage_mode: memmap
model:
  branching_factor: 256
  tree_height: 2
  embedding_dim: 96
  encoder_hidden: 1024
  embed_dim: 256
  num_decoder_layers: 1
  num_heads: 8
  ff_dim: 1024
  dropout: 0.1
  alpha: 1.0
  num_trees: 1
training:
  batch_size: 16384
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
  num_leaves: [1]
  rerank_backend: numpy_cpu
EOF

cd BATL
INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"
RUN_STATUS=0
nvidia-smi

echo "Building one standard Deep100M BATL tree..."
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

if [ "$RUN_STATUS" -eq 0 ]; then
    echo "Evaluating top-1 bucket recall..."
    "$PYTHON_EXEC" -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
        --num-leaves 1 \
        --n-queries 10000 \
        --batch-search 50
    RUN_STATUS=$?
fi

cd "$SCRATCHDIR"
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="${STORAGE_DIR}_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying results back to $OUTPUT_DIR"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$OUTPUT_DIR/"
else
    echo "No results/${RESULT_NAME} directory was produced."
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
