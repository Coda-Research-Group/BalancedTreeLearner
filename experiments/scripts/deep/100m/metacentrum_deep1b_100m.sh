#!/bin/bash
#PBS -N batl_deep100m_gpu_timing
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=48gb:mem=250gb:scratch_local=200gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_gpu_timing.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

# 1. Setup scratch and cleanup trap
trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd $SCRATCHDIR

# 2. Copy BATL repository
echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m

# 3. Map actual files to expected filenames
SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying data..."
cp $SRC_DATA/deep100M_base.fbin data/deep100m/base.fbin
cp $SRC_DATA/deep1B_queries.fbin data/deep100m/query.fbin
cp $SRC_DATA/deep100M_groundtruth.ivecs data/deep100m/groundtruth.ivecs

# 4. Activate GPU environment
module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$SCRATCHDIR/BATL"

# 5. Check GPU and FAISS-GPU visibility before spending the allocation
echo "Checking Python GPU stack..."
$PYTHON_EXEC -u - <<'PY'
import faiss
import torch

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is not visible to PyTorch; refusing GPU timing run.")

print(f"faiss_gpu_available: {hasattr(faiss, 'StandardGpuResources')}")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing GPU timing run.")
PY

# 6. Write temporary Deep100M GPU timing config
CONFIG_PATH="$SCRATCHDIR/deep100m_gpu_timing.yaml"
cat > "$CONFIG_PATH" <<EOF
experiment:
  name: deep100m_gpu_timing
  seed: 42
  output_dir: $SCRATCHDIR/results/deep100m_gpu_timing
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
  neighbor_search_subset: 100000
  neighbor_search_mode: random_subset
  neighbor_search_chunk_size: 1000000
  neighbor_search_backend: faiss_gpu
  tree_update_cache_embeddings: false
  device: cuda

evaluation:
  recall_at: [10]
  num_queries: 1000
  beam_size: 100
  num_leaves: [10, 40, 80, 100]
  rerank_backend: numpy_cpu
EOF

# 7. Execution
cd BATL
INDEX_PATH="$SCRATCHDIR/results/deep100m_gpu_timing/index_confidence.pkl"
echo "Running BATL Deep100M GPU timing run..."
nvidia-smi
RUN_STATUS=0
$PYTHON_EXEC -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/deep100m_gpu_timing" \
    --batch-tree-update 32768
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ]; then
    $PYTHON_EXEC -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/deep100m_gpu_timing" \
        --num-leaves 10 40 80 100 \
        --n-queries 10000 \
        --batch-search 2000
    RUN_STATUS=$?
fi

# 8. Save results to storage
cd $SCRATCHDIR
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_gpu_timing_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying results back to $OUTPUT_DIR"
if [ -d results/deep100m_gpu_timing ]; then
    cp -r results/deep100m_gpu_timing "$OUTPUT_DIR/"
else
    echo "No results/deep100m_gpu_timing directory was produced."
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
