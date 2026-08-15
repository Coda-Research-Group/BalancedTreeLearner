#!/bin/bash
#PBS -N batl_laion_10m_gpu_timing
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=16gb:mem=64gb:scratch_local=100gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/laion_10m_gpu_timing.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

# 1. Setup scratch and cleanup trap
trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd $SCRATCHDIR

# 2. Copy BATL repository
echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/laion

# 3. Map actual files to expected filenames
SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/laion5b"
echo "Mapping and copying data..."
# Loader auto-resolves LAION 'emb' / 'knns' keys and converts official one-based
# gold-standard ids to zero-based BATL row ids.
cp $SRC_DATA/laion2B-en-clip768v2-n=10M.h5 data/laion/base.h5
cp $SRC_DATA/public-queries-10k-clip768v2.h5 data/laion/query.h5
cp $SRC_DATA/laion2B-en-public-gold-standard-v2-10M.h5 data/laion/groundtruth.h5

# 4. Activate GPU environment
module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu/bin/python"

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

# 6. Write temporary LAION 10M GPU timing config
CONFIG_PATH="$SCRATCHDIR/laion_10m_gpu_timing.yaml"
cat > "$CONFIG_PATH" <<EOF
experiment:
  name: laion_10m_gpu_timing
  seed: 42
  output_dir: $SCRATCHDIR/results/laion_10m_gpu_timing
  tree_assignment_order: confidence

dataset:
  name: laion5b
  path: $SCRATCHDIR/data/laion
  base_path: $SCRATCHDIR/data/laion/base.h5
  query_path: $SCRATCHDIR/data/laion/query.h5
  ground_truth_path: $SCRATCHDIR/data/laion/groundtruth.h5
  source_name: LAION-5B
  source_url: laion
  metric: angular
  split: train
  subset_size: 10000000
  storage_mode: auto

model:
  branching_factor: 256
  tree_height: 2
  embedding_dim: 768
  encoder_hidden: 1024
  embed_dim: 256
  num_decoder_layers: 1
  num_heads: 8
  ff_dim: 1024
  dropout: 0.1
  alpha: 1.0
  num_trees: 4

training:
  batch_size: 4096
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  alternating_interval: 2
  max_alternating_cycles: 10
  convergence_patience: 3
  convergence_min_delta: 0.005
  top_k_neighbors: 100
  neighbor_search_subset: 100000
  neighbor_search_mode: sequential_chunked
  neighbor_search_chunk_size: 1000000
  neighbor_search_backend: faiss_gpu
  tree_update_cache_embeddings: false
  device: cuda

evaluation:
  recall_at: [10]
  num_queries: 1000
  beam_size: 100
  num_leaves: [10, 40, 80, 100]
  rerank_backend: torch_gpu
EOF

# 7. Execution
cd BATL
INDEX_PATH="$SCRATCHDIR/results/laion_10m_gpu_timing/index_confidence.pkl"
echo "Running BATL LAION 10M GPU timing run..."
nvidia-smi
RUN_STATUS=0
$PYTHON_EXEC -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/laion_10m_gpu_timing" \
    --batch-tree-update 8192
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ]; then
    $PYTHON_EXEC -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/laion_10m_gpu_timing" \
        --num-leaves 10 40 80 100 \
        --n-queries 1000 \
        --batch-search 25
    RUN_STATUS=$?
fi

# 8. Save results to storage
cd $SCRATCHDIR
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_laion_10m_gpu_timing_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying results back to $OUTPUT_DIR"
if [ -d results/laion_10m_gpu_timing ]; then
    cp -r results/laion_10m_gpu_timing "$OUTPUT_DIR/"
else
    echo "No results/laion_10m_gpu_timing directory was produced."
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
