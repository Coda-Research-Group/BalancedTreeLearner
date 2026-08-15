#!/bin/bash
#PBS -N batl_deep100m_full_t4
# Full Deep100M 4-tree ensemble build + search sweep.
#
# Builds on the proven Deep100M one-cycle smoke (which validated the root
# tree-update OOM surgical fix at K=256, H=2, N=100M with mem<<200 GB peak)
# and the search-side CUDA attention guard (clamps --batch-search down to
# (65535 // num_heads) // beam_size when needed).
#
# Per-tree cost from the smoke: ~43 min per cycle (label mining + 2 epochs
# + L1 + L2 tree update). Training stops on convergence_patience=2, and
# realistic cycles per tree is 3-6, so total build cost ~12-20 h for 4
# trees. The 48h walltime is the MetaCentrum cap;
# build_batl_index saves a partial index after each completed tree, so
# trees 1..K-1 are retained if the job times out mid-build.
#
# Install GPU env with: bash experiments/scripts/deep/100m/setup_batl_gpu_env.sh
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=48gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_full_t4.log
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
echo "Mapping and copying Deep100M data..."
cp $SRC_DATA/deep100M_base.fbin data/deep100m/base.fbin
cp $SRC_DATA/deep1B_queries.fbin data/deep100m/query.fbin
cp $SRC_DATA/deep100M_groundtruth.ivecs data/deep100m/groundtruth.ivecs

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
    raise SystemExit("CUDA is not visible to PyTorch; refusing full Deep100M build.")

print(f"faiss_gpu_available: {hasattr(faiss, 'StandardGpuResources')}")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing full Deep100M build.")
PY

# 6. Write full Deep100M T=4 ensemble config
CONFIG_PATH="$SCRATCHDIR/deep100m_full_t4.yaml"
cat > "$CONFIG_PATH" <<EOF
experiment:
  name: deep100m_full_t4
  seed: 42
  output_dir: $SCRATCHDIR/results/deep100m_full_t4
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
  num_trees: 4

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
  num_leaves: [10, 40, 80, 100]
  rerank_backend: numpy_cpu
EOF

# 7. Execution
cd BATL
INDEX_PATH="$SCRATCHDIR/results/deep100m_full_t4/index_confidence.pkl"
echo "Running BATL Deep100M full T=4 ensemble build..."
nvidia-smi
RUN_STATUS=0
$PYTHON_EXEC -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/deep100m_full_t4" \
    --batch-tree-update 32768
RUN_STATUS=$?

# Mid-job snapshot: copy partial index out of scratch even if build did not
# finish all 4 trees, so the next allocation can resume or query early trees.
echo "Mid-job snapshot of partial index..."
RESULT_TS_MID=$(date +%Y%m%d_%H%M)
MID_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_full_t4_partial_${RESULT_TS_MID}"
mkdir -p "$MID_DIR"
if [ -d "$SCRATCHDIR/results/deep100m_full_t4" ]; then
    cp -r "$SCRATCHDIR/results/deep100m_full_t4" "$MID_DIR/"
fi
cp "$CONFIG_PATH" "$MID_DIR/"

if [ "$RUN_STATUS" -eq 0 ]; then
    echo "Build complete; running search sweep over num_leaves=[10, 40, 80, 100]..."
    $PYTHON_EXEC -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/deep100m_full_t4" \
        --num-leaves 10 40 80 100 \
        --n-queries 10000 \
        --batch-search 2000
    RUN_STATUS=$?
fi

# 8. Save results to storage
cd $SCRATCHDIR
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_full_t4_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying results back to $OUTPUT_DIR"
if [ -d results/deep100m_full_t4 ]; then
    cp -r results/deep100m_full_t4 "$OUTPUT_DIR/"
else
    echo "No results/deep100m_full_t4 directory was produced."
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
