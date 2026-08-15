#!/bin/bash
#PBS -N batl_deep100m_one_cycle_smoke
# NOTE: Diagnostic Deep100M one-cycle run. This is intended to validate the
# single-tree memory path before launching full timing jobs.
# Install GPU env with: bash experiments/scripts/deep/100m/setup_batl_gpu_env.sh
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=48gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_one_cycle_smoke.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd $SCRATCHDIR

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep100M data..."
cp $SRC_DATA/deep100M_base.fbin data/deep100m/base.fbin
cp $SRC_DATA/deep1B_queries.fbin data/deep100m/query.fbin
cp $SRC_DATA/deep100M_groundtruth.ivecs data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu/bin/python"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python GPU stack..."
$PYTHON_EXEC -u - <<'PY'
import faiss
import torch

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is not visible to PyTorch; refusing GPU smoke run.")

print(f"faiss_gpu_available: {hasattr(faiss, 'StandardGpuResources')}")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing GPU smoke run.")
PY

CONFIG_PATH="$SCRATCHDIR/deep100m_one_cycle_smoke.yaml"
cat > "$CONFIG_PATH" <<EOF
experiment:
  name: deep100m_one_cycle_smoke
  seed: 42
  output_dir: $SCRATCHDIR/results/deep100m_one_cycle_smoke
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
  max_alternating_cycles: 1
  convergence_patience: 0
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
  num_leaves: [10]
  rerank_backend: numpy_cpu
EOF

cd BATL
INDEX_PATH="$SCRATCHDIR/results/deep100m_one_cycle_smoke/index_confidence.pkl"
echo "Running BATL Deep100M one-cycle smoke..."
nvidia-smi
RUN_STATUS=0
$PYTHON_EXEC -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/deep100m_one_cycle_smoke" \
    --batch-tree-update 32768
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ]; then
    $PYTHON_EXEC -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/deep100m_one_cycle_smoke" \
        --num-leaves 10 40 80 100 \
        --n-queries 10000 \
        --batch-search 2000
    RUN_STATUS=$?
fi

cd $SCRATCHDIR
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_one_cycle_smoke_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying results back to $OUTPUT_DIR"
if [ -d results/deep100m_one_cycle_smoke ]; then
    cp -r results/deep100m_one_cycle_smoke "$OUTPUT_DIR/"
else
    echo "No results/deep100m_one_cycle_smoke directory was produced."
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
