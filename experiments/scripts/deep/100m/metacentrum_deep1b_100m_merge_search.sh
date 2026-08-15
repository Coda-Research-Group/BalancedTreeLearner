#!/bin/bash
#PBS -N batl_deep100m_merge_search
# Merge four independently built Deep100M tree indexes and run the search sweep.
# Submit after metacentrum_deep1b_100m_tree_{0,1,2,3}.sh have copied their
# index_confidence_tree_<k>.pkl files into STORAGE_TREE_DIR.
# gpu_mem must hold the whole base resident for torch_gpu_resident rerank:
# 100M x 96 float32 + norms = 36.1 GiB, plus the 2 GiB headroom the capacity
# check reserves. 40gb leaves only 1.5 GiB for the gather, model, and beam
# tensors; 48gb leaves 6.6 GiB. Dropping below this makes the reranker fall
# back to numpy_cpu and the sweep re-measures the old CPU path.
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=44gb:mem=200gb:scratch_local=200gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_merge_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

RESULT_NAME="deep100m_full_t4_parallel_merged"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_full_t4_parallel_trees"

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

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

# FAISS resolves the system libstdc++.so.6, which on some nodes predates
# CXXABI_1.3.15 and fails to load libfaiss.so. Same fix the BIGANN/SPACEV
# wrappers carry (2026-05-24 REVIEW in experiments/scripts/discussion.md);
# this script never got it.
export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

# Fail in seconds rather than after the 40 GB data copy on the next attempt.
echo "Checking Python stack..."
"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

print(f"faiss imported from: {faiss.__file__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing search run.")
PY

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
  # auto resolves to torch_gpu_resident on CUDA and falls back to numpy_cpu
  # with a WARN if the base does not fit. Pinning numpy_cpu here would skip
  # the resident path entirely, which is what this sweep exists to measure.
  rerank_backend: auto
  # Stage timings put a CUDA synchronize at every stage boundary, so the QPS
  # this run reports is slightly pessimistic and is NOT the Table-1 headline
  # number. Re-run with performance_profile: false once the resident reranker
  # is known to work, and quote that run instead.
  performance_profile: true
  search_repetitions: 3
EOF

cd BATL
MERGED_INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"
CACHED_INDEX="$MERGED_INDEX_PATH"

for TREE_INDEX in 0 1 2 3; do
    TREE_PATH="$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
    if [ ! -f "$TREE_PATH" ]; then
        echo "Missing required tree index: $TREE_PATH" >&2
        exit 2
    fi
done

RUN_STATUS=0
nvidia-smi
"$PYTHON_EXEC" -u merge_index.py \
    --output "$MERGED_INDEX_PATH" \
    "$STORAGE_TREE_DIR/index_confidence_tree_0.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_1.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_2.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_3.pkl"
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ]; then
    "$PYTHON_EXEC" -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$CACHED_INDEX" \
        --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
        --num-leaves 10 40 80 100 \
        --n-queries 10000 \
        --batch-search 2000
    RUN_STATUS=$?
fi

cd "$SCRATCHDIR"
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_full_t4_parallel_merged_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying merged search results back to $OUTPUT_DIR"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$OUTPUT_DIR/"
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
