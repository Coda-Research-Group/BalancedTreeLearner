#!/bin/bash
# Common body for the Deep10M label-quality A/B (SPEC_performance C4).
#
# The two arms differ in exactly one config value, NEIGHBOR_SEARCH_SUBSET:
# the database subset that training labels are mined from. Everything else —
# seed, model, schedule, search sweep — is identical, so any recall-per-bucket
# difference is attributable to label quality alone.
#
# Why this matters: mining top-100 neighbours inside a p-fraction subset
# approximates the (100/p)-th true neighbours. At the inherited 1% setting
# (100k of 10M) the labels are roughly rank-10,000 neighbours, while recall@10
# is what gets evaluated. The BATL paper uses the same technique but never
# states its subset size, and says only that "the quality of nearest neighbors
# improves with the increasing size of the subset" (S3.2.3).
#
# Required variables from wrapper:
#   ARM_NAME, NEIGHBOR_SEARCH_SUBSET

set -u

RUN_STATUS=0
RESULT_NAME="deep10m_label_${ARM_NAME}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep10m "results/$RESULT_NAME"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Mapping and copying Deep10M data..."
cp "$SRC_DATA/deep_10m_base.fbin" data/deep10m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep10m/query.fbin
cp "$SRC_DATA/deep10M_groundtruth.ivecs" data/deep10m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

# See the 2026-05-24 REVIEW in experiments/scripts/discussion.md: some
# nodes resolve a system libstdc++ too old for libfaiss.so.
export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python GPU stack..."
"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing label-ablation run.")
print(f"faiss_gpu_available: {hasattr(faiss, 'StandardGpuResources')}")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("FAISS-GPU bindings are unavailable; refusing label-ablation run.")
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
  path: $SCRATCHDIR/data/deep10m
  base_path: $SCRATCHDIR/data/deep10m/base.fbin
  query_path: $SCRATCHDIR/data/deep10m/query.fbin
  ground_truth_path: $SCRATCHDIR/data/deep10m/groundtruth.ivecs
  source_name: Deep1B
  source_url: yandex
  metric: euclidean
  split: train
  subset_size: 10000000
  storage_mode: auto

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
  # One tree: this A/B measures label quality, not ensemble behaviour, and a
  # single tree quarters the build cost. min_trees filtering is inactive at
  # T=1, so recall here is lower than a T=4 run by construction — compare the
  # two arms against each other, never against the T=4 headline numbers.
  num_trees: 1

training:
  batch_size: 4096
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  alternating_interval: 2
  convergence_patience: 3
  convergence_min_delta: 0.005
  top_k_neighbors: 100
  # THE ONLY VARIABLE IN THIS A/B.
  neighbor_search_subset: ${NEIGHBOR_SEARCH_SUBSET}
  neighbor_search_mode: random_subset
  neighbor_search_chunk_size: 1000000
  neighbor_search_backend: faiss_gpu
  tree_update_cache_embeddings: auto
  device: cuda

evaluation:
  recall_at: [10]
  num_queries: 1000
  beam_size: 100
  # A wide sweep: the question is recall per bucket, so the shape of the curve
  # matters more than any single point.
  num_leaves: [10, 40, 80, 100, 150, 200]
  rerank_backend: auto
  performance_profile: false
  search_repetitions: 1
EOF

cd BATL
INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"
echo "Building ${ARM_NAME} arm (neighbor_search_subset=${NEIGHBOR_SEARCH_SUBSET})..."
nvidia-smi

# --cycle-diagnostics also captures the chosen-rank histogram (C2a), which is
# the other open measurement and costs nothing extra here.
"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
    --batch-tree-update auto \
    --cycle-diagnostics \
    --cycle-diagnostics-queries 1000 \
    --cycle-diagnostics-loss-pairs 100000
RUN_STATUS=$?

if [ "$RUN_STATUS" -eq 0 ]; then
    "$PYTHON_EXEC" -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
        --num-leaves 10 40 80 100 150 200 \
        --n-queries 1000 \
        --batch-search 25
    RUN_STATUS=$?
fi

cd "$SCRATCHDIR"
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying results to $OUTPUT_DIR"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$OUTPUT_DIR/"
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
