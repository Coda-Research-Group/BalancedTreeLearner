#!/bin/bash
# Write one strict Deep100M config for the selected-ablation matrix.
#
# Every caller supplies every ablated value. Keeping the baseline values here
# explicit prevents a search wrapper from silently inheriting dataclass
# defaults that differ from the index it is evaluating.
#
# Arms stop on CONVERGENCE_PATIENCE. MAX_ALTERNATING_CYCLES is optional and
# exists for the epochs arm, which ablates the cycle count itself and pins it
# with CONVERGENCE_PATIENCE=0; leave it unset everywhere else.

write_selected_config() {
    : "${CONFIG_PATH:?CONFIG_PATH is required}"
    : "${RESULT_NAME:?RESULT_NAME is required}"
    : "${SCRATCHDIR:?SCRATCHDIR is required}"
    : "${NUM_TREES:?NUM_TREES is required}"
    : "${BRANCHING_FACTOR:?BRANCHING_FACTOR is required}"
    : "${CONVERGENCE_PATIENCE:?CONVERGENCE_PATIENCE is required}"
    : "${TOP_K_NEIGHBORS:?TOP_K_NEIGHBORS is required}"
    : "${NEIGHBOR_SEARCH_SUBSET:?NEIGHBOR_SEARCH_SUBSET is required}"
    : "${MIN_TREES:?MIN_TREES is required}"
    : "${BEAM_SIZE:?BEAM_SIZE is required}"
    : "${NUM_LEAVES:?NUM_LEAVES is required}"

    if [ "${CONVERGENCE_PATIENCE}" = 0 ] && [ -z "${MAX_ALTERNATING_CYCLES:-}" ]; then
        echo "write_selected_config: CONVERGENCE_PATIENCE=0 needs MAX_ALTERNATING_CYCLES" >&2
        return 1
    fi

    {
        cat <<EOF
experiment:
  name: ${RESULT_NAME}
  seed: 42
  output_dir: ${SCRATCHDIR}/results/${RESULT_NAME}
  tree_assignment_mode: round
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
  embed_dim: 256
  num_decoder_layers: 1
  num_heads: 8
  ff_dim: 1024
  dropout: 0.1
  alpha: 1.0
  num_trees: ${NUM_TREES}
training:
  batch_size: 16384
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  alternating_interval: 2
EOF
        if [ -n "${MAX_ALTERNATING_CYCLES:-}" ]; then
            echo "  max_alternating_cycles: ${MAX_ALTERNATING_CYCLES}"
        fi
        cat <<EOF
  convergence_patience: ${CONVERGENCE_PATIENCE}
  convergence_min_delta: 0.005
  top_k_neighbors: ${TOP_K_NEIGHBORS}
  neighbor_search_subset: ${NEIGHBOR_SEARCH_SUBSET}
  neighbor_search_mode: random_subset
  neighbor_search_chunk_size: 1000000
  neighbor_search_backend: faiss_gpu
  label_refresh: per_cycle
  tree_update_cache_embeddings: false
  tree_update_batch_size: 8191
  device: cuda
evaluation:
  recall_at: [10]
  num_queries: 10000
  beam_size: ${BEAM_SIZE}
  num_leaves: [${NUM_LEAVES}]
  min_trees: ${MIN_TREES}
  rerank_backend: numpy_cpu
  performance_profile: false
  search_repetitions: 1
EOF
    } > "$CONFIG_PATH"
}
