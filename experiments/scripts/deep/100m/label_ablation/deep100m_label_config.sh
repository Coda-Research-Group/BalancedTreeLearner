#!/bin/bash
# Single source of truth for the Deep100M label-ablation config.
#
# Both the build and the search wrapper source this file, so the two phases
# cannot drift apart. Paths are resolved per job because $SCRATCHDIR differs
# between them, which is why the config is regenerated rather than copied.
#
# Requires: ARM_NAME, NEIGHBOR_SEARCH_SUBSET, RESULT_NAME, SCRATCHDIR, CONFIG_PATH

write_label_config() {
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
  # One tree by default. The >=2-of-4 frequency filter is inactive at T=1
  # (see _select_candidates in batl/search.py), so ablation recall sits
  # well below the T=4 headline
  # numbers by construction — compare arms to each other, never to an ensemble
  # run. The T=4 headline build overrides this.
  num_trees: ${NUM_TREES:-1}
training:
  batch_size: 16384
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  alternating_interval: 2
  convergence_patience: 2
  convergence_min_delta: 0.005
  top_k_neighbors: 100
  # THE ONLY VARIABLE IN THIS ABLATION.
  neighbor_search_subset: ${NEIGHBOR_SEARCH_SUBSET}
  neighbor_search_mode: random_subset
  # Inert for the subset arms — they index the sample in one allocation and
  # never chunk. Defaulted so their configs stay byte-identical to the ones
  # that already ran; the exact arm overrides it.
  neighbor_search_chunk_size: ${NEIGHBOR_SEARCH_CHUNK_SIZE:-1000000}
  neighbor_search_backend: faiss_gpu
  tree_update_cache_embeddings: false
  # The build wrappers pass --batch-tree-update on the CLI, which overrides
  # this; the assignment control has no such flag and sets it here so the
  # recorded config matches. On CUDA both resolve to the attention guard
  # (65535 // num_heads = 8191), so this is cosmetic there and only differs on
  # CPU. Defaulted to auto so the label arms are unchanged.
  tree_update_batch_size: ${TREE_UPDATE_BATCH_SIZE:-auto}
  device: cuda
evaluation:
  recall_at: [10]
  num_queries: 10000
  # M may not exceed beam_size (code-review A3 raises rather than clamping,
  # after silent clamping fabricated phantom M=150/200 rows in the Deep10M
  # runs). A fixed beam therefore caps the reachable recall: at beam 100 the
  # sweep stops at M=100, which on the T=4 index is Recall@10 0.897 — short of
  # the paper's 0.9539 operating point. To go further, raise both together, as
  # the alpha ablations do with one beam point per config.
  beam_size: ${BEAM_SIZE:-100}
  num_leaves: [${NUM_LEAVES:-10, 40, 80, 100}]
  # Pinned so the rerank path cannot differ with whichever GPU an arm lands
  # on. Only recall and mean_distcomp are compared here. The rerank-backend
  # control overrides both; defaulted so the label arms' configs are unchanged.
  rerank_backend: ${RERANK_BACKEND:-numpy_cpu}
  performance_profile: ${PERFORMANCE_PROFILE:-false}
  search_repetitions: 1
EOF
}
