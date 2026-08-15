#!/bin/bash
# Submit the complete five-family Deep100M matrix with build/search dependencies.
# Run this from the selected_ablation directory on MetaCentrum.

set -eu

BASELINE_BUILD=$(qsub metacentrum_deep100m_selected_baseline_build.sh)
BUCKETS_BUILD=$(qsub metacentrum_deep100m_selected_buckets_build.sh)
EPOCHS_BUILD=$(qsub metacentrum_deep100m_selected_epochs_build.sh)
TOPK_BUILD=$(qsub metacentrum_deep100m_selected_topk_build.sh)
SAMPLE_BUILD=$(qsub metacentrum_deep100m_selected_sample_build.sh)

BASELINE_SEARCH=$(qsub -W depend=afterok:$BASELINE_BUILD metacentrum_deep100m_selected_baseline_search.sh)
REPETITIONS_SEARCH=$(qsub -W depend=afterok:$BASELINE_BUILD metacentrum_deep100m_selected_repetitions_search.sh)
BUCKETS_SEARCH=$(qsub -W depend=afterok:$BASELINE_BUILD:${BUCKETS_BUILD} metacentrum_deep100m_selected_buckets_search.sh)
EPOCHS_SEARCH=$(qsub -W depend=afterok:$BASELINE_BUILD:${EPOCHS_BUILD} metacentrum_deep100m_selected_epochs_search.sh)
TOPK_SEARCH=$(qsub -W depend=afterok:$BASELINE_BUILD:${TOPK_BUILD} metacentrum_deep100m_selected_topk_search.sh)
SAMPLE_SEARCH=$(qsub -W depend=afterok:$BASELINE_BUILD:${SAMPLE_BUILD} metacentrum_deep100m_selected_sample_search.sh)

printf '%-22s %s\n' \
    baseline_build "$BASELINE_BUILD" \
    buckets_build "$BUCKETS_BUILD" \
    epochs_build "$EPOCHS_BUILD" \
    topk_build "$TOPK_BUILD" \
    sample_build "$SAMPLE_BUILD" \
    baseline_search "$BASELINE_SEARCH" \
    repetitions_search "$REPETITIONS_SEARCH" \
    buckets_search "$BUCKETS_SEARCH" \
    epochs_search "$EPOCHS_SEARCH" \
    topk_search "$TOPK_SEARCH" \
    sample_search "$SAMPLE_SEARCH"
