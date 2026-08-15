#!/bin/bash
#PBS -N batl_d100m_t4_merge
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=16gb:mem=150gb:scratch_ssd=200gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_t4_merge_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Merge the four Deep100M trees and sweep. Run after all four tree jobs finish.
#
# It refuses a partial ensemble rather than merging three trees: `num_trees`
# drives the >=2-of-N frequency filter, so a three-tree index silently changes
# the retrieval rule and produces a curve that looks valid and is comparable to
# nothing.
#
# Rerank is numpy_cpu, matching every other Deep100M sweep so this index's
# curve is directly comparable to the ablation arms and the July baseline. For
# the GPU-resident throughput number, point
# rerank_backend_control/metacentrum_deep100m_rerank_backend_control.sh at the
# merged index afterwards — that job needs a 44gb card, this one does not.
#
# scratch_ssd, not scratch_local: numpy_cpu rerank reads scattered rows out of
# a 38 GB memmap, so the scratch medium is on the critical path. Every ablation
# merge-search job in this repo already asks for scratch_ssd; the label-ablation
# and T=4 wrappers asked for scratch_local by inheritance.
#
# SIZING, measured from job 22821720 (9m45s walltime, single beam point):
#   cput 43m12s over 16 cpus = 4.4 cores actually busy   -> ncpus 8
#   mem 90 GiB of the ~279 GiB requested                 -> mem 150gb
#   walltime 9m45s of 6h                                 -> walltime 4h
#
# The GPU sat at 1% because rerank is numpy_cpu; the card is here only for beam
# decode. That is not free — it holds a GPU allocation for the whole run — but
# the alternatives are both worse for a shared cluster: switching this job to
# torch_gpu_resident would use the card properly and cut search ~3x, at the
# cost of requesting a much scarcer 44gb card, and dropping ngpus entirely
# would need `device: cpu`, whose decode cost at K=256 has never been measured.
# Left as-is deliberately; see the discussion note.

set -u

RUN_STATUS=0
RESULT_NAME="deep100m_t4"
NUM_TREES=4
NEIGHBOR_SEARCH_SUBSET=1000000
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_trees"
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
for TREE_INDEX in 0 1 2 3; do
    TREE_PATH="$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.pkl"
    if [ ! -f "$TREE_PATH" ]; then
        echo "Missing tree index: $TREE_PATH" >&2
        echo "All four per-tree jobs must finish before the merge." >&2
        exit 2
    fi
done

cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m "results/$RESULT_NAME" "$STORAGE_DIR"

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
write_label_config

cd BATL
MERGED_INDEX="$RESULT_DIR/index_confidence.pkl"
nvidia-smi

"$PYTHON_EXEC" -u merge_index.py \
    --output "$MERGED_INDEX" \
    "$STORAGE_TREE_DIR/index_confidence_tree_0.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_1.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_2.pkl" \
    "$STORAGE_TREE_DIR/index_confidence_tree_3.pkl"
RUN_STATUS=$?

# One beam point per config, with num_leaves == beam_size. A single search at
# beam 100 stops at M=100 and Recall@10 0.897, short of the paper's 0.9539
# operating point, because A3 forbids M > beam. The alpha ablations solved this
# the same way; job 20370198 reached 0.9097 at beam 60 that way.
SEARCH_POINTS=(10 20 40 60 80 100 150 200 250 300)

if [ "$RUN_STATUS" -eq 0 ]; then
    for POINT in "${SEARCH_POINTS[@]}"; do
        POINT_CONFIG="$SCRATCHDIR/${RESULT_NAME}_beam_${POINT}.yaml"
        POINT_DIR="$RESULT_DIR/beam_${POINT}"
        mkdir -p "$POINT_DIR"

        RESULT_NAME="${RESULT_NAME}_beam_${POINT}" \
        CONFIG_PATH="$POINT_CONFIG" \
        BEAM_SIZE="$POINT" \
        NUM_LEAVES="$POINT" \
            write_label_config

        echo "=== beam_size=${POINT}, num_leaves=${POINT} ($(date)) ==="
        "$PYTHON_EXEC" -u search.py \
            "$POINT_CONFIG" \
            --log \
            --index-path "$MERGED_INDEX" \
            --result-dir "$POINT_DIR"
        STATUS=$?
        if [ "$STATUS" -ne 0 ]; then
            echo "beam ${POINT} failed with status $STATUS" >&2
            RUN_STATUS=$STATUS
        fi
        # Copy after each point so a later failure cannot cost the earlier ones.
        cp -r "$RESULT_DIR/." "$STORAGE_DIR/" 2>/dev/null || true
    done
fi

cd "$SCRATCHDIR"
if [ "$RUN_STATUS" -eq 0 ]; then
    echo "Copying results to $STORAGE_DIR"
    cp -r "results/${RESULT_NAME}/." "$STORAGE_DIR/" 2>/dev/null || true
    cp "$CONFIG_PATH" "$STORAGE_DIR/"
else
    echo "Merge or search failed; nothing copied out." >&2
    RUN_STATUS=${RUN_STATUS:-1}
fi
echo "Done at $(date)."
exit $RUN_STATUS
