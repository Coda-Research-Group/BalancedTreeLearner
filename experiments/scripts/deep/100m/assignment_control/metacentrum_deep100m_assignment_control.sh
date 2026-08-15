#!/bin/bash
#PBS -N batl_d100m_assign_control
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_assignment_control.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Does S8 hold at 100M?
#
# The iteration-order result — confidence ordering worth 1.43-1.89x on the
# round kernel — is the only lever the whole tree-quality investigation has
# found, and it rests on SIFT1M alone. `tree_quality_suspects.md` says to
# confirm once at Deep100M before headline results quote it. This is that
# confirmation.
#
# Two arms only: round_confidence and round_input. Sequential mode is not
# available here — it needs the full-K branch order materialized, which is
# 100M x 256 at the root — but the ordering question does not need it. What
# S8 claims is testable with the round kernel alone.
#
# No retraining and no rebuild: this reassigns a saved index, so the two arms
# share one set of model weights and differ only in the priority used to
# resolve capacity contention. There is no training noise to see through, which
# is why the SIFT1M version needed no seed repeat.
#
# round_confidence must reproduce the cached source tree (within the float-drift
# tolerance) before the comparison means anything; the harness raises if not.
#
# Submit with the source index, e.g. the paper-faithful 1% label arm:
#   qsub -v SOURCE_INDEX=/storage/brno2/home/jozefsprlak/results/\
# batl_deep100m_label_subset_1pct/index_confidence.pkl <this script>
#
# Those arms are num_trees=1, so recall sits below the T=4 headline numbers by
# construction. Compare the two arms to each other, never to an ensemble run.

set -u

RUN_STATUS=0
: "${SOURCE_INDEX:?Submit with -v SOURCE_INDEX=/storage/.../index_confidence.pkl}"
if [ ! -f "$SOURCE_INDEX" ]; then
    echo "Source index not found: $SOURCE_INDEX" >&2
    exit 2
fi

# Match the label-ablation build so the regenerated config describes the run
# that produced the source index.
NEIGHBOR_SEARCH_SUBSET=1000000
# Mirrors the --batch-tree-update the build wrappers pass, so the recorded
# config matches the run that produced the source index. On CUDA this is
# cosmetic: the attention guard caps any explicit value to 65535 // num_heads
# = 8191, which is also what `auto` resolves to, and job 22750115 logged
# exactly that capping. It only bites on CPU, where `auto` gives 4096 and an
# explicit value is honoured.
TREE_UPDATE_BATCH_SIZE=32768
RESULT_NAME="deep100m_assignment_control"
TIMESTAMP=$(date +%Y%m%d_%H%M)
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${RESULT_NAME}_${TIMESTAMP}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
echo "Source index: $SOURCE_INDEX"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m input results configs "$STORAGE_DIR"
cp "$SOURCE_INDEX" input/index_confidence.pkl

SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/deep1b"
echo "Copying Deep100M data..."
cp "$SRC_DATA/deep100M_base.fbin" data/deep100m/base.fbin
cp "$SRC_DATA/deep1B_queries.fbin" data/deep100m/query.fbin
cp "$SRC_DATA/deep100M_groundtruth.ivecs" data/deep100m/groundtruth.ivecs

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

# See the 2026-05-24 REVIEW in experiments/scripts/discussion.md.
export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python GPU stack..."
"$PYTHON_EXEC" -u - <<'PY'
import torch

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing assignment control.")
print(f"device: {torch.cuda.get_device_name(0)}")
PY

BASE_CONFIG="$SCRATCHDIR/configs/${RESULT_NAME}.yaml"
CONTROL_DIR="$SCRATCHDIR/results/${RESULT_NAME}"
source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
CONFIG_PATH="$BASE_CONFIG" write_label_config

cd BATL
nvidia-smi
echo "=== reassigning ($(date)) ==="
"$PYTHON_EXEC" -u experiments/scripts/sift128/run_assignment_control.py \
    "$BASE_CONFIG" \
    --log \
    --source-index "$SCRATCHDIR/input/index_confidence.pkl" \
    --output-dir "$CONTROL_DIR" \
    --arms round_confidence,round_input
RUN_STATUS=$?
cp -r "$CONTROL_DIR/." "$STORAGE_DIR/" 2>/dev/null || true

if [ "$RUN_STATUS" -ne 0 ]; then
    echo "Reassignment failed with status $RUN_STATUS; not searching." >&2
    exit $RUN_STATUS
fi

ARMS=(
    "round_confidence:round:confidence"
    "round_input:round:input"
)

for ARM_SPEC in "${ARMS[@]}"; do
    IFS=: read -r ARM MODE ORDER <<< "$ARM_SPEC"
    ARM_CONFIG="$SCRATCHDIR/configs/${ARM}.yaml"
    ARM_RESULT_DIR="$CONTROL_DIR/$ARM/search"
    ARM_INDEX="$CONTROL_DIR/$ARM/index_${ORDER}.pkl"
    mkdir -p "$ARM_RESULT_DIR"

    SRC="$BASE_CONFIG" OUT="$ARM_CONFIG" RES="$ARM_RESULT_DIR" \
    ARM="$ARM" MODE="$MODE" ORDER="$ORDER" \
    "$PYTHON_EXEC" -u - <<'PY'
import os
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())
cfg["experiment"]["name"] = f"deep100m_assignment_{os.environ['ARM']}"
cfg["experiment"]["output_dir"] = os.environ["RES"]
cfg["evaluation"]["tree_assignment_mode"] = os.environ["MODE"]
cfg["evaluation"]["tree_assignment_order"] = os.environ["ORDER"]
Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

    echo "=== searching $ARM ($(date)) ==="
    "$PYTHON_EXEC" -u search.py "$ARM_CONFIG" \
        --log \
        --index-path "$ARM_INDEX" \
        --result-dir "$ARM_RESULT_DIR"
    STATUS=$?
    if [ "$STATUS" -ne 0 ]; then
        echo "search $ARM failed with status $STATUS" >&2
        RUN_STATUS=$STATUS
    fi
    cp -r "$CONTROL_DIR/." "$STORAGE_DIR/" 2>/dev/null || true
done

cp "$SCRATCHDIR"/configs/*.yaml "$STORAGE_DIR/" 2>/dev/null || true
echo "Done at $(date). Results: $STORAGE_DIR"
exit $RUN_STATUS
