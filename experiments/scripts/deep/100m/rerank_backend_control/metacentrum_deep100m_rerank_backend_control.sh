#!/bin/bash
#PBS -N batl_d100m_rerank_control
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=44gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_rerank_backend_control.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Does the GPU actually pay on the Deep100M search, and which stage dominates?
#
# Every Deep100M QPS figure so far was measured with `rerank_backend:
# numpy_cpu`, pinned deliberately so the label-ablation arms stayed comparable.
# The GPU-resident path (C1) exists and has never been timed against a CPU run
# on the same node with the same index.
#
# Four runs, one job, one node, in a fixed order:
#
#   numpy_cpu          unprofiled   <- headline QPS baseline
#   numpy_cpu          profiled     <- stage attribution baseline
#   torch_gpu_resident unprofiled   <- headline QPS, C1
#   torch_gpu_resident profiled     <- stage attribution, C1
#
# Profiled runs synchronize CUDA at stage boundaries and are intentionally
# pessimistic, so attribution and headline throughput must come from different
# runs (throughput_gap_attribution.md §7.3).
#
# ORDER MATTERS. The resident upload reads the whole database and warms the
# host page cache, which would flatter a numpy_cpu run that followed it. The
# CPU runs therefore go first, and the database is read once up front so both
# backends start from the same cache state.
#
# gpu_mem=44gb: a resident Deep100M float32 database plus norms is 36.14 GiB,
# and check_device_capacity adds 2 GiB headroom -> 38.14 GiB. PBS gb is 10^9,
# so a 40gb request is only 37.25 GiB and would fail the capacity check; 44gb
# is 40.97 GiB. This is the one job in the Deep100M set that genuinely needs a
# large card, because residency is the thing being measured. If it lands
# somewhere too small the run fails fast with RerankGpuMemoryError naming the
# shortfall, rather than crashing mid-sweep.

set -u

RUN_STATUS=0
ARM_NAME="exact"
NEIGHBOR_SEARCH_SUBSET=100000000
NEIGHBOR_SEARCH_CHUNK_SIZE=10000000
SOURCE_NAME="deep100m_label_${ARM_NAME}"
CACHED_INDEX="/storage/brno2/home/jozefsprlak/results/batl_${SOURCE_NAME}/index_confidence.pkl"
TIMESTAMP=$(date +%Y%m%d_%H%M)
STORAGE_DIR="/storage/brno2/home/jozefsprlak/results/batl_deep100m_rerank_control_${TIMESTAMP}"

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Job ${PBS_JOBID} on $(hostname -f) at $(date)"
if [ ! -f "$CACHED_INDEX" ]; then
    echo "Missing index from the build job: $CACHED_INDEX" >&2
    exit 2
fi

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/deep100m input results "$STORAGE_DIR"
cp "$CACHED_INDEX" input/index_confidence.pkl

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
    raise SystemExit("CUDA is not visible to PyTorch; refusing rerank control.")
free, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"free VRAM: {free / 1024**3:.2f} GiB of {total / 1024**3:.2f} GiB")
print("resident rerank needs 36.14 GiB + 2.00 GiB headroom = 38.14 GiB")
PY

# Read the database once so neither backend benefits from the other's I/O.
echo "Warming the page cache ($(date))..."
cat data/deep100m/base.fbin > /dev/null
echo "Warm at $(date)."

nvidia-smi

run_point() {
    local backend="$1"
    local profile="$2"
    local label="${backend}_$([ "$profile" = "true" ] && echo profiled || echo plain)"
    local result_dir="$SCRATCHDIR/results/$label"
    local config_path="$SCRATCHDIR/${label}.yaml"

    mkdir -p "$result_dir"
    RESULT_NAME="$label" \
    CONFIG_PATH="$config_path" \
    RERANK_BACKEND="$backend" \
    PERFORMANCE_PROFILE="$profile" \
        write_label_config

    echo "=== $label ($(date)) ==="
    "$PYTHON_EXEC" -u search.py \
        "$config_path" \
        --log \
        --index-path "$SCRATCHDIR/input/index_confidence.pkl" \
        --result-dir "$result_dir"
    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "run $label failed with status $status" >&2
        RUN_STATUS=$status
    fi
    cp -r "$result_dir" "$STORAGE_DIR/" 2>/dev/null || true
    cp "$config_path" "$STORAGE_DIR/" 2>/dev/null || true
}

source "$SCRATCHDIR/BATL/experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
cd BATL

# CPU first: see the ORDER MATTERS note above.
run_point numpy_cpu false
run_point numpy_cpu true
run_point torch_gpu_resident false
run_point torch_gpu_resident true

echo "Done at $(date). Results: $STORAGE_DIR"
exit $RUN_STATUS
