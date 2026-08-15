#!/bin/bash

set -u

: "${TREE_INDEX:?TREE_INDEX must be set by the wrapper}"

RESULT_NAME="bigann100m_full_t4_tree_${TREE_INDEX}"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_bigann100m_full_t4_parallel_trees"
SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/bigann"

copy_first_existing() {
    local target="$1"
    shift
    for source in "$@"; do
        if [ -f "$source" ]; then
            echo "Copying $source -> $target"
            cp "$source" "$target"
            printf '%s\n' "$source" > "${target}.source"
            return 0
        fi
    done
    echo "Missing any source for $target:" >&2
    for source in "$@"; do
        echo "  $source" >&2
    done
    return 2
}

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/bigann100m "$STORAGE_TREE_DIR"

echo "Copying BIGANN100M data..."
copy_first_existing \
    data/bigann100m/base.u8bin \
    "$SRC_DATA/base.100M.u8bin" \
    "$SRC_DATA/base.100m.u8bin" \
    "$SRC_DATA/bigann_base.100M.u8bin" \
    "$SRC_DATA/bigann_base.100m.u8bin" || {
    echo "BIGANN100M recall requires the first 100M database vectors from ANN_SIFT1B." >&2
    echo "Do not substitute learn.100M.u8bin: the README states the learning set was not used for this ground truth." >&2
    exit 2
}
copy_first_existing \
    data/bigann100m/query.u8bin \
    "$SRC_DATA/query.public.10K.u8bin" \
    "$SRC_DATA/query.public.10k.u8bin" \
    "$SRC_DATA/query.10K.u8bin" \
    "$SRC_DATA/query.10k.u8bin" \
    "$SRC_DATA/bigann_query.10K.u8bin" \
    "$SRC_DATA/bigann_query.10k.u8bin"
copy_first_existing \
    data/bigann100m/groundtruth.ivecs \
    "$SRC_DATA/idx_100M.ivecs" \
    "$SRC_DATA/idx_100m.ivecs" \
    "$SRC_DATA/gnd/idx_100M.ivecs" \
    "$SRC_DATA/gnd/idx_100m.ivecs"

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/bin/python"

export LD_LIBRARY_PATH="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"

echo "Checking Python stack..."
"$PYTHON_EXEC" -u - <<'PY'
import faiss
import torch

print(f"faiss imported from: {faiss.__file__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing tree build.")
PY

BASE_CONFIG="$SCRATCHDIR/BATL/experiments/configs/bigann100m_full_t4.yaml"
CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
NAME="$RESULT_NAME" OUT_DIR="$SCRATCHDIR/results/${RESULT_NAME}" \
DATA_DIR="$SCRATCHDIR/data/bigann100m" \
SRC="$BASE_CONFIG" OUT="$CONFIG_PATH" \
"$PYTHON_EXEC" -u - <<'PY'
import os, yaml
from pathlib import Path

cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())
data_dir = os.environ["DATA_DIR"]
cfg["experiment"]["name"] = os.environ["NAME"]
cfg["experiment"]["output_dir"] = os.environ["OUT_DIR"]
cfg["dataset"]["path"] = data_dir
cfg["dataset"]["base_path"] = f"{data_dir}/base.u8bin"
cfg["dataset"]["query_path"] = f"{data_dir}/query.u8bin"
cfg["dataset"]["ground_truth_path"] = f"{data_dir}/groundtruth.ivecs"
Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cd BATL
INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"
TREE_INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence_tree_${TREE_INDEX}.pkl"
RUN_STATUS=0
nvidia-smi
"$PYTHON_EXEC" -u build.py \
    "$CONFIG_PATH" \
    --log \
    --index-path "$INDEX_PATH" \
    --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
    --batch-tree-update 32768 \
    --tree-index "$TREE_INDEX"
RUN_STATUS=$?

cd "$SCRATCHDIR"
echo "Copying tree ${TREE_INDEX} artifacts to $STORAGE_TREE_DIR"
if [ -f "$TREE_INDEX_PATH" ]; then
    cp "$TREE_INDEX_PATH" "$STORAGE_TREE_DIR/"
    {
        echo "base=$(cat data/bigann100m/base.u8bin.source)"
        echo "query=$(cat data/bigann100m/query.u8bin.source)"
        echo "groundtruth=$(cat data/bigann100m/groundtruth.ivecs.source)"
    } > "$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.sources"
else
    echo "Missing expected tree index: $TREE_INDEX_PATH"
fi
cp "$CONFIG_PATH" "$STORAGE_TREE_DIR/${RESULT_NAME}.yaml"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$STORAGE_TREE_DIR/"
fi
exit $RUN_STATUS
