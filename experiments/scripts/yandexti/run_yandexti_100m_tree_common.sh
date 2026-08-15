#!/bin/bash
# Build ONE tree of the YandexTI-100M ensemble.
#
# Uses `metric: inner_product` (see experiments/configs/yandexti100m_full_t4.yaml)
# -- YandexTI's native similarity is a raw dot product, not cosine.
#
# Also requires groundtruth.100M.ibin to already be the *converted* file (see
# convert_yandexti_100m_groundtruth.py in this directory) -- the raw GT_100M
# file from dl.fbaipublicfiles.com has interleaved distances and will fail
# BATL's ibin size check as-is.
#
# Required from wrapper: TREE_INDEX

set -u

: "${TREE_INDEX:?TREE_INDEX must be set by the wrapper}"

RESULT_NAME="yandexti100m_full_t4_tree_${TREE_INDEX}"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_yandexti100m_full_t4_parallel_trees"
SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data/YandexTI"

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

echo "Job ${PBS_JOBID:-local} on $(hostname -f) at $(date)"
echo "Building YandexTI100M tree ${TREE_INDEX}"
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/yandexti100m "$STORAGE_TREE_DIR"

echo "Copying YandexTI100M data..."
copy_first_existing \
    data/yandexti100m/base.fbin \
    "$SRC_DATA/base.100M.fbin" \
    "$SRC_DATA/base.100m.fbin" \
    "$SRC_DATA/text2image_base.100M.fbin" \
    "$SRC_DATA/text2image_base.100m.fbin" || {
    echo "YandexTI100M requires the range-cropped first 100M vectors of base.1B.fbin" >&2
    echo "with the header vector count patched to 100000000 (see README notes)." >&2
    exit 2
}
copy_first_existing \
    data/yandexti100m/query.fbin \
    "$SRC_DATA/query.public.100K.fbin" \
    "$SRC_DATA/query.public.100k.fbin"
copy_first_existing \
    data/yandexti100m/groundtruth.ibin \
    "$SRC_DATA/groundtruth.100M.ibin" \
    "$SRC_DATA/groundtruth.100m.ibin" || {
    echo "Missing converted groundtruth.100M.ibin -- run convert_yandexti_100m_groundtruth.py" >&2
    echo "on GT_100M/text2image-100M first; the raw FAIR file is not plain ibin." >&2
    exit 2
}

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

BASE_CONFIG="$SCRATCHDIR/BATL/experiments/configs/yandexti100m_full_t4.yaml"
CONFIG_PATH="$SCRATCHDIR/${RESULT_NAME}.yaml"
NAME="$RESULT_NAME" OUT_DIR="$SCRATCHDIR/results/${RESULT_NAME}" \
DATA_DIR="$SCRATCHDIR/data/yandexti100m" \
SRC="$BASE_CONFIG" OUT="$CONFIG_PATH" \
"$PYTHON_EXEC" -u - <<'PY'
import os, yaml
from pathlib import Path

cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())
data_dir = os.environ["DATA_DIR"]
cfg["experiment"]["name"] = os.environ["NAME"]
cfg["experiment"]["output_dir"] = os.environ["OUT_DIR"]
cfg["dataset"]["path"] = data_dir
cfg["dataset"]["base_path"] = f"{data_dir}/base.fbin"
cfg["dataset"]["query_path"] = f"{data_dir}/query.fbin"
cfg["dataset"]["ground_truth_path"] = f"{data_dir}/groundtruth.ibin"
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
        echo "base=$(cat data/yandexti100m/base.fbin.source)"
        echo "query=$(cat data/yandexti100m/query.fbin.source)"
        echo "groundtruth=$(cat data/yandexti100m/groundtruth.ibin.source)"
    } > "$STORAGE_TREE_DIR/index_confidence_tree_${TREE_INDEX}.sources"
else
    echo "Missing expected tree index: $TREE_INDEX_PATH"
fi
cp "$CONFIG_PATH" "$STORAGE_TREE_DIR/${RESULT_NAME}.yaml"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$STORAGE_TREE_DIR/"
fi
exit $RUN_STATUS
