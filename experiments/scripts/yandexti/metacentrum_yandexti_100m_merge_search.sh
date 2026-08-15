#!/bin/bash
#PBS -N batl_yandexti100m_merge_search
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=300gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/yandexti100m_merge_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

RESULT_NAME="yandexti100m_full_t4_parallel_merged"
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
    echo "Missing any source for $target" >&2
    return 2
}

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir -p data/yandexti100m results/"$RESULT_NAME"

echo "Copying YandexTI100M data..."
copy_first_existing data/yandexti100m/base.fbin "$SRC_DATA/base.100M.fbin" "$SRC_DATA/base.100m.fbin" || {
    echo "YandexTI100M requires the range-cropped first 100M vectors of base.1B.fbin with the header patched to 100000000." >&2
    exit 2
}
copy_first_existing data/yandexti100m/query.fbin "$SRC_DATA/query.public.100K.fbin" "$SRC_DATA/query.public.100k.fbin"
copy_first_existing data/yandexti100m/groundtruth.ibin "$SRC_DATA/groundtruth.100M.ibin" "$SRC_DATA/groundtruth.100m.ibin" || {
    echo "Missing converted groundtruth.100M.ibin -- run convert_yandexti_100m_groundtruth.py first." >&2
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
    raise SystemExit("CUDA is not visible to PyTorch; refusing search run.")
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
cfg["evaluation"]["beam_size"] = 300
cfg["evaluation"]["num_leaves"] = [100, 150, 200, 250, 300]
Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

cd BATL
MERGED_INDEX_PATH="$SCRATCHDIR/results/${RESULT_NAME}/index_confidence.pkl"
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
        --index-path "$MERGED_INDEX_PATH" \
        --result-dir "$SCRATCHDIR/results/${RESULT_NAME}" \
        --num-leaves 100 150 200 250 300 \
        --n-queries 100000 \
        --batch-search 25
    RUN_STATUS=$?
fi

cd "$SCRATCHDIR"
RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="/storage/brno2/home/jozefsprlak/results/batl_yandexti100m_full_t4_parallel_merged_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"

echo "Copying merged search results back to $OUTPUT_DIR"
if [ -d "results/${RESULT_NAME}" ]; then
    cp -r "results/${RESULT_NAME}" "$OUTPUT_DIR/"
fi
cp "$CONFIG_PATH" "$OUTPUT_DIR/"
exit $RUN_STATUS
