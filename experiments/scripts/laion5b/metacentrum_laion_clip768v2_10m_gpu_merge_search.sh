#!/bin/bash
#PBS -N batl_laion_clip768_10m_t4_search
# Merge the four independently trained LAION-2B CLIP768v2 trees, then run the
# versioned T=4 recall/QPS sweep. Submit only after all four tree jobs succeed.
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:gpu_cap=sm_89:mem=128gb:scratch_local=100gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/laion_clip768_10m_t4_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail

RESULT_NAME="laion_clip768v2_10m_gpu_t4_search"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_laion_clip768v2_10m_gpu_parallel_trees"
STORAGE_RESULT_DIR="/storage/brno2/home/jozefsprlak/results"
SOURCE_REPO="/auto/brno2/home/jozefsprlak/repos/batl2"
SOURCE_DATA="/storage/brno2/home/jozefsprlak/repos/data/laion5b"
ENV_PREFIX="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128"
CONFIG_RELATIVE="experiments/configs/laion5b/laion5b_10m_clip768v2_lmi.yaml"
JOB_RELATIVE="experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_merge_search.sh"

: "${SCRATCHDIR:?PBS did not provide SCRATCHDIR}"
trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

EXPECTED_GPU_NAME="NVIDIA L40S"
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
ACTUAL_GPU_NAME="${GPU_NAMES[0]:-unknown}"
echo "Job ${PBS_JOBID:-unknown} on $(hostname -f): GPU=$ACTUAL_GPU_NAME"
if [ "$ACTUAL_GPU_NAME" != "$EXPECTED_GPU_NAME" ]; then
    echo "Expected $EXPECTED_GPU_NAME for hardware-comparable runs, got $ACTUAL_GPU_NAME." >&2
    exit 2
fi

resolve_tree_index() {
    local tree_index="$1"
    local override_name="TREE_${tree_index}_INDEX"
    local explicit_path="${!override_name:-}"
    local matches=()

    if [ -n "$explicit_path" ]; then
        if [ ! -f "$explicit_path" ]; then
            echo "$override_name does not name a file: $explicit_path" >&2
            return 2
        fi
        printf '%s\n' "$explicit_path"
        return 0
    fi

    shopt -s nullglob
    matches=(
        "$STORAGE_TREE_DIR"/tree_"${tree_index}"_*/index_confidence_tree_"${tree_index}".pkl
    )
    shopt -u nullglob
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "Tree $tree_index needs exactly one saved index; found ${#matches[@]}." >&2
        echo "Set $override_name to the intended index path when reruns exist." >&2
        printf 'candidate: %s\n' "${matches[@]}" >&2
        return 2
    fi
    printf '%s\n' "${matches[0]}"
}

summary_value() {
    local summary_path="$1"
    local key="$2"
    awk -F= -v wanted="$key" '$1 == wanted {print substr($0, index($0, "=") + 1)}' \
        "$summary_path"
}

TREE_PATHS=()
REFERENCE_CONFIG=""
REFERENCE_COMMIT=""
for tree_index in 0 1 2 3; do
    # Optional explicit overrides are TREE_${tree_index}_INDEX.
    TREE_PATH=$(resolve_tree_index "$tree_index")
    TREE_DIR=$(dirname "$TREE_PATH")
    SUMMARY_PATH="$TREE_DIR/job_summary.txt"
    TREE_CONFIG="$TREE_DIR/submitted_config.yaml"
    TREE_METRICS="$TREE_DIR/metrics.json"
    TREE_HARDWARE="$TREE_DIR/hardware.json"

    if [ ! -f "$SUMMARY_PATH" ] || [ ! -f "$TREE_CONFIG" ] \
        || [ ! -f "$TREE_METRICS" ] || [ ! -f "$TREE_HARDWARE" ]; then
        echo "Tree $tree_index lacks its summary, config, metrics, or hardware: $TREE_DIR" >&2
        exit 2
    fi
    SUMMARY_STATUS=$(summary_value "$SUMMARY_PATH" "run_status")
    SUMMARY_TREE=$(summary_value "$SUMMARY_PATH" "tree_index")
    SUMMARY_COMMIT=$(summary_value "$SUMMARY_PATH" "git_commit")
    SUMMARY_GPU=$(summary_value "$SUMMARY_PATH" "gpu_name")
    if [ "$SUMMARY_STATUS" != "0" ] || [ "$SUMMARY_TREE" != "$tree_index" ]; then
        echo "Tree $tree_index has an invalid job summary: $SUMMARY_PATH" >&2
        exit 2
    fi
    if [ -n "$SUMMARY_GPU" ] && [ "$SUMMARY_GPU" != "$EXPECTED_GPU_NAME" ]; then
        echo "Tree $tree_index used $SUMMARY_GPU instead of $EXPECTED_GPU_NAME." >&2
        exit 2
    fi

    if [ -z "$REFERENCE_CONFIG" ]; then
        REFERENCE_CONFIG="$TREE_CONFIG"
        REFERENCE_COMMIT="$SUMMARY_COMMIT"
    else
        if ! cmp -s "$REFERENCE_CONFIG" "$TREE_CONFIG"; then
            echo "Tree $tree_index used a different submitted config: $TREE_CONFIG" >&2
            exit 2
        fi
        if [ "$SUMMARY_COMMIT" != "$REFERENCE_COMMIT" ]; then
            echo "WARNING: tree commits differ: tree 0=$REFERENCE_COMMIT, tree $tree_index=$SUMMARY_COMMIT" >&2
        fi
    fi

    echo "tree_$tree_index=$TREE_PATH commit=$SUMMARY_COMMIT"
    TREE_PATHS+=("$TREE_PATH")
done

echo "Copying BATL checkout and exact SISAP 2023 LAION-2B files..."
cp -r "$SOURCE_REPO" ./BATL
DATA_DIR="$SCRATCHDIR/BATL/data/laion5b"
RESULT_DIR="$SCRATCHDIR/results/$RESULT_NAME"
mkdir -p "$DATA_DIR" "$RESULT_DIR" "$STORAGE_RESULT_DIR"

for filename in \
    "laion2B-en-clip768v2-n=10M.h5" \
    "public-queries-10k-clip768v2.h5" \
    "laion2B-en-public-gold-standard-v2-10M.h5"; do
    if [ ! -f "$SOURCE_DATA/$filename" ]; then
        echo "Required dataset file is missing: $SOURCE_DATA/$filename" >&2
        exit 2
    fi
    cp "$SOURCE_DATA/$filename" "$DATA_DIR/$filename"
done

require_md5() {
    local path="$1"
    local expected="$2"
    local digest_output
    local actual
    digest_output=$(md5sum "$path")
    actual="${digest_output%% *}"
    echo "$path: md5=$actual"
    if [ "$actual" != "$expected" ]; then
        echo "$path: expected md5 $expected, got $actual." >&2
        return 2
    fi
}

require_md5 "$DATA_DIR/laion2B-en-clip768v2-n=10M.h5" \
    "c05e4b1d2b2a0c7663ac9767753e25e1"
require_md5 "$DATA_DIR/public-queries-10k-clip768v2.h5" \
    "257b9eb3f7f25776e0d33b22451b7b32"
require_md5 "$DATA_DIR/laion2B-en-public-gold-standard-v2-10M.h5" \
    "b68b17693253d95e1fc94c217af25e95"

module load mambaforge
mamba activate "$ENV_PREFIX"
PYTHON_EXEC="$ENV_PREFIX/bin/python"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPATH="$SCRATCHDIR/BATL"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SCRATCHDIR/BATL"
CONFIG_PATH="$SCRATCHDIR/BATL/$CONFIG_RELATIVE"
MERGED_INDEX="$RESULT_DIR/index_confidence.pkl"
if ! cmp -s "$REFERENCE_CONFIG" "$CONFIG_PATH"; then
    echo "Saved tree config differs from the current versioned config: $CONFIG_PATH" >&2
    exit 2
fi

echo "Checking CUDA, FAISS, the config, and the four single-tree payloads..."
"$PYTHON_EXEC" -u - \
    "$CONFIG_PATH" \
    "$RESULT_DIR/build_times.tsv" \
    "${TREE_PATHS[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import torch
from batl.utils.config_parsing import load_experiment_config
from batl.utils.index_parsing import load_index

cfg = load_experiment_config(sys.argv[1])
build_times_path = Path(sys.argv[2])
if cfg.model.num_trees != 4:
    raise SystemExit(f"Expected num_trees=4, got {cfg.model.num_trees}.")
build_rows = []
for tree_index, path in enumerate(sys.argv[3:]):
    models, trees = load_index(path)
    if len(models) != 1 or len(trees) != 1:
        raise SystemExit(
            f"{path}: expected exactly one model/tree, got {len(models)}/{len(trees)}."
        )
    tree = trees[0]
    if (tree.K, tree.H, tree.N) != (
        cfg.model.branching_factor,
        cfg.model.tree_height,
        cfg.subset_size,
    ):
        raise SystemExit(
            f"{path}: incompatible tree K/H/N={(tree.K, tree.H, tree.N)}."
        )
    summary = {}
    for line in (Path(path).parent / "job_summary.txt").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            summary[key] = value
    expected_seed = cfg.seed + tree_index
    if summary.get("seed") != str(expected_seed):
        raise SystemExit(
            f"{path}: expected seed {expected_seed}, got {summary.get('seed')!r}."
        )
    metrics = json.loads((Path(path).parent / "metrics.json").read_text())
    hardware = json.loads((Path(path).parent / "hardware.json").read_text())
    gpu_name = hardware.get("gpu_name")
    if gpu_name != "NVIDIA L40S":
        raise SystemExit(f"{path}: expected NVIDIA L40S hardware, got {gpu_name!r}.")
    train_time_s = metrics.get("train_time_s")
    if not isinstance(train_time_s, (int, float)) or train_time_s <= 0:
        raise SystemExit(f"{path}: invalid train_time_s={train_time_s!r}.")
    build_rows.append(
        (
            tree_index,
            expected_seed,
            float(train_time_s),
            int(summary["build_command_wall_time_s"]),
            gpu_name,
            path,
        )
    )

with build_times_path.open("w", encoding="utf-8") as handle:
    handle.write("tree_index\tseed\ttrain_time_s\tcommand_wall_time_s\tgpu_name\tindex_path\n")
    for row in build_rows:
        handle.write("\t".join(map(str, row)) + "\n")
    handle.write(
        "SUM_GPU_WORK\t-\t"
        f"{sum(row[2] for row in build_rows):.6f}\t"
        f"{sum(row[3] for row in build_rows)}\t-\t-\n"
    )
    handle.write(
        "IDEAL_PARALLEL_MAX\t-\t"
        f"{max(row[2] for row in build_rows):.6f}\t"
        f"{max(row[3] for row in build_rows)}\t-\t-\n"
    )

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch; refusing search run.")
print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
if torch.cuda.get_device_name(0) != "NVIDIA L40S":
    raise SystemExit("PyTorch is not using the required NVIDIA L40S.")
print(f"faiss imported from: {faiss.__file__}")
PY

{
    echo -e "tree_index\tindex_path\tsha256\tgit_commit"
    for tree_index in 0 1 2 3; do
        TREE_PATH="${TREE_PATHS[$tree_index]}"
        TREE_DIR=$(dirname "$TREE_PATH")
        SUMMARY_PATH="$TREE_DIR/job_summary.txt"
        SUMMARY_COMMIT=$(summary_value "$SUMMARY_PATH" "git_commit")
        CHECKSUM_OUTPUT=$(sha256sum "$TREE_PATH")
        TREE_SHA256="${CHECKSUM_OUTPUT%% *}"
        echo -e "$tree_index\t$TREE_PATH\t$TREE_SHA256\t$SUMMARY_COMMIT"
        mkdir -p "$RESULT_DIR/tree_${tree_index}_provenance"
        cp "$SUMMARY_PATH" "$RESULT_DIR/tree_${tree_index}_provenance/"
        cp "$TREE_DIR/submitted_config.yaml" "$RESULT_DIR/tree_${tree_index}_provenance/"
        cp "$TREE_DIR/metrics.json" "$RESULT_DIR/tree_${tree_index}_provenance/"
        for artifact_name in hardware.json environment.json seed.txt; do
            if [ -f "$TREE_DIR/$artifact_name" ]; then
                cp "$TREE_DIR/$artifact_name" "$RESULT_DIR/tree_${tree_index}_provenance/"
            fi
        done
        if [ -f "$TREE_DIR/git_status.txt" ]; then
            cp "$TREE_DIR/git_status.txt" "$RESULT_DIR/tree_${tree_index}_provenance/"
        fi
    done
} > "$RESULT_DIR/tree_inputs.tsv"

if SEARCH_GIT_COMMIT=$(git -C "$SCRATCHDIR/BATL" rev-parse HEAD 2>/dev/null); then
    :
else
    SEARCH_GIT_COMMIT="unknown"
fi
git -C "$SCRATCHDIR/BATL" status --short > "$RESULT_DIR/search_git_status.txt" || true

RUN_STATUS=0
MERGE_START_EPOCH=$(date +%s)
set +e
"$PYTHON_EXEC" -u merge_index.py \
    --output "$MERGED_INDEX" \
    "${TREE_PATHS[@]}"
RUN_STATUS=$?
set -e
MERGE_END_EPOCH=$(date +%s)

SEARCH_START_EPOCH=0
SEARCH_END_EPOCH=0
if [ "$RUN_STATUS" -eq 0 ]; then
    echo "Searching the four-tree ensemble at $(date)..."
    nvidia-smi
    SEARCH_START_EPOCH=$(date +%s)
    set +e
    "$PYTHON_EXEC" -u search.py \
        "$CONFIG_PATH" \
        --log \
        --index-path "$MERGED_INDEX" \
        --result-dir "$RESULT_DIR" \
        --batch-search 50
    RUN_STATUS=$?
    set -e
    SEARCH_END_EPOCH=$(date +%s)
fi

{
    echo "pbs_job_id=${PBS_JOBID:-unknown}"
    echo "search_git_commit=$SEARCH_GIT_COMMIT"
    echo "gpu_name=$ACTUAL_GPU_NAME"
    echo "merge_wall_time_s=$((MERGE_END_EPOCH - MERGE_START_EPOCH))"
    if [ "$SEARCH_START_EPOCH" -gt 0 ]; then
        echo "search_command_wall_time_s=$((SEARCH_END_EPOCH - SEARCH_START_EPOCH))"
    fi
    echo "run_status=$RUN_STATUS"
} > "$RESULT_DIR/job_summary.txt"

RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="$STORAGE_RESULT_DIR/batl_${RESULT_NAME}_${RESULT_TS}"
mkdir -p "$OUTPUT_DIR"
echo "Copying merged index, search results, and provenance to $OUTPUT_DIR"
cp -r "$RESULT_DIR/." "$OUTPUT_DIR/"
cp "$CONFIG_PATH" "$OUTPUT_DIR/submitted_config.yaml"
cp "$SCRATCHDIR/BATL/$JOB_RELATIVE" "$OUTPUT_DIR/submitted_job.sh"

if [ "$RUN_STATUS" -ne 0 ]; then
    echo "Merge/search failed with status $RUN_STATUS; partial artifacts were preserved." >&2
fi
exit "$RUN_STATUS"
