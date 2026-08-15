#!/bin/bash
#PBS -N batl_laion_clip768_100m_prepare
# Convert the exact SISAP 2023 HDF5 database once to a reusable float32 NPY memmap.
#PBS -l select=1:ncpus=4:mem=32gb:scratch_ssd=350gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/laion_clip768_100m_prepare.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail

SOURCE_REPO="/auto/brno2/home/jozefsprlak/repos/batl2"
SOURCE_DATA="/storage/brno2/home/jozefsprlak/repos/data/laion5b"
STORAGE_RESULT_DIR="/storage/brno2/home/jozefsprlak/results"
ENV_PREFIX="/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128"
SOURCE_NAME="laion2B-en-clip768v2-n=100M.h5"
OUTPUT_NAME="laion2B-en-clip768v2-n=100M-f32.npy"
MANIFEST_NAME="laion2B-en-clip768v2-n=100M-f32.manifest.json"
SOURCE_MD5="9d8ee3347b1edf136b3ef38162ac05c3"
QUERY_MD5="257b9eb3f7f25776e0d33b22451b7b32"
GROUND_TRUTH_MD5="35de58992c6446c85c56e710b144c90c"
EXPECTED_ROWS=102144212
EXPECTED_DIM=768
CONVERSION_CHUNK_ROWS=65536

: "${SCRATCHDIR:?PBS did not provide SCRATCHDIR}"
trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd "$SCRATCHDIR"

PBS_JOB_ID_SAFE="${PBS_JOBID:-manual}"
PBS_JOB_ID_SAFE="${PBS_JOB_ID_SAFE//[^[:alnum:]._-]/_}"
printf 'pbs_job_id=%s\n' "$PBS_JOB_ID_SAFE"

echo "Copying BATL checkout..."
cp -r "$SOURCE_REPO" ./BATL
module load mambaforge
mamba activate "$ENV_PREFIX"
PYTHON_EXEC="$ENV_PREFIX/bin/python"
export PYTHONPATH="$SCRATCHDIR/BATL"

FINAL_OUTPUT="$SOURCE_DATA/$OUTPUT_NAME"
FINAL_MANIFEST="$SOURCE_DATA/$MANIFEST_NAME"
if [ -f "$FINAL_OUTPUT" ] && [ -f "$FINAL_MANIFEST" ]; then
    echo "A complete conversion already exists; verifying it before exiting..."
    "$PYTHON_EXEC" -m experiments.utils.prepare_laion_memmap verify \
        --output "$FINAL_OUTPUT" \
        --manifest "$FINAL_MANIFEST" \
        --expected-source-md5 "$SOURCE_MD5" \
        --expected-shape "$EXPECTED_ROWS" "$EXPECTED_DIM"
    exit 0
fi
if [ -e "$FINAL_OUTPUT" ] || [ -e "$FINAL_MANIFEST" ]; then
    echo "Refusing incomplete final state: $FINAL_OUTPUT / $FINAL_MANIFEST" >&2
    exit 2
fi

for required in \
    "$SOURCE_DATA/$SOURCE_NAME" \
    "$SOURCE_DATA/public-queries-10k-clip768v2.h5" \
    "$SOURCE_DATA/laion2B-en-public-gold-standard-v2-100M.h5"; do
    if [ ! -f "$required" ]; then
        echo "Required SISAP 2023 file is missing: $required" >&2
        exit 2
    fi
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
require_md5 "$SOURCE_DATA/public-queries-10k-clip768v2.h5" "$QUERY_MD5"
require_md5 \
    "$SOURCE_DATA/laion2B-en-public-gold-standard-v2-100M.h5" \
    "$GROUND_TRUTH_MD5"

if GIT_COMMIT=$(git -C "$SCRATCHDIR/BATL" rev-parse HEAD 2>/dev/null); then
    :
else
    GIT_COMMIT="unknown"
fi

SCRATCH_OUTPUT="$SCRATCHDIR/$OUTPUT_NAME"
SCRATCH_MANIFEST="$SCRATCHDIR/$MANIFEST_NAME"
echo "Converting $EXPECTED_ROWS x $EXPECTED_DIM float16 rows to float32 memmap..."
"$PYTHON_EXEC" -u -m experiments.utils.prepare_laion_memmap convert \
    --source "$SOURCE_DATA/$SOURCE_NAME" \
    --output "$SCRATCH_OUTPUT" \
    --manifest "$SCRATCH_MANIFEST" \
    --key emb \
    --expected-md5 "$SOURCE_MD5" \
    --expected-shape "$EXPECTED_ROWS" "$EXPECTED_DIM" \
    --chunk-rows "$CONVERSION_CHUNK_ROWS" \
    --pbs-job-id "${PBS_JOBID:-unknown}" \
    --git-commit "$GIT_COMMIT"
"$PYTHON_EXEC" -u -m experiments.utils.prepare_laion_memmap verify \
    --output "$SCRATCH_OUTPUT" \
    --manifest "$SCRATCH_MANIFEST" \
    --expected-source-md5 "$SOURCE_MD5" \
    --expected-shape "$EXPECTED_ROWS" "$EXPECTED_DIM"

PARTIAL_OUTPUT="$FINAL_OUTPUT.partial.${PBS_JOB_ID_SAFE}"
PARTIAL_MANIFEST="$FINAL_MANIFEST.partial.${PBS_JOB_ID_SAFE}"
if [ -e "$PARTIAL_OUTPUT" ] || [ -e "$PARTIAL_MANIFEST" ]; then
    echo "Job-scoped partial artifact already exists; refusing overwrite." >&2
    exit 2
fi

echo "Copying conversion to persistent job-scoped partial files..."
cp "$SCRATCH_OUTPUT" "$PARTIAL_OUTPUT"
cp "$SCRATCH_MANIFEST" "$PARTIAL_MANIFEST"
"$PYTHON_EXEC" -u -m experiments.utils.prepare_laion_memmap verify \
    --output "$PARTIAL_OUTPUT" \
    --manifest "$PARTIAL_MANIFEST" \
    --expected-source-md5 "$SOURCE_MD5" \
    --expected-shape "$EXPECTED_ROWS" "$EXPECTED_DIM"

# Publish data first and manifest last. Consumers require both, so they never
# accept an incomplete copy. GNU mv -n prevents a concurrent job overwrite.
mv -n "$PARTIAL_OUTPUT" "$FINAL_OUTPUT"
if [ -e "$PARTIAL_OUTPUT" ]; then
    echo "Final data appeared concurrently; partial remains at $PARTIAL_OUTPUT." >&2
    exit 2
fi
mv -n "$PARTIAL_MANIFEST" "$FINAL_MANIFEST"
if [ -e "$PARTIAL_MANIFEST" ]; then
    echo "Final manifest appeared concurrently; partial remains at $PARTIAL_MANIFEST." >&2
    exit 2
fi

RESULT_TS=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="$STORAGE_RESULT_DIR/batl_laion_clip768v2_100m_prepare_$RESULT_TS"
mkdir -p "$OUTPUT_DIR"
cp "$FINAL_MANIFEST" "$OUTPUT_DIR/"
{
    echo "pbs_job_id=${PBS_JOBID:-unknown}"
    echo "git_commit=$GIT_COMMIT"
    echo "source_md5=$SOURCE_MD5"
    echo "output_path=$FINAL_OUTPUT"
    echo "manifest_path=$FINAL_MANIFEST"
    echo "run_status=0"
} > "$OUTPUT_DIR/job_summary.txt"

echo "Published verified memmap: $FINAL_OUTPUT"
