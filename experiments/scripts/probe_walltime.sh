#!/bin/bash
#PBS -N probe_walltime_gpu
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=100gb:ngpus=1:gpu_mem=16gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/walltime_gpu.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd $SCRATCHDIR


DO_CPU=0
DO_GPU=1
GPU_DEVICE="cuda"
CYCLES=2


echo "Copying BATL repo..."
cp -r /auto/brno2/home/jozefsprlak/repos/batl2 ./BATL
mkdir $SCRATCHDIR/BATL/experiments/data

echo "Copying stuff"
SRC_DATA="/storage/brno2/home/jozefsprlak/repos/data"
cp $SRC_DATA/sift/sift-128-euclidean.hdf5 $SCRATCHDIR/BATL/experiments/data/sift-128-euclidean.hdf5
cp $SRC_DATA/glove/glove-100-angular.hdf5 $SCRATCHDIR/BATL/experiments/data/glove-100-angular.hdf5


while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only) DO_GPU=0; shift ;;
        --gpu-only) DO_CPU=0; shift ;;
        --device)   GPU_DEVICE="$2"; shift 2 ;;
        --cycles)   CYCLES="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

CONFIGS=(
    experiments/configs/sift1m/sift1m_h2_paper.yaml
    experiments/configs/sift1m/sift1m_h3_paper.yaml
    experiments/configs/sift1m/sift1m_h4_paper.yaml
    experiments/configs/glove100/glove100_h2_paper.yaml
    experiments/configs/glove100/glove100_h3_paper.yaml
    experiments/configs/glove100/glove100_h4_paper.yaml
)

module load mambaforge
mamba activate /storage/brno2/home/jozefsprlak/conda/batl
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl/bin/python"

cd BATL
if [[ "$DO_CPU" == 1 ]]; then
    echo "########### CPU PROBE ###########"
    python -u experiments/scripts/estimate_runtime.py \
        "${CONFIGS[@]}" --device cpu --cycles "$CYCLES" --batch-train 2048
fi

mamba activate /storage/brno2/home/jozefsprlak/conda/batl-gpu
PYTHON_EXEC="/storage/brno2/home/jozefsprlak/conda/batl-gpu/bin/python"

if [[ "$DO_GPU" == 1 ]]; then
    echo "########### GPU PROBE (${GPU_DEVICE}) ###########"
    python experiments/scripts/estimate_runtime.py \
        "${CONFIGS[@]}" --device "$GPU_DEVICE" --cycles "$CYCLES" --batch-train 4096
fi
