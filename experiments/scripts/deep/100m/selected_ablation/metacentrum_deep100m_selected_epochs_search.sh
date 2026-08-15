#!/bin/bash
#PBS -N batl_d100m_sel_epoch_s
#PBS -J 0-3
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=16gb:mem=150gb:scratch_ssd=200gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_selected_epochs_search^array_index^.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u
EPOCH_VALUES=(2 4 10 30)
CYCLE_VALUES=(1 2 5 15)
ARRAY_INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required}"
EPOCHS="${EPOCH_VALUES[$ARRAY_INDEX]}"
MAX_ALTERNATING_CYCLES="${CYCLE_VALUES[$ARRAY_INDEX]}"
CONVERGENCE_PATIENCE=0
ARM_NAME="epochs_${EPOCHS}"
SOURCE_RESULT_NAME="deep100m_selected_${ARM_NAME}"
NUM_TREES=4
MIN_TREES=2
BRANCHING_FACTOR=256
TOP_K_NEIGHBORS=100
NEIGHBOR_SEARCH_SUBSET=1000000

if [ "${BATL_ARRAY_DRY_RUN:-0}" = 1 ]; then
    printf 'ARM_NAME=%s\nNUM_TREES=%s\nMIN_TREES=%s\n' "$ARM_NAME" "$NUM_TREES" "$MIN_TREES"
    exit 0
fi

RESULT_NAME="deep100m_selected_${ARM_NAME}"
source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/selected_ablation/run_selected_search_common.sh
