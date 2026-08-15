#!/bin/bash
#PBS -N batl_d100m_sel_topk
#PBS -J 0-11
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_ssd=200gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_selected_topk_build^array_index^.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u
TOP_K_VALUES=(10 25 50)
ARRAY_INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required}"
ARM_INDEX=$((ARRAY_INDEX / 4))
TREE_INDEX=$((ARRAY_INDEX % 4))
TOP_K_NEIGHBORS="${TOP_K_VALUES[$ARM_INDEX]}"
ARM_NAME="topk_${TOP_K_NEIGHBORS}"
NUM_TREES=4
BRANCHING_FACTOR=256
CONVERGENCE_PATIENCE=2
NEIGHBOR_SEARCH_SUBSET=1000000

if [ "${BATL_ARRAY_DRY_RUN:-0}" = 1 ]; then
    printf 'ARM_NAME=%s\nTREE_INDEX=%s\n' "$ARM_NAME" "$TREE_INDEX"
    exit 0
fi

RESULT_NAME="deep100m_selected_${ARM_NAME}"
source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/selected_ablation/run_selected_tree_common.sh
