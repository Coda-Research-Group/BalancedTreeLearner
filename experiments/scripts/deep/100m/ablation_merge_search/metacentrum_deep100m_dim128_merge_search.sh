#!/bin/bash
#PBS -N batl_deep100m_dim128_merge
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=200gb:scratch_ssd=200gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_dim128_merge_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

ABLATION_NAME="deep100m_ablation_embed_dim_128"
RESULT_NAME="${ABLATION_NAME}_merged_sweep"
STORAGE_TREE_DIR="/storage/brno2/home/jozefsprlak/results/batl_${ABLATION_NAME}_parallel_trees"
BRANCHING_FACTOR=256
EMBED_DIM=128
ALPHA=1.0
SEARCH_POINTS=(2 3 4 5 6 8 10 15 20 30 40)

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/ablation_merge_search/run_ablation_merge_search_common.sh
