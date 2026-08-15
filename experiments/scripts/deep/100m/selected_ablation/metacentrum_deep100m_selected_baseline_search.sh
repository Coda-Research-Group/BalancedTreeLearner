#!/bin/bash
#PBS -N batl_d100m_sel_base_s
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=16gb:mem=150gb:scratch_ssd=200gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_selected_baseline_search.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

RESULT_NAME=deep100m_selected_baseline
SOURCE_RESULT_NAME=deep100m_selected_baseline
NUM_TREES=4
MIN_TREES=2
BRANCHING_FACTOR=256
CONVERGENCE_PATIENCE=2
TOP_K_NEIGHBORS=100
NEIGHBOR_SEARCH_SUBSET=1000000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/selected_ablation/run_selected_search_common.sh
