#!/bin/bash
#PBS -N batl_d100m_lbl_search_subset_10m
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_label_search_subset_10m.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Reads the index the build job persisted. Re-runnable without rebuilding.

ARM_NAME="subset_10m"
NEIGHBOR_SEARCH_SUBSET=10000000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/label_ablation/run_deep100m_label_search_common.sh
