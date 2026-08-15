#!/bin/bash
#PBS -N batl_deep10m_label_1pct
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=16gb:mem=64gb:scratch_local=100gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep10m_label_1pct.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Baseline arm: the inherited 1%-of-N mining subset (100k of 10M). Labels are
# roughly rank-10,000 true neighbours.
ARM_NAME="subset_1pct"
NEIGHBOR_SEARCH_SUBSET=100000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/10m/label_ablation/run_deep10m_label_ablation_common.sh
