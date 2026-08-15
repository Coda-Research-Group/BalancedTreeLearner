#!/bin/bash
#PBS -N batl_d100m_lbl_build_subset_10m
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_label_build_subset_10m.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# 10% of the database: labels approximate rank-1,000 true neighbours.
# FAISS flat index is 3.58 GiB so a 16gb card fits; mining ~3 min/cycle.

# walltime=12h: the build is ~3.8-4.3 h on an L40S-class node (tree update is
# ~70% of each cycle and scales with N, not with the mining subset). 12 h covers
# a node up to ~2.8x slower. With num_trees=1 nothing is written until training
# finishes, so a walltime kill loses the entire build.

ARM_NAME="subset_10m"
NEIGHBOR_SEARCH_SUBSET=10000000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/label_ablation/run_deep100m_label_build_common.sh
