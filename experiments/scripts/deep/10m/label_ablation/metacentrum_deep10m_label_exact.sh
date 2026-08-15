#!/bin/bash
#PBS -N batl_deep10m_label_exact
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=24gb:mem=64gb:scratch_local=100gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep10m_label_exact.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Treatment arm: mine against the whole 10M database, so labels are the true
# top-100 neighbours. gpu_mem is raised to 24gb because the FAISS flat index
# holds 10M x 96 float32 = 3.6 GiB on the GPU alongside the routing model and
# tree-update batches.
ARM_NAME="exact"
NEIGHBOR_SEARCH_SUBSET=10000000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/10m/label_ablation/run_deep10m_label_ablation_common.sh
