#!/bin/bash
#PBS -N batl_d100m_lbl_search_exact
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_label_search_exact.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Reads the index the build job persisted. Re-runnable without rebuilding.
# numpy_cpu rerank is pinned, so this phase does not need the big GPU.

ARM_NAME="exact"
NEIGHBOR_SEARCH_SUBSET=100000000
# Inert here — search does no neighbour mining — but the config is regenerated
# per phase, so leaving it unset would make the search artifact record a
# different chunk size than the build actually used.
NEIGHBOR_SEARCH_CHUNK_SIZE=10000000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/label_ablation/run_deep100m_label_search_common.sh
