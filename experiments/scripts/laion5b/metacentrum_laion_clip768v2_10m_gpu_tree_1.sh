#!/bin/bash
#PBS -N batl_laion_clip768_10m_tree_1
# Build tree 1 of the T=4 BATL ensemble on the exact SISAP 2023
# LAION-2B 10M CLIP768v2 data used by LMI.
# Tree 0 used 87.5 GiB RAM and 34:22 walltime; keep 28% RAM and 3.5x time headroom.
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:gpu_cap=sm_89:mem=112gb:scratch_local=50gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/laion_clip768_10m_tree_1.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail

TREE_INDEX=1
JOB_RELATIVE="experiments/scripts/laion5b/metacentrum_laion_clip768v2_10m_gpu_tree_1.sh"

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/laion5b/run_laion_clip768v2_10m_gpu_tree_common.sh
