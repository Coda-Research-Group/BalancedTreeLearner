#!/bin/bash
#PBS -N batl_laion_clip768_100m_tree_2
# Build tree 2 of the exact SISAP 2023 LAION CLIP768v2 100M T=4 index.
# Tree 0 is the mandatory resource pilot; submit trees 1-3 only after reviewing it.
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:gpu_cap=sm_89:cluster=fobos:mem=700gb:scratch_ssd=400gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/laion_clip768_100m_tree_2.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -euo pipefail

TREE_INDEX=2
JOB_RELATIVE="experiments/scripts/laion5b/metacentrum_laion_clip768v2_100m_gpu_tree_2.sh"

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/laion5b/run_laion_clip768v2_100m_gpu_tree_common.sh
