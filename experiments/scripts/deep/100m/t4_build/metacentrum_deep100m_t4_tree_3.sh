#!/bin/bash
#PBS -N batl_d100m_t4_t3
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_ssd=200gb
#PBS -l walltime=16:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_t4_tree_3.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Tree 3 of the Deep100M T=4 headline ensemble.
#
# walltime=16h: the 1% label arm built one tree in 3.7h, and a contended node
# cost 6x on job 22764327. One tree is cheap to resubmit; four in one job are
# not — which is the whole reason this is split.
#
# gpu_mem=16gb: mining indexes a 1M sample (0.36 GiB) and the tree update
# decodes in 8191-row batches, so no large card is needed. Only the resident
# rerank control needs 44gb.

TREE_INDEX=3

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/t4_build/run_deep100m_t4_tree_common.sh
