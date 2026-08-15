#!/bin/bash
#PBS -N batl_spacev100m_tree_2
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=18:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/spacev100m_tree_2.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

TREE_INDEX=2
source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/spacev/run_spacev_100m_tree_common.sh
