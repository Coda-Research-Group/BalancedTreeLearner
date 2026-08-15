#!/bin/bash
#PBS -N batl_sift1m_b256_t2
#PBS -l select=1:ncpus=6:mem=16gb:scratch_local=50gb
#PBS -l walltime=20:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/sift1m_batch256_tree_2.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# S0 treatment arm, tree 2 of 4. Batch 256 is the SPEC default; the wrappers
# that produced every plotted 1M curve patch it to 8192.
#
# walltime=20h: the batch-8192 arm took ~50 min/tree on an uncontended node,
# and batch 256 runs 31.8x more optimizer steps per epoch. A contended node
# cost 6x on job 22764327, so the margin is deliberate — one tree is cheap to
# resubmit, a four-tree job is not.

TREE_INDEX=2
ARM_NAME="batch256"
BATCH_SIZE=256

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/sift128/batch_ablation/run_sift1m_batch_tree_common.sh
