#!/bin/bash
#PBS -N batl_sift1m_dropout_00
#PBS -l select=1:ncpus=6:mem=16gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/sift1m_dropout_00.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# S5 treatment arm: dropout off.
#
# No dropout value appears anywhere in the BATL paper, and the 2026-05-07
# DECISION lists dropout: 0.1 among values that must not be presented as paper
# values. On a one-layer decoder trained ~2 epochs per cycle it may be
# regularising a model that is already under-trained.

ARM_NAME="dropout_00"
DROPOUT=0.0
BATCH_SIZE=8192

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/sift128/ablation/run_sift1m_ablation_common.sh
