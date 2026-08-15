#!/bin/bash
#PBS -N batl_sift1m_dropout_01
#PBS -l select=1:ncpus=6:mem=16gb:scratch_local=50gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/sift1m_dropout_01.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# S5 baseline arm: dropout 0.1, the value the plotted curves used.
#
# This arm is NOT redundant with the existing metacentrum_sift128_cpu_h2.sh
# results. Those ran with early stopping live and a 20-cycle cap, so their
# cycle count is not pinned to this arm's. Comparing dropout 0.0 against them
# would confound dropout with however many cycles each happened to take.

ARM_NAME="dropout_01"
DROPOUT=0.1
# Held at the value the plotted curves used, so S5 measures dropout alone.
# S0 is the separate test of this knob.
BATCH_SIZE=8192

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/sift128/ablation/run_sift1m_ablation_common.sh
