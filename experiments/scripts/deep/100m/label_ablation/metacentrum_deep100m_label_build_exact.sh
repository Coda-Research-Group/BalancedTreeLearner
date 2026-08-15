#!/bin/bash
#PBS -N batl_d100m_lbl_build_exact
#PBS -l select=1:ncpus=16:ngpus=1:gpu_mem=16gb:mem=300gb:scratch_local=200gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/deep100m_label_build_exact.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

# Exact labels over the whole 100M database.
#
#   gpu_mem=16gb  - matches the other two arms. An earlier revision asked for
#     44gb because the flat index over all 100M vectors is 35.76 GiB, and job
#     22710911 still died on it: FAISS grows its device buffer geometrically
#     and copies, so the peak exceeds the resident size and a 44 GiB card runs
#     out mid-build. Neighbour mining now searches chunk by chunk and merges
#     the running top-k, so the device holds one chunk at a time
#     (10M x 96 x 4 B = 3.84 GB) regardless of subset size.
#   walltime=24h  - mining is ~10x the 10m arm's 160 s/cycle, so ~27 min x 10
#     on top of the ~5 h everything else takes: expect ~10 h.

ARM_NAME="exact"
NEIGHBOR_SEARCH_SUBSET=100000000
# Only this arm takes the chunked path, so only this arm reads the knob. Ten
# rounds of 10M beats a hundred rounds of 1M: the distance work is identical
# either way, but each round costs an index build and a top-k merge.
NEIGHBOR_SEARCH_CHUNK_SIZE=10000000

source /auto/brno2/home/jozefsprlak/repos/batl2/experiments/scripts/deep/100m/label_ablation/run_deep100m_label_build_common.sh
