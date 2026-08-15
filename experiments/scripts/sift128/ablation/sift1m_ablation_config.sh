#!/bin/bash
# Single source of truth for the SIFT1M one-knob ablations (tree-quality
# suspects S0 and S5 in docs/tree_quality_suspects.md).
#
# The config is produced by patching experiments/configs/sift1m/
# sift1m_h2_paper.yaml rather than being written out in full, so the paper
# config stays the source of truth for every value nobody is varying and the
# difference between two arms is literally one assignment below.
#
# Requires: RESULT_NAME, CONFIG_PATH, RESULT_DIR, DATA_PATH, PYTHON_EXEC,
#           SRC_CONFIG, DROPOUT, BATCH_SIZE

write_ablation_config() {
    NAME="$RESULT_NAME" SRC="$SRC_CONFIG" OUT="$CONFIG_PATH" RES="$RESULT_DIR" \
    DATA="$DATA_PATH" DROPOUT="$DROPOUT" BATCH_SIZE="$BATCH_SIZE" \
    "$PYTHON_EXEC" -u - <<'PY'
import os
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(os.environ["SRC"]).read_text())

cfg["experiment"]["name"] = os.environ["NAME"]
cfg["experiment"]["output_dir"] = os.environ["RES"]
cfg["dataset"]["path"] = os.environ["DATA"]
cfg["dataset"]["metric"] = "euclidean"
cfg["dataset"]["storage_mode"] = "preload"

# --- the two knobs the ablations vary ------------------------------------
# S5 varies dropout; S0 varies batch_size. Each wrapper sets exactly one of
# them away from the value the plotted curves used, so a single run pair is
# always a one-variable comparison.
cfg["model"]["dropout"] = float(os.environ["DROPOUT"])
cfg["training"]["batch_size"] = int(os.environ["BATCH_SIZE"])

# --- held fixed across every arm -----------------------------------------
# Every arm stops on the same convergence rule rather than a pinned cycle
# count, so arms may end on different cycles. S4 showed cycle count does not
# move the curve on its own, so the comparison stays one-variable in
# practice; read arm-to-arm differences against the logged cycle counts.
cfg["training"]["convergence_patience"] = 2
cfg["training"]["convergence_min_delta"] = 0.005

cfg["training"]["device"] = "cpu"
cfg["training"]["neighbor_search_backend"] = "faiss_cpu"
cfg["training"]["tree_update_cache_embeddings"] = False
cfg["evaluation"]["rerank_backend"] = "numpy_cpu"

Path(os.environ["OUT"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(
    f"config: dropout={cfg['model']['dropout']} "
    f"batch_size={cfg['training']['batch_size']} "
    f"patience={cfg['training']['convergence_patience']}"
)
PY
}
