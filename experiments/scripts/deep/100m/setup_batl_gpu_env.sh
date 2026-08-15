#!/bin/bash
# Install / refresh faiss-gpu in the batl-gpu conda env so Deep100M jobs can
# run on H100 (Hopper / SM 9.0). The previously installed conda-forge faiss-gpu
# build had no SM 9.0 kernels and failed at L2Norm.cu with CUDA error 209
# "no kernel image is available for execution on the device" on H100 NVL.
#
# The conda-forge faiss-gpu package was re-released in March 2026 with CUDA 12
# builds that ship Hopper kernels — this script pulls that build.
#
# Run once per cluster login (or after env recreation):
#     bash experiments/scripts/deep/100m/setup_batl_gpu_env.sh
#
# Tested combination:
#   CUDA driver: 13.2  (compatible with CUDA 12.x runtime via forward-compat)
#   torch:       cu121 wheel
#   faiss-gpu:   >=1.10.0 (conda-forge, March 2026; ships SM 8.0/8.6/8.9/9.0)

set -euo pipefail

ENV_PREFIX="${BATL_GPU_ENV:-/storage/brno2/home/jozefsprlak/conda/batl-gpu}"
PY="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

if [ ! -x "$PY" ]; then
    echo "ERROR: conda env not found at ${ENV_PREFIX}" >&2
    echo "       set BATL_GPU_ENV to override the prefix." >&2
    exit 1
fi

# mamba / conda must be on PATH for the env install. metacentrum exposes it via
# the mambaforge module — match the production job script.
if ! command -v mamba >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        echo "Loading mambaforge module..."
        module load mambaforge
    fi
fi
if ! command -v mamba >/dev/null 2>&1; then
    echo "ERROR: mamba not on PATH; load mambaforge before running this script." >&2
    exit 1
fi

echo "Target env: ${ENV_PREFIX}"
"$PY" --version
"$PY" -c "import sys; print('site-packages:', [p for p in sys.path if 'site-packages' in p][0])"

echo "Removing any pre-existing faiss installs (conda + pip)..."
mamba remove -y -p "${ENV_PREFIX}" faiss faiss-cpu faiss-gpu 2>/dev/null || true
"$PIP" uninstall -y faiss faiss-cpu faiss-gpu faiss-gpu-cu11 faiss-gpu-cu12 2>/dev/null || true

echo "Installing faiss-gpu from conda-forge (March 2026 build, SM 9.0 / Hopper kernels)..."
mamba install -y -p "${ENV_PREFIX}" -c conda-forge "faiss-gpu>=1.10.0"

echo "Verifying FAISS-GPU sees the CUDA device..."
"$PY" - <<'PY'
import faiss
import torch

print(f"faiss version: {faiss.__version__}")
print(f"faiss has StandardGpuResources: {hasattr(faiss, 'StandardGpuResources')}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
    cc = torch.cuda.get_device_capability(0)
    print(f"compute capability: sm_{cc[0]}{cc[1]}")
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, 8, cfg)
    import numpy as np
    index.add(np.random.rand(16, 8).astype("float32"))
    D, I = index.search(np.random.rand(2, 8).astype("float32"), 4)
    print(f"smoke search ok: D.shape={D.shape}, I.shape={I.shape}")
PY

echo "Done."
