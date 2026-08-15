"""Reproducibility helpers for BATL experiments."""

from __future__ import annotations

import os
import random

import faiss
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    thread_count = _thread_count_from_env()
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    faiss.omp_set_num_threads(thread_count)
    torch.set_num_threads(thread_count)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32 matmul on Ampere+ — PyTorch picks the precision; we just opt in.
    # No effect on pre-Ampere GPUs or on CPU.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _thread_count_from_env() -> int:
    value = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        thread_count = int(value)
    except ValueError:
        thread_count = 1
    return max(1, thread_count)
