import random

import faiss
import numpy as np
import torch

from batl.utils.reproducibility import set_seed


def test_set_seed_makes_rngs_deterministic() -> None:
    set_seed(123)
    first = (random.random(), np.random.random(), torch.rand(3))

    set_seed(123)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_pins_worker_threads_from_omp_env(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    set_seed(123)

    assert faiss.omp_get_max_threads() == 1
    assert torch.get_num_threads() == 1
