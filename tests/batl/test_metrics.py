import numpy as np
import pytest

from batl.model import BATLModel
from batl.tree import BATLTree
from batl.utils.config import ModelConfig
from batl.utils.metrics import (
    estimate_candidate_set_size,
    index_size_mb,
    recall_at_k,
)


def test_recall_at_k_computes_mean_recall() -> None:
    retrieved = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    ground_truth = np.array([[3, 2, 9], [4, 8, 9]], dtype=np.int64)

    recall = recall_at_k(retrieved, ground_truth, k=2)

    assert recall == pytest.approx((1 / 2 + 1 / 2) / 2)


def test_recall_at_k_uses_ground_truth_top_k_only() -> None:
    retrieved = np.array([[7, 8]], dtype=np.int64)
    ground_truth = np.array([[1, 2, 7, 8]], dtype=np.int64)

    assert recall_at_k(retrieved, ground_truth, k=2) == 0.0


def test_recall_at_k_ignores_search_padding_sentinel() -> None:
    retrieved = np.array([[-1, 2]], dtype=np.int64)
    ground_truth = np.array([[2, 3]], dtype=np.int64)

    assert recall_at_k(retrieved, ground_truth, k=2) == pytest.approx(1 / 2)


def test_recall_at_k_returns_per_query_distribution() -> None:
    retrieved = np.array([[1, 2], [9, 9], [3, 4]], dtype=np.int64)
    ground_truth = np.array([[1, 2], [3, 4], [3, 4]], dtype=np.int64)

    per_query = recall_at_k(retrieved, ground_truth, k=2)

    assert per_query.shape == (3,)
    assert per_query.tolist() == [1.0, 0.0, 1.0]


def test_recall_at_k_vectorized_path_handles_duplicate_retrieved_ids() -> None:
    retrieved = np.array([[1, 1, 2]], dtype=np.int64)
    ground_truth = np.array([[1, 2, 3]], dtype=np.int64)

    per_query = recall_at_k(retrieved, ground_truth, k=3)

    assert per_query.tolist() == pytest.approx([2 / 3])


def test_index_size_mb_serializes_models_and_trees() -> None:
    tree = BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0)
    model = BATLModel(
        ModelConfig(
            branching_factor=2,
            tree_height=2,
            embedding_dim=3,
            encoder_hidden=4,
            embed_dim=4,
            num_heads=2,
            ff_dim=8,
            dropout=0.0,
            num_trees=1,
        )
    )

    size = index_size_mb(models=[model], trees=[tree])

    assert size > 0.0


def test_recall_at_k_raises_on_1d_input() -> None:
    retrieved = np.array([1, 2, 3], dtype=np.int64)
    ground_truth = np.array([1, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="2D"):
        recall_at_k(retrieved, ground_truth, k=2)


def test_recall_at_k_raises_on_row_mismatch() -> None:
    retrieved = np.array([[1, 2]], dtype=np.int64)
    ground_truth = np.array([[1, 2], [3, 4]], dtype=np.int64)

    with pytest.raises(ValueError, match="same number of queries"):
        recall_at_k(retrieved, ground_truth, k=2)


def test_recall_at_k_raises_when_k_too_large() -> None:
    retrieved = np.array([[1, 2]], dtype=np.int64)
    ground_truth = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="at least k"):
        recall_at_k(retrieved, ground_truth, k=5)


def test_estimate_candidate_set_size() -> None:
    tree = BATLTree.random_init(N=16, K=2, H=2, alpha=1.0, seed=0)

    size = estimate_candidate_set_size([tree], num_return_leaves=2)

    assert size > 0.0
