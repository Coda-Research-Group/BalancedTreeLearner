import numpy as np
import pytest

from batl.tree import BATLTree, leaf_id_to_path, path_to_leaf_id


def test_leaf_id_path_round_trip() -> None:
    leaf_ids = np.arange(16)
    paths = leaf_id_to_path(leaf_ids, K=4, H=2)

    recovered = np.asarray([path_to_leaf_id(tuple(row), K=4) for row in paths])

    assert paths.dtype == np.uint16
    assert np.array_equal(recovered, leaf_ids)


def test_leaf_id_to_path_uses_most_significant_digit_first() -> None:
    path = leaf_id_to_path(np.array([1]), K=4, H=2)

    assert path.tolist() == [[0, 1]]


def test_random_init_balances_leaf_sizes() -> None:
    tree = BATLTree.random_init(N=18, K=4, H=2, alpha=1.0, seed=123)
    stats = tree.leaf_size_stats()

    assert tree.paths.shape == (18, 2)
    assert tree.paths.dtype == np.uint16
    assert stats["min"] == 1
    assert stats["max"] == 2
    assert stats["num_empty"] == 0


def test_random_init_is_seeded() -> None:
    first = BATLTree.random_init(N=20, K=4, H=2, alpha=1.0, seed=5)
    second = BATLTree.random_init(N=20, K=4, H=2, alpha=1.0, seed=5)
    third = BATLTree.random_init(N=20, K=4, H=2, alpha=1.0, seed=6)

    assert np.array_equal(first.paths, second.paths)
    assert not np.array_equal(first.paths, third.paths)


def test_get_leaf_indices_returns_indices_and_empty_array() -> None:
    paths = np.array([[0, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint16)
    tree = BATLTree(K=4, H=2, alpha=1.0, N=4, paths=paths)

    assert np.array_equal(tree.get_leaf_indices((0, 0)), np.array([0, 2], dtype=np.int32))
    assert tree.get_leaf_indices((3, 3)).dtype == np.int32
    assert tree.get_leaf_indices((3, 3)).size == 0


def test_leaf_id_lookup_matches_tuple_path_lookup() -> None:
    paths = np.array([[0, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint16)
    tree = BATLTree(K=4, H=2, alpha=1.0, N=4, paths=paths)

    assert np.array_equal(tree._get_leaf_indices_by_id(0), tree.get_leaf_indices((0, 0)))
    assert np.array_equal(tree._get_leaf_indices_by_id(5), tree.get_leaf_indices((1, 1)))
    assert tree._get_leaf_indices_by_id(15).dtype == np.int32
    assert tree._get_leaf_indices_by_id(15).size == 0


def test_leaf_id_lookup_recreates_cache_for_older_pickled_trees() -> None:
    paths = np.array([[0, 0], [0, 1], [0, 0], [1, 1]], dtype=np.uint16)
    tree = BATLTree(K=4, H=2, alpha=1.0, N=4, paths=paths)
    del tree._leaf_buckets_by_id

    assert np.array_equal(tree._get_leaf_indices_by_id(0), np.array([0, 2], dtype=np.int32))
    assert hasattr(tree, "_leaf_buckets_by_id")


def test_leaf_size_stats_includes_empty_leaves() -> None:
    paths = np.array([[0, 0], [0, 0], [1, 0]], dtype=np.uint16)
    tree = BATLTree(K=2, H=2, alpha=1.0, N=3, paths=paths)
    stats = tree.leaf_size_stats()

    assert stats["mean"] == pytest.approx(0.75)
    assert stats["std"] == pytest.approx(np.std([2, 0, 1, 0]))
    assert stats["min"] == 0
    assert stats["max"] == 2
    assert stats["num_empty"] == 2
    assert stats["gini"] == pytest.approx(7 / 12)


def test_tree_save_load_round_trip(tmp_path) -> None:
    tree = BATLTree.random_init(N=10, K=4, H=2, alpha=1.5, seed=9)
    path = tmp_path / "tree.pkl"

    tree.save(str(path))
    loaded = BATLTree.load(str(path))

    assert loaded.K == tree.K
    assert loaded.H == tree.H
    assert loaded.alpha == tree.alpha
    assert loaded.N == tree.N
    assert np.array_equal(loaded.paths, tree.paths)


def test_invalid_path_rejected() -> None:
    tree = BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0)

    with pytest.raises(ValueError):
        tree.get_leaf_indices((0,))


def test_rank_histogram_labels_cover_every_bucket_edge() -> None:
    from batl.tree_update import _RANK_HISTOGRAM_LABELS, RANK_HISTOGRAM_EDGES

    assert len(_RANK_HISTOGRAM_LABELS) == len(RANK_HISTOGRAM_EDGES)
    assert _RANK_HISTOGRAM_LABELS[:4] == ("rank_0", "rank_1", "rank_2", "rank_3")
    assert _RANK_HISTOGRAM_LABELS[4] == "rank_4_7"
    assert _RANK_HISTOGRAM_LABELS[-1] == "rank_256_plus"


def test_rank_histogram_buckets_ranks_by_edge() -> None:
    from batl.tree_update import _RANK_HISTOGRAM_LABELS, _rank_histogram_counts

    counts = _rank_histogram_counts(np.array([0, 0, 1, 3, 4, 7, 8, 300], dtype=np.int64))
    named = dict(zip(_RANK_HISTOGRAM_LABELS, counts.tolist(), strict=True))

    assert named["rank_0"] == 2
    assert named["rank_1"] == 1
    assert named["rank_3"] == 1
    assert named["rank_4_7"] == 2  # 4 and 7 share a bucket
    assert named["rank_8_15"] == 1
    assert named["rank_256_plus"] == 1
    assert int(counts.sum()) == 8


def test_min_top_r_covering_returns_edge_that_bounds_stragglers() -> None:
    from batl.tree_update import _min_top_r_covering, _rank_histogram_counts

    # 999 vectors at rank 0, one straggler at rank 40: R=4 already covers
    # 99.9%, so nothing below the next edge is needed.
    ranks = np.array([0] * 999 + [40], dtype=np.int64)
    assert _min_top_r_covering(_rank_histogram_counts(ranks), ranks.size) == 1

    # A heavy tail forces R past the bucket holding it.
    spread = np.array([0] * 500 + [20] * 500, dtype=np.int64)
    assert _min_top_r_covering(_rank_histogram_counts(spread), spread.size) == 32

    assert _min_top_r_covering(_rank_histogram_counts(np.array([], dtype=np.int64)), 0) == 0


def test_regroup_by_branch_preserves_input_order_within_each_child() -> None:
    from batl.tree_update import _regroup_by_branch

    vector_ids = np.arange(6, dtype=np.int64)
    branches = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)

    groups = _regroup_by_branch(vector_ids, branches)

    assert [(branch, ids.tolist()) for branch, ids in groups] == [
        (0, [1, 3, 5]),
        (1, [0, 2, 4]),
    ]


def test_tree_update_stats_exclude_fallback_sentinel_from_histogram() -> None:
    from batl.tree_update import _TreeUpdateStats

    stats = _TreeUpdateStats(assignment_order="input")
    # chosen_ranks carries the sentinel K for fallback rows; counting it would
    # fabricate a rank-8 choice that never happened.
    stats.record_batch(
        level=0,
        chosen_ranks=np.array([0, 1, 8], dtype=np.int32),
        used_fallback=np.array([False, False, True]),
        assignment_probs=np.array([0.9, 0.5, np.nan]),
    )
    level = stats.to_diagnostics().levels[0]

    assert level["num_vectors"] == 3
    assert level["fallback_count"] == 1
    assert level["rank_hist_rank_0"] == 1
    assert level["rank_hist_rank_1"] == 1
    assert level["rank_hist_rank_8_15"] == 0
    assert level["max_chosen_rank"] == 1
