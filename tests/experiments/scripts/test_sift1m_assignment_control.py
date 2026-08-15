import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from batl.model import BATLModel
from batl.tree import BATLTree
from batl.tree_update import update_tree
from batl.utils.config import ModelConfig
from batl.utils.index_parsing import load_index

RUNNER_PATH = Path("experiments/scripts/sift128/run_assignment_control.py")
WRAPPER_PATH = Path(
    "experiments/scripts/sift128/assignment_control/metacentrum_sift1m_assignment_control.sh"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("batl_sift_assignment_control", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree(paths: list[list[int]]) -> BATLTree:
    array = np.asarray(paths, dtype=np.uint16)
    return BATLTree(K=2, H=array.shape[1], alpha=1.0, N=array.shape[0], paths=array)


def test_assignment_arms_cover_the_full_kernel_x_order_grid() -> None:
    """Both dimensions, both levels — the first run showed they are separable.

    Confidence ordering was worth 1.43-1.89x on the round kernel while the
    round -> sequential switch cost 0.81-0.87x at input order, so a design that
    varies only one dimension at a time cannot attribute either effect.
    """
    runner = _load_runner()
    arms = {(arm.mode, arm.order): arm.name for arm in runner.ASSIGNMENT_ARMS}

    assert arms == {
        ("round", "confidence"): "round_confidence",
        ("round", "input"): "round_input",
        ("sequential", "input"): "sequential_input",
        ("sequential", "confidence"): "sequential_confidence",
    }
    # round_confidence must stay first: it is the arm the source-reproduction
    # guard checks, and a failure there invalidates the rest.
    assert runner.ASSIGNMENT_ARMS[0].name == "round_confidence"


def test_leaf_divergence_reports_per_tree_and_overall_fraction() -> None:
    runner = _load_runner()
    left = [_tree([[0], [0], [1], [1]]), _tree([[0], [1], [0], [1]])]
    right = [_tree([[0], [1], [1], [1]]), _tree([[1], [1], [0], [0]])]

    result = runner.leaf_divergence(left, right)

    assert result["per_tree_fraction"] == [0.25, 0.5]
    assert result["overall_fraction"] == 0.375
    assert result["changed_vectors"] == 3
    assert result["total_vectors"] == 8


def test_leaf_divergence_rejects_incompatible_ensembles() -> None:
    runner = _load_runner()

    try:
        runner.leaf_divergence([_tree([[0], [1]])], [])
    except ValueError as exc:
        assert "same number" in str(exc)
    else:
        raise AssertionError("leaf_divergence accepted mismatched ensemble lengths")


def test_reassign_index_creates_every_variant_from_one_model_state(tmp_path: Path) -> None:
    runner = _load_runner()
    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=1,
        embedding_dim=2,
        encoder_hidden=8,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        alpha=1.0,
        num_trees=1,
    )
    torch.manual_seed(7)
    model = BATLModel(model_cfg)
    vectors = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0]], dtype=np.float32)
    initial = BATLTree.random_init(N=4, K=2, H=1, alpha=1.0, seed=7)
    source = update_tree(
        model=model,
        vectors=vectors,
        current_tree=initial,
        batch_size=4,
        device=torch.device("cpu"),
        assignment_mode="round",
        assignment_order="confidence",
    )

    report = runner.reassign_index(
        models=[model],
        source_trees=[source],
        vectors=vectors,
        output_dir=tmp_path,
        batch_size=4,
        device=torch.device("cpu"),
        cache_embeddings=False,
        round_top_r=None,
    )

    assert report["source_round_confidence_exact_match"] is True
    assert set(report["arms"]) == {
        "round_confidence",
        "round_input",
        "sequential_input",
        "sequential_confidence",
    }
    for arm in runner.ASSIGNMENT_ARMS:
        index_path = Path(report["arms"][arm.name]["index_path"])
        assert index_path.is_file()
        models, trees = load_index(str(index_path))
        assert len(models) == len(trees) == 1
    # Kernel effect at each ordering and ordering effect at each kernel; one
    # pair alone cannot separate the two dimensions.
    assert set(report["partition_divergence"]) == {
        "round_input_vs_sequential_input",
        "round_confidence_vs_sequential_confidence",
        "round_input_vs_round_confidence",
        "sequential_input_vs_sequential_confidence",
    }


def test_metacentrum_wrapper_runs_every_shared_state_arm() -> None:
    text = WRAPPER_PATH.read_text(encoding="utf-8")

    assert "#PBS -l select=1:ncpus=6:mem=24gb:scratch_local=50gb" in text
    assert "#PBS -l walltime=12:00:00" in text
    assert "ngpus" not in text
    assert ': "${SOURCE_INDEX:?' in text
    assert '[ ! -f "$SOURCE_INDEX" ]' in text
    assert text.count("run_assignment_control.py") == 1
    assert '"round_confidence:round:confidence"' in text
    assert '"round_input:round:input"' in text
    assert '"sequential_input:sequential:input"' in text
    assert '"sequential_confidence:sequential:confidence"' in text
    assert 'cfg["evaluation"]["tree_assignment_mode"]' in text
    assert 'cfg["evaluation"]["tree_assignment_order"]' in text
    assert "search.py" in text
    assert "--num-leaves 10 20 40 80 100" in text
    assert 'cfg["training"]["device"] = "cpu"' in text
    assert 'cfg["training"]["neighbor_search_backend"] = "faiss_cpu"' in text
    assert 'cfg["evaluation"]["rerank_backend"] = "numpy_cpu"' in text


def test_source_guard_accepts_float_drift_but_rejects_real_mismatch() -> None:
    """Bit-identity is not achievable on a heterogeneous cluster.

    The same code, thread count and source index reproduced exactly on one
    MetaCentrum node (job 22788625) and drifted by 41 vectors in 4,000,000 on
    another (job 22791165), because a different CPU selects different BLAS
    kernels and so a different reduction order, which flips assignments
    wherever two branches are near-tied at a capacity boundary. The guard must
    tolerate that while still catching a harness decoding something else — a
    failure mode worth tens of percent.
    """
    runner = _load_runner()
    observed_drift = 41 / 4_000_000
    smallest_measured_effect = 0.2567  # round_input vs sequential_input

    assert observed_drift < runner.SOURCE_REPRODUCTION_TOLERANCE, (
        "drift actually observed on the cluster must pass the guard"
    )
    assert runner.SOURCE_REPRODUCTION_TOLERANCE < smallest_measured_effect / 100, (
        "tolerance must stay far below the smallest effect the control measures, "
        "so it cannot mask a genuinely wrong partition"
    )


def test_reassign_index_requires_the_reproduction_arm(tmp_path: Path) -> None:
    """Dropping round_confidence would leave nothing checking the harness.

    Deep100M runs a subset of the grid because sequential mode cannot scale to
    K=256, so arm selection is a real code path — and the one arm it must never
    drop is the one compared against the cached source tree.
    """
    runner = _load_runner()
    model_cfg = ModelConfig(
        branching_factor=2,
        tree_height=1,
        embedding_dim=2,
        encoder_hidden=8,
        embed_dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        alpha=1.0,
        num_trees=1,
    )
    torch.manual_seed(3)
    model = BATLModel(model_cfg)
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    tree = BATLTree.random_init(N=2, K=2, H=1, alpha=1.0, seed=3)

    with pytest.raises(ValueError, match="round_confidence must be among the arms"):
        runner.reassign_index(
            models=[model],
            source_trees=[tree],
            vectors=vectors,
            output_dir=tmp_path,
            batch_size=2,
            device=torch.device("cpu"),
            cache_embeddings=False,
            round_top_r=None,
            arms=(runner.AssignmentArm("round_input", "round", "input"),),
        )
