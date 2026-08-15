import pickle

import pytest
import torch

from batl.model import BATLModel
from batl.tree import BATLTree
from batl.utils.config import ModelConfig
from batl.utils.index_parsing import load_index, merge_indexes, save_index
from merge_index import main as merge_index_main


def _model_config(**overrides) -> ModelConfig:
    values = {
        "branching_factor": 2,
        "tree_height": 2,
        "embedding_dim": 3,
        "encoder_hidden": 4,
        "embed_dim": 4,
        "num_heads": 2,
        "ff_dim": 8,
        "dropout": 0.0,
        "num_trees": 1,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _save_single_tree_index(
    path, *, model_cfg: ModelConfig | None = None, tree: BATLTree | None = None
) -> None:
    cfg = model_cfg or _model_config()
    save_index(
        models=[BATLModel(cfg)],
        trees=[
            tree
            or BATLTree.random_init(
                N=4, K=cfg.branching_factor, H=cfg.tree_height, alpha=1.0, seed=0
            )
        ],
        path=str(path),
    )


def test_save_and_load_index_round_trip(tmp_path) -> None:
    tree = BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0)
    model_cfg = _model_config()
    model = BATLModel(model_cfg)
    path = tmp_path / "index.pkl"

    save_index(models=[model], trees=[tree], path=str(path))
    models, trees = load_index(str(path))

    assert len(models) == 1
    assert isinstance(models[0], BATLModel)
    assert all(param.device.type == "cpu" for param in models[0].parameters())
    assert models[0].K == model.K
    assert set(models[0].state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(models[0].state_dict()[name], tensor)
    assert len(trees) == 1
    assert trees[0].K == tree.K
    assert trees[0].paths.tolist() == tree.paths.tolist()


def test_save_index_rejects_non_batl_models(tmp_path) -> None:
    tree = BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0)

    with pytest.raises(TypeError, match="BATLModel"):
        save_index(models=[{"name": "not-a-model"}], trees=[tree], path=str(tmp_path / "bad.pkl"))


def test_save_index_rejects_model_tree_count_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="same length"):
        save_index(
            models=[],
            trees=[BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0)],
            path=str(tmp_path / "bad.pkl"),
        )


def test_load_index_keeps_legacy_pickle_checkpoints_readable(tmp_path) -> None:
    tree = BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0)
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as f:
        pickle.dump({"models": [{"legacy": True}], "trees": [tree]}, f)

    models, trees = load_index(str(path))

    assert models == [{"legacy": True}]
    assert len(trees) == 1
    assert trees[0].paths.tolist() == tree.paths.tolist()


def test_merge_indexes_round_trips_single_tree_indexes(tmp_path) -> None:
    first = tmp_path / "tree0.pkl"
    second = tmp_path / "tree1.pkl"
    output = tmp_path / "merged.pkl"
    _save_single_tree_index(first, tree=BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0))
    _save_single_tree_index(second, tree=BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=1))

    n_models, n_trees = merge_indexes([first, second], output)

    assert (n_models, n_trees) == (2, 2)
    models, trees = load_index(str(output))
    assert len(models) == 2
    assert len(trees) == 2
    assert [tree.paths.tolist() for tree in trees] == [
        BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0).paths.tolist(),
        BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=1).paths.tolist(),
    ]


def test_merge_indexes_rejects_tree_shape_mismatch(tmp_path) -> None:
    first = tmp_path / "tree0.pkl"
    second = tmp_path / "tree1.pkl"
    _save_single_tree_index(first)
    _save_single_tree_index(
        second,
        tree=BATLTree.random_init(N=4, K=3, H=2, alpha=1.0, seed=1),
    )

    with pytest.raises(ValueError, match="tree compatibility"):
        merge_indexes([first, second], tmp_path / "merged.pkl")


def test_merge_indexes_rejects_model_config_mismatch(tmp_path) -> None:
    first = tmp_path / "tree0.pkl"
    second = tmp_path / "tree1.pkl"
    _save_single_tree_index(first)
    _save_single_tree_index(second, model_cfg=_model_config(embed_dim=8, ff_dim=16))

    with pytest.raises(ValueError, match="ModelConfig"):
        merge_indexes([first, second], tmp_path / "merged.pkl")


def test_merge_index_cli_writes_output(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    first = tmp_path / "tree0.pkl"
    second = tmp_path / "tree1.pkl"
    output = tmp_path / "merged.pkl"
    _save_single_tree_index(first, tree=BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=0))
    _save_single_tree_index(second, tree=BATLTree.random_init(N=4, K=2, H=2, alpha=1.0, seed=1))

    merge_index_main(["--output", str(output), str(first), str(second)])

    assert output.exists()
    assert "merged 2 model(s) / 2 tree(s)" in capsys.readouterr().out
