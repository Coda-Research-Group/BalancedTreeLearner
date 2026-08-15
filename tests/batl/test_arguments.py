"""Tests for batl.arguments."""

import argparse
from pathlib import Path

import pytest

from batl.utils.arguments import (
    add_batch_search_arg,
    add_batch_train_arg,
    add_batch_tree_update_arg,
    add_config_arg,
    add_datapath_arg,
    add_index_path_arg,
    add_log_arg,
    add_n_queries_arg,
    add_num_leaves_arg,
    add_result_dir_arg,
    apply_args_to_config,
    non_negative_int,
    positive_int,
)
from batl.utils.config import ExperimentConfig, ModelConfig, TrainConfig


def test_positive_int() -> None:
    assert positive_int("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-1")
    with pytest.raises(ValueError):
        positive_int("abc")


def test_non_negative_int() -> None:
    assert non_negative_int("5") == 5
    assert non_negative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")
    with pytest.raises(ValueError):
        non_negative_int("abc")


def test_add_args() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    add_datapath_arg(parser)
    add_log_arg(parser)
    add_result_dir_arg(parser)
    add_index_path_arg(parser)
    add_batch_train_arg(parser)
    add_batch_tree_update_arg(parser)
    add_num_leaves_arg(parser)
    add_n_queries_arg(parser)
    add_batch_search_arg(parser)

    args = parser.parse_args(
        [
            "myconfig.yaml",
            "--datapath",
            "mydata",
            "--log",
            "--result-dir",
            "myresults",
            "--index-path",
            "myindex.pkl",
            "--batch-train",
            "128",
            "--batch-tree-update",
            "256",
            "--num-leaves",
            "10",
            "20",
            "--n-queries",
            "1000",
            "--batch-search",
            "50",
        ]
    )
    assert args.config == "myconfig.yaml"
    assert args.datapath == "mydata"
    assert args.log is True
    assert args.result_dir == "myresults"
    assert args.index_path == "myindex.pkl"
    assert args.batch_train == 128
    assert args.batch_tree_update == 256
    assert args.num_leaves == [10, 20]
    assert args.n_queries == 1000
    assert args.batch_search == 50


def test_apply_args_to_config() -> None:
    cfg = ExperimentConfig(
        name="test",
        seed=0,
        output_dir="out",
        dataset_name="ds",
        dataset_path="path",
        split="test",
        subset_size=None,
        recall_at=[10],
        num_queries=10,
        model=ModelConfig(),
        train=TrainConfig(batch_size=10, tree_update_batch_size=None),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--batch-train", type=int)
    parser.add_argument("--batch-tree-update", type=int)
    args = parser.parse_args(
        ["--datapath", "newpath", "--batch-train", "20", "--batch-tree-update", "30"]
    )
    apply_args_to_config(cfg, args)
    assert cfg.dataset_path == "newpath"
    assert cfg.train.batch_size == 20
    assert cfg.train.tree_update_batch_size == 30
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=int)
    parser.add_argument("--bar", type=Path)
