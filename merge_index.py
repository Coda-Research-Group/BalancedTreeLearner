"""Merge compatible BATL per-tree indexes into one ensemble index."""

from __future__ import annotations

import argparse
from pathlib import Path

from batl.utils.index_parsing import merge_indexes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge BATL index files.")
    parser.add_argument("inputs", type=Path, nargs="+", help="Input index files to merge.")
    parser.add_argument("--output", type=Path, required=True, help="Merged output index path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    n_models, n_trees = merge_indexes(args.inputs, args.output)
    print(f"merged {n_models} model(s) / {n_trees} tree(s) into {args.output}")


if __name__ == "__main__":
    main()
