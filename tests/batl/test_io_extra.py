"""Additional tests for batl.io."""

from pathlib import Path

import numpy as np

from batl.utils.io import jsonable, write_rows


def test_jsonable() -> None:
    assert jsonable({"a": 1}) == {"a": 1}
    assert jsonable([1, 2, 3]) == [1, 2, 3]
    assert jsonable((1, 2)) == [1, 2]
    assert jsonable(np.array([1, 2])) == [1, 2]
    assert isinstance(jsonable(np.float32(1.5)), float)
    assert isinstance(jsonable(np.int32(5)), int)


def test_write_rows(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    write_rows(tmp_path, "test", rows)
    assert (tmp_path / "test.json").exists()
    assert (tmp_path / "test.csv").exists()


def test_write_rows_empty(tmp_path: Path) -> None:
    write_rows(tmp_path, "empty", [])
    assert (tmp_path / "empty.json").exists()
    assert not (tmp_path / "empty.csv").exists()
