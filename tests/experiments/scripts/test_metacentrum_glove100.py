from pathlib import Path

import pytest

SCRIPT_DIR = Path("experiments/scripts/glov100")
HEIGHTS = (2, 3, 4)


@pytest.mark.parametrize("h", HEIGHTS)
def test_glove100_per_height_script(h: int) -> None:
    text = (SCRIPT_DIR / f"metacentrum_glove100_h{h}.sh").read_text(encoding="utf-8")

    assert f"#PBS -N batl_glove100_h{h}" in text
    assert f"H={h}" in text
    assert 'NAME="glove100_h${H}_paper"' in text
    assert "/storage/brno2/home/jozefsprlak/repos/data/glove/glove-100-angular.hdf5" in text
    assert "ngpus" not in text
    assert 'cfg["dataset"]["metric"] = "angular"' in text
    assert 'cfg["training"]["device"] = "cpu"' in text
    assert 'cfg["training"]["batch_size"] = 8192' in text
    assert "build.py" in text
    assert "search.py" in text
    assert "--num-leaves" in text
    assert "160" not in text
