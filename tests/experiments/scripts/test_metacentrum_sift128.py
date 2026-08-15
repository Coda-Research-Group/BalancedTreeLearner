from pathlib import Path

import pytest

SCRIPT_DIR = Path("experiments/scripts/sift128")
HEIGHTS = (2, 3, 4)


@pytest.mark.parametrize("h", HEIGHTS)
def test_sift128_cpu_per_height_script(h: int) -> None:
    text = (SCRIPT_DIR / f"metacentrum_sift128_cpu_h{h}.sh").read_text(encoding="utf-8")

    assert f"#PBS -N batl_sift128_cpu_h{h}" in text
    assert f"H={h}" in text
    assert 'NAME="sift1m_h${H}_paper"' in text
    assert "/storage/brno2/home/jozefsprlak/repos/data/sift/sift-128-euclidean.hdf5" in text
    assert "ngpus" not in text
    assert "nvidia-smi" not in text
    assert 'cfg["training"]["device"] = "cpu"' in text
    assert 'cfg["training"]["batch_size"] = 8192' in text
    assert "build.py" in text
    assert "search.py" in text
    assert "--num-leaves" in text
    assert "160" not in text
