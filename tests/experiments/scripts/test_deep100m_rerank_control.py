import re
from pathlib import Path

WRAPPER = Path(
    "experiments/scripts/deep/100m/rerank_backend_control/"
    "metacentrum_deep100m_rerank_backend_control.sh"
)


def test_runs_every_backend_and_profile_combination() -> None:
    """Attribution and headline throughput cannot come from the same run.

    Profiled runs synchronize CUDA at stage boundaries and are deliberately
    pessimistic, so each backend needs both a profiled and a plain run.
    """
    calls = re.findall(r"^run_point (\S+) (\S+)$", WRAPPER.read_text(encoding="utf-8"), re.M)

    assert set(calls) == {
        ("numpy_cpu", "false"),
        ("numpy_cpu", "true"),
        ("torch_gpu_resident", "false"),
        ("torch_gpu_resident", "true"),
    }


def test_cpu_runs_precede_gpu_runs() -> None:
    """The resident upload reads the whole database and warms the page cache.

    A numpy_cpu run following it would be measuring the GPU run's I/O, not its
    own, and would understate the gap the control exists to measure.
    """
    backends = re.findall(r"^run_point (\S+) \S+$", WRAPPER.read_text(encoding="utf-8"), re.M)

    assert backends.index("torch_gpu_resident") > max(
        i for i, b in enumerate(backends) if b == "numpy_cpu"
    ), "every numpy_cpu run must come before the first torch_gpu_resident run"


def test_page_cache_is_warmed_once_before_any_timing() -> None:
    """Both backends must start from the same cache state."""
    text = WRAPPER.read_text(encoding="utf-8")

    warm_at = text.index("Warming the page cache")
    first_run = text.index("run_point numpy_cpu false")
    assert warm_at < first_run
    assert "cat data/deep100m/base.fbin > /dev/null" in text


def test_card_is_large_enough_for_the_resident_database() -> None:
    """36.14 GiB database plus the 2 GiB headroom check_device_capacity adds."""
    text = WRAPPER.read_text(encoding="utf-8")
    requested_gb = int(re.search(r"gpu_mem=(\d+)gb", text).group(1))
    required_gib = (100_000_000 * 96 * 4 + 100_000_000 * 4) / 1024**3 + 2.0

    assert requested_gb * 1000**3 / 1024**3 >= required_gib


def test_refuses_without_the_source_index() -> None:
    """A four-run sweep should not burn an allocation discovering a bad path."""
    text = WRAPPER.read_text(encoding="utf-8")

    assert "Missing index from the build job" in text
    assert "exit 2" in text
    assert "build.py" not in text, "this control must never rebuild"
