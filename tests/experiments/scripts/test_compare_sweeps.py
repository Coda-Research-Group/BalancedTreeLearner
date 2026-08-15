import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves sys.modules[cls.__module__]; register before exec.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _load("experiments/scripts/compare_sweeps.py", "batl_compare_sweeps")

HEADER = "method,model_id,config,knob_value,recall@10,mean_distcomp,qps\n"


def _write(path: Path, config: str, rows: list[tuple[int, float, float]]) -> Path:
    body = "".join(
        f"batl,K256_H2,{config},{leaves},{recall},{cands},10.0\n" for leaves, recall, cands in rows
    )
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_read_curve_sorts_by_leaves(tmp_path: Path) -> None:
    path = _write(tmp_path / "a.csv", "arm", [(80, 0.88, 96000.0), (10, 0.50, 6800.0)])

    curve = compare.read_curve(path)

    assert [point.leaves for point in curve] == [10, 80]
    assert curve[0].recall == 0.50


def test_candidates_at_recall_interpolates_and_refuses_extrapolation() -> None:
    curve = [
        compare.Point(leaves=10, recall=0.50, candidates=1000.0, qps=1.0),
        compare.Point(leaves=20, recall=0.70, candidates=2000.0, qps=1.0),
    ]

    assert compare.candidates_at_recall(curve, 0.60) == pytest.approx(1500.0)
    assert compare.candidates_at_recall(curve, 0.50) == pytest.approx(1000.0)
    # Outside the measured range there is no honest answer.
    assert compare.candidates_at_recall(curve, 0.90) is None
    assert compare.candidates_at_recall(curve, 0.10) is None


def test_balance_warning_fires_when_arms_scan_different_candidate_counts() -> None:
    baseline = [compare.Point(leaves=10, recall=0.5, candidates=1000.0, qps=1.0)]
    same = [compare.Point(leaves=10, recall=0.6, candidates=1020.0, qps=1.0)]
    different = [compare.Point(leaves=10, recall=0.6, candidates=1500.0, qps=1.0)]

    assert compare.balance_warnings(baseline, same) == []
    assert len(compare.balance_warnings(baseline, different)) == 1


def test_report_quantifies_a_leftward_curve_shift(tmp_path: Path) -> None:
    """A treatment reaching the same recall on half the candidates reads as 2x."""
    baseline = _write(
        tmp_path / "base.csv",
        "subset_1pct",
        [(10, 0.50, 1000.0), (40, 0.70, 4000.0), (80, 0.90, 8000.0)],
    )
    treatment = _write(
        tmp_path / "treat.csv",
        "exact",
        [(10, 0.70, 1000.0), (40, 0.90, 4000.0), (80, 0.98, 8000.0)],
    )

    report = compare.render(
        compare.read_curve(baseline),
        compare.read_curve(treatment),
        ("1pct", "exact"),
    )

    assert "Candidates needed at matched recall" in report
    assert "WARNING" not in report
    # At recall 0.90 the baseline needs 8000 candidates, the treatment 4000.
    assert "2.00x" in report


def test_report_flags_confounded_comparison(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "b.csv", "a", [(10, 0.50, 1000.0), (40, 0.80, 4000.0)])
    treatment = _write(tmp_path / "t.csv", "b", [(10, 0.55, 3000.0), (40, 0.85, 9000.0)])

    report = compare.render(compare.read_curve(baseline), compare.read_curve(treatment), ("a", "b"))

    assert "WARNING" in report
    assert "confounded" in report


def test_report_handles_non_overlapping_curves(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "b.csv", "a", [(10, 0.10, 100.0), (40, 0.20, 400.0)])
    treatment = _write(tmp_path / "t.csv", "b", [(10, 0.80, 100.0), (40, 0.90, 400.0)])

    report = compare.render(compare.read_curve(baseline), compare.read_curve(treatment), ("a", "b"))

    assert "do not overlap" in report


def test_plot_curve_keeps_ablation_arms_as_separate_series() -> None:
    plot = _load("experiments/scripts/plot_curve.py", "batl_plot_curve")
    rows = [
        {
            "method": "batl",
            "model_id": "K256_H2",
            "config": "subset_1pct",
            "mean_distcomp": "1000",
            "recall@10": "0.5",
        },
        {
            "method": "batl",
            "model_id": "K256_H2",
            "config": "exact",
            "mean_distcomp": "1000",
            "recall@10": "0.7",
        },
    ]

    svg = plot.render_svg(rows)

    # Both arms share method and model_id; without config they would merge.
    assert "subset_1pct" in svg
    assert "exact" in svg
