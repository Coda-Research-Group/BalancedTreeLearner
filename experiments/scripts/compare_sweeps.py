"""Compare two BATL search sweeps as recall-per-bucket curves.

Built for the Deep10M label-quality A/B (SPEC_performance C4), but takes any
two `search_rows.csv` files from runs that differ in one variable.

The headline number is not recall at a fixed number of leaves. Because the
tree is balanced by construction, both arms scan roughly the same candidates
at the same M, so the question is how far *left* the curve moved: at matched
recall, how many candidates does each arm need? That ratio is what translates
into a QPS ratio, and it is the comparison the paper's Table 1 implies.

Usage:
    python experiments/scripts/compare_sweeps.py \
        baseline/search_rows.csv treatment/search_rows.csv
    python experiments/scripts/compare_sweeps.py a.csv b.csv --labels 1pct exact
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

# Arms with the same balance factor should scan the same candidates at the same
# M. A wider gap than this means something other than the intended variable
# moved, and the comparison is confounded.
BALANCE_MISMATCH_TOLERANCE = 0.05


@dataclass(frozen=True)
class Point:
    leaves: int
    recall: float
    candidates: float
    qps: float


def read_curve(path: Path) -> list[Point]:
    """Read a search_rows.csv into an M-ordered curve."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows.")

    points = [
        Point(
            leaves=int(float(row["knob_value"])),
            recall=float(row["recall@10"]),
            candidates=float(row.get("mean_distcomp") or row["mean_n_distcomp"]),
            qps=float(row.get("qps", "nan")),
        )
        for row in rows
    ]
    return sorted(points, key=lambda point: point.leaves)


def candidates_at_recall(curve: list[Point], target: float) -> float | None:
    """Candidates needed to reach ``target`` recall, linearly interpolated.

    Returns None when the target lies outside the measured range — extrapolating
    a recall curve past its endpoints is not meaningful.
    """
    ordered = sorted(curve, key=lambda point: point.recall)
    if not ordered or target < ordered[0].recall or target > ordered[-1].recall:
        return None
    for low, high in itertools.pairwise(ordered):
        if low.recall <= target <= high.recall:
            span = high.recall - low.recall
            if span <= 0:
                return low.candidates
            weight = (target - low.recall) / span
            return low.candidates + weight * (high.candidates - low.candidates)
    return ordered[-1].candidates


def balance_warnings(baseline: list[Point], treatment: list[Point]) -> list[str]:
    """Flag M values where the two arms scanned materially different candidates."""
    by_leaves = {point.leaves: point for point in treatment}
    warnings = []
    for point in baseline:
        other = by_leaves.get(point.leaves)
        if other is None or point.candidates <= 0:
            continue
        drift = abs(other.candidates - point.candidates) / point.candidates
        if drift > BALANCE_MISMATCH_TOLERANCE:
            warnings.append(
                f"  M={point.leaves}: candidates differ by {drift * 100:.1f}% "
                f"({point.candidates:.0f} vs {other.candidates:.0f})"
            )
    return warnings


def recall_targets(baseline: list[Point], treatment: list[Point]) -> list[float]:
    """Recall levels both curves actually cover."""
    low = max(min(p.recall for p in baseline), min(p.recall for p in treatment))
    high = min(max(p.recall for p in baseline), max(p.recall for p in treatment))
    if high <= low:
        return []
    steps = 5
    return [low + (high - low) * i / (steps - 1) for i in range(steps)]


def render(
    baseline: list[Point],
    treatment: list[Point],
    labels: tuple[str, str],
) -> str:
    lines: list[str] = []
    left, right = labels

    lines.append(f"Curve by leaves   ({left} -> {right})")
    lines.append(f"{'M':>6} {'cands':>10} {'recall A':>10} {'recall B':>10} {'delta':>9}")
    by_leaves = {point.leaves: point for point in treatment}
    for point in baseline:
        other = by_leaves.get(point.leaves)
        if other is None:
            continue
        lines.append(
            f"{point.leaves:>6} {point.candidates:>10.0f} "
            f"{point.recall:>10.4f} {other.recall:>10.4f} {other.recall - point.recall:>+9.4f}"
        )

    warnings = balance_warnings(baseline, treatment)
    if warnings:
        lines.append("")
        lines.append("WARNING: arms did not scan comparable candidate counts:")
        lines.extend(warnings)
        lines.append("  The curve-shift numbers below are confounded.")

    lines.append("")
    lines.append("Candidates needed at matched recall (the number that maps to QPS)")
    targets = recall_targets(baseline, treatment)
    if not targets:
        lines.append("  curves do not overlap in recall; nothing to compare")
        return "\n".join(lines)

    lines.append(f"{'recall':>8} {'cands A':>12} {'cands B':>12} {'B/A':>8} {'speedup':>9}")
    for target in targets:
        first = candidates_at_recall(baseline, target)
        second = candidates_at_recall(treatment, target)
        if first is None or second is None or first <= 0:
            continue
        ratio = second / first
        lines.append(
            f"{target:>8.4f} {first:>12.0f} {second:>12.0f} {ratio:>8.3f} {1 / ratio:>8.2f}x"
        )

    lines.append("")
    lines.append(
        "'speedup' is the candidate reduction at equal recall, i.e. roughly the "
        "QPS factor once rerank dominates. Values near 1.00 mean the change did "
        "not move recall-per-bucket."
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two BATL search sweeps.")
    parser.add_argument("baseline_csv", help="search_rows.csv for the baseline arm.")
    parser.add_argument("treatment_csv", help="search_rows.csv for the treatment arm.")
    parser.add_argument(
        "--labels",
        nargs=2,
        default=("A", "B"),
        metavar=("BASELINE", "TREATMENT"),
        help="Names for the two arms in the report.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    baseline = read_curve(Path(args.baseline_csv))
    treatment = read_curve(Path(args.treatment_csv))
    print(render(baseline, treatment, (args.labels[0], args.labels[1])))


if __name__ == "__main__":
    main()
