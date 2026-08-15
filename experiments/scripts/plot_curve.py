"""Render Recall@10-vs-distance-computation CSV rows as a simple SVG plot."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot BATL/FAISS curve rows to SVG.")
    parser.add_argument("csv_paths", nargs="+", help="CSV files containing curve rows.")
    parser.add_argument("--output", required=True, help="Output SVG path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = []
    for path in args.csv_paths:
        rows.extend(_read_rows(Path(path)))
    Path(args.output).write_text(render_svg(rows), encoding="utf-8")


def render_svg(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("at least one row is required to render a plot.")
    # Ablation arms share method and model_id and differ only in config, so
    # config joins the label whenever the rows span more than one. Without it
    # an A/B silently renders as a single merged curve.
    configs = {row.get("config", "") for row in rows}
    include_config = len(configs - {""}) > 1

    series = defaultdict(list)
    for row in rows:
        x_value = row.get("mean_distcomp") or row.get("mean_n_distcomp")
        if x_value is None:
            raise ValueError("row must include mean_distcomp or mean_n_distcomp.")
        x = float(x_value)
        y = float(row["recall@10"])
        label = row.get("method", "unknown")
        if row.get("model_id") and row["model_id"] != label:
            label = f"{label}:{row['model_id']}"
        if include_config and row.get("config"):
            label = f"{label}:{row['config']}"
        series[label].append((x, y))

    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    width, height = 900, 520
    left, right, top, bottom = 80, 220, 40, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 20}" text-anchor="middle" '
        'font-family="sans-serif" font-size="16">Mean distance computations</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 '
        f'{top + plot_h / 2})" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">Recall@10</text>',
    ]

    for i, (label, points) in enumerate(sorted(series.items())):
        points = sorted(points)
        color = colors[i % len(colors)]
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}"/>'
        )
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        legend_y = top + i * 24
        parts.append(
            f'<line x1="{width - right + 30}" y1="{legend_y}" x2="{width - right + 55}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{width - right + 62}" y="{legend_y + 5}" font-family="sans-serif" font-size="13">{_escape(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
