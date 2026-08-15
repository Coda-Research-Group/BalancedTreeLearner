"""Collect every QPS measurement out of job logs and result JSON.

A bare `grep '"qps"'` gives a column of numbers that cannot be compared to each
other. Two facts from this project make that concrete: the same merged index
searched at beam 100 gave 83.47 QPS on an RTX PRO 6000 and 53.56 on an A40
(jobs 22821720 vs 22828626) at identical recall and identical candidate counts,
and a QPS quoted without its recall flatters whichever run was measured at the
lower one. So this reports qps next to recall, candidates, the knobs, and the
card the run landed on, and refuses to sort rows from different hosts together
unless asked.

Usage:
    python experiments/scripts/scan_qps.py <path> [<path> ...]
    python experiments/scripts/scan_qps.py logs/ --min-recall 0.9
    python experiments/scripts/scan_qps.py logs/ --csv > qps.csv

Paths may be files or directories; directories are walked for *.log, *.json,
*.out and *.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SCANNED_SUFFIXES = (".log", ".json", ".out", ".txt")

# "Job 22828626.pbs-m1.metacentrum.cz on luna205.fzu.cz at Tue Aug 11 ..."
JOB_LINE = re.compile(r"^Job\s+(\S+)\s+on\s+(\S+)", re.MULTILINE)
# The nvidia-smi device row; the name is truncated with "..." on long models.
GPU_LINE = re.compile(r"\|\s+\d+\s+(NVIDIA .+?)\s{2,}(?:On|Off)\s*\|")


@dataclass(frozen=True)
class Measurement:
    qps: float
    recall: float | None
    candidates: float | None
    beam: int | None
    knob: int | None
    trees: int | None
    config: str
    host: str
    gpu: str
    source: str

    @property
    def candidates_per_second(self) -> float | None:
        """Throughput independent of where on the curve the run sat.

        Comparing this across two runs of the same index isolates the machine.
        """
        if self.candidates is None:
            return None
        return self.qps * self.candidates


def iter_json_objects(text: str) -> Iterator[dict]:
    """Yield every JSON object embedded in a log, in order.

    Metric blocks are pretty-printed into the log stream between progress
    lines, so the file as a whole is not valid JSON. raw_decode consumes one
    value at a time from an offset, which handles both that and plain .json.
    """
    decoder = json.JSONDecoder()
    index = 0
    while True:
        index = text.find("{", index)
        if index < 0:
            return
        try:
            obj, end = decoder.raw_decode(text, index)
        except ValueError:
            index += 1
            continue
        if isinstance(obj, dict):
            yield obj
        index = end


def iter_rows(obj: dict) -> Iterator[dict]:
    """A metrics.json wraps its points in "rows"; a log emits them bare."""
    if "qps" in obj:
        yield obj
    for row in obj.get("rows", []):
        if isinstance(row, dict) and "qps" in row:
            yield row


def _context(text: str) -> tuple[str, str]:
    job = JOB_LINE.search(text)
    gpu = GPU_LINE.search(text)
    host = job.group(2) if job else ""
    return host, gpu.group(1).strip() if gpu else ""


def scan_file(path: Path) -> list[Measurement]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # unreadable file should not abort a directory walk
        print(f"skipping {path}: {exc}", file=sys.stderr)
        return []

    host, gpu = _context(text)
    out = []
    for obj in iter_json_objects(text):
        for row in iter_rows(obj):
            out.append(
                Measurement(
                    qps=float(row["qps"]),
                    recall=_first_recall(row),
                    candidates=_opt_float(row.get("mean_distcomp")),
                    beam=_opt_int(row.get("beam_size")),
                    knob=_opt_int(row.get("knob_value")),
                    trees=_opt_int(row.get("num_trees")),
                    config=str(row.get("config", row.get("model_id", ""))),
                    host=host,
                    gpu=gpu,
                    source=str(path),
                )
            )
    return out


def _first_recall(row: dict) -> float | None:
    """recall@10 is the convention here, but do not hard-code the k."""
    for key, value in row.items():
        if key.startswith("recall@") and not key.endswith("_count"):
            return _opt_float(value)
    return None


def _opt_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _opt_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def collect(paths: list[Path]) -> list[Measurement]:
    found: list[Measurement] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix in SCANNED_SUFFIXES:
                    found.extend(scan_file(child))
        elif path.is_file():
            found.extend(scan_file(path))
        else:
            print(f"no such path: {path}", file=sys.stderr)
    return found


def _format_table(rows: list[Measurement]) -> str:
    header = (
        f"{'qps':>9} {'recall':>8} {'candidates':>12} {'beam':>5} {'M':>5} {'T':>2}  {'config'}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        recall = f"{row.recall:.5f}" if row.recall is not None else "-"
        cands = f"{row.candidates:,.0f}" if row.candidates is not None else "-"
        lines.append(
            f"{row.qps:>9.2f} {recall:>8} {cands:>12} "
            f"{row.beam if row.beam is not None else '-':>5} "
            f"{row.knob if row.knob is not None else '-':>5} "
            f"{row.trees if row.trees is not None else '-':>2}  {row.config}"
        )
    return "\n".join(lines)


def render(rows: list[Measurement]) -> str:
    """Group by machine, because QPS is only comparable within one."""
    if not rows:
        return "no qps measurements found"

    machines = sorted({(r.host, r.gpu) for r in rows})
    blocks = []
    if len(machines) > 1:
        blocks.append(
            f"NOTE: {len(machines)} different machines below. QPS is not comparable\n"
            "across them — the same Deep100M index measured 83.47 QPS on one card and\n"
            "53.56 on another at identical recall and identical candidate counts.\n"
            "Recall and mean_distcomp are deterministic and do compare.\n"
        )
    for host, gpu in machines:
        subset = [r for r in rows if (r.host, r.gpu) == (host, gpu)]
        label = " / ".join(p for p in (host, gpu) if p) or "unknown machine"
        blocks.append(f"=== {label} — {len(subset)} measurement(s) ===")
        blocks.append(_format_table(subset))
        blocks.append("")
    return "\n".join(blocks).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="log/json files or directories")
    parser.add_argument("--min-recall", type=float, help="drop rows below this recall")
    parser.add_argument("--max-recall", type=float, help="drop rows above this recall")
    parser.add_argument("--config", help="substring match on the config name")
    parser.add_argument(
        "--sort",
        choices=("qps", "recall", "candidates", "source"),
        default="source",
        help="sort key within each machine (default: order found)",
    )
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    args = parser.parse_args(argv)

    rows = collect(args.paths)
    if args.min_recall is not None:
        rows = [r for r in rows if r.recall is not None and r.recall >= args.min_recall]
    if args.max_recall is not None:
        rows = [r for r in rows if r.recall is not None and r.recall <= args.max_recall]
    if args.config:
        rows = [r for r in rows if args.config in r.config]

    if args.sort != "source":
        key = {"qps": lambda r: r.qps, "recall": lambda r: r.recall or 0.0}.get(
            args.sort, lambda r: r.candidates or 0.0
        )
        rows.sort(key=key)

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=[f.name for f in fields(Measurement)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    else:
        print(render(rows))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
