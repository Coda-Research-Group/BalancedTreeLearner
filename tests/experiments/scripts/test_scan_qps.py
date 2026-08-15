import json
from pathlib import Path

import pytest

from experiments.scripts import scan_qps

LOG = """Job 22828626.pbs-m1.metacentrum.cz on luna205.fzu.cz at Tue Aug 11 23:36:32 CEST 2026
Copying Deep100M data...
+-----------------------------------------------------------------------------------------+
|   0  NVIDIA A40                     On  |   00000000:61:00.0 Off |                    0 |
+-----------------------------------------------------------------------------------------+
K256_H2 b=100 M=100: queries done 81/10000, remaining 9919, elapsed 2.5s, 32.63 q/s
{
  "method": "batl",
  "config": "deep100m_t4_beam_100",
  "beam_size": 100,
  "num_trees": 4,
  "knob_value": 100,
  "recall@10": 0.89697,
  "recall@10_zero_count": 8,
  "mean_distcomp": 123079.33,
  "qps": 53.55558702005799
}
=== beam_size=200, num_leaves=200 ===
{
  "config": "deep100m_t4_beam_200",
  "beam_size": 200,
  "num_trees": 4,
  "knob_value": 200,
  "recall@10": 0.95409,
  "mean_distcomp": 267551.7195,
  "qps": 23.63443496333855
}
Done.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_extracts_every_metric_block_from_one_log(tmp_path: Path) -> None:
    rows = scan_qps.scan_file(_write(tmp_path, "job.log", LOG))

    assert [r.qps for r in rows] == pytest.approx([53.55558702005799, 23.63443496333855])
    assert [r.recall for r in rows] == pytest.approx([0.89697, 0.95409])
    assert [r.knob for r in rows] == [100, 200]


def test_captures_the_machine_because_qps_is_not_portable(tmp_path: Path) -> None:
    """83.47 vs 53.56 QPS on one index came down to the card alone."""
    rows = scan_qps.scan_file(_write(tmp_path, "job.log", LOG))

    assert all(r.host == "luna205.fzu.cz" for r in rows)
    assert all(r.gpu == "NVIDIA A40" for r in rows)


def test_progress_lines_are_not_mistaken_for_measurements(tmp_path: Path) -> None:
    """The log prints a running "q/s" per batch; only the JSON blocks count."""
    rows = scan_qps.scan_file(_write(tmp_path, "job.log", LOG))

    assert len(rows) == 2
    assert 32.63 not in [r.qps for r in rows]


def test_reads_metrics_json_rows_as_well_as_logs(tmp_path: Path) -> None:
    payload = {"rows": [{"qps": 47.5, "recall@10": 0.0, "knob_value": 100}]}
    rows = scan_qps.scan_file(_write(tmp_path, "metrics.json", json.dumps(payload)))

    assert [r.qps for r in rows] == pytest.approx([47.5])
    assert rows[0].recall == 0.0


def test_a_truncated_recall_key_other_than_at_10_still_resolves(tmp_path: Path) -> None:
    text = json.dumps({"qps": 1.0, "recall@100": 0.5, "mean_distcomp": 10})
    rows = scan_qps.scan_file(_write(tmp_path, "r.json", text))

    assert rows[0].recall == 0.5


def test_zero_count_fields_are_not_read_as_the_recall(tmp_path: Path) -> None:
    text = json.dumps({"qps": 1.0, "recall@10_zero_count": 493, "recall@10": 0.478})
    rows = scan_qps.scan_file(_write(tmp_path, "r.json", text))

    assert rows[0].recall == 0.478


def test_truncated_or_malformed_json_does_not_abort_the_scan(tmp_path: Path) -> None:
    """A walltime kill leaves a half-written block; later files must still scan."""
    text = '{"qps": 1.0, "recall@10": 0.5}\n{"qps": 2.0, "recall@10":\n'
    rows = scan_qps.scan_file(_write(tmp_path, "cut.log", text))

    assert [r.qps for r in rows] == pytest.approx([1.0])


def test_directories_are_walked_and_non_logs_ignored(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub", "a.log", LOG)
    _write(tmp_path, "notes.md", '{"qps": 999}')

    rows = scan_qps.collect([tmp_path])

    assert len(rows) == 2
    assert 999 not in [r.qps for r in rows]


def test_output_groups_by_machine_and_warns_when_mixing(tmp_path: Path) -> None:
    other = LOG.replace("luna205.fzu.cz", "grogu1.cerit-sc.cz").replace(
        "NVIDIA A40                    ", "NVIDIA RTX PRO 6000 Blac...   "
    )
    rows = scan_qps.scan_file(_write(tmp_path, "a.log", LOG))
    rows += scan_qps.scan_file(_write(tmp_path, "b.log", other))

    rendered = scan_qps.render(rows)

    assert "luna205.fzu.cz / NVIDIA A40" in rendered
    assert "grogu1.cerit-sc.cz" in rendered
    assert "not comparable" in rendered


def test_single_machine_output_carries_no_warning(tmp_path: Path) -> None:
    rendered = scan_qps.render(scan_qps.scan_file(_write(tmp_path, "a.log", LOG)))

    assert "not comparable" not in rendered


def test_candidates_per_second_isolates_the_machine(tmp_path: Path) -> None:
    rows = scan_qps.scan_file(_write(tmp_path, "job.log", LOG))

    assert rows[0].candidates_per_second == pytest.approx(123079.33 * 53.55558702005799)


def test_recall_filter_and_csv_round_trip(tmp_path: Path, capsys) -> None:
    log = _write(tmp_path, "job.log", LOG)

    assert scan_qps.main([str(log), "--min-recall", "0.9", "--csv"]) == 0
    out = capsys.readouterr().out

    assert "0.95409" in out
    assert "0.89697" not in out


def test_exit_code_is_nonzero_when_nothing_matched(tmp_path: Path) -> None:
    _write(tmp_path, "empty.log", "no metrics here\n")

    assert scan_qps.main([str(tmp_path)]) == 1
