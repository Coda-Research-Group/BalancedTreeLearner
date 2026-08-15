from pathlib import Path

SCRIPT_ROOT = Path("experiments/scripts")
STALE_PATTERNS = [
    "run_deep1b_smoke.py",
    "run_sift1m_curve.py",
    "run_glove100_curve.py",
    "--configs",
    "--max-queries",
    "--progress-every",
    "--require-cached-index",
    "--tree-assignment-order",
]


def test_experiment_shell_scripts_do_not_call_deleted_runners_or_old_flags() -> None:
    scripts = sorted(SCRIPT_ROOT.rglob("*.sh"))
    assert scripts

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            assert pattern not in text, f"{script} still contains {pattern!r}"
