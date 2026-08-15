import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

WRAPPER = Path(
    "experiments/scripts/deep/100m/assignment_control/metacentrum_deep100m_assignment_control.sh"
)
RUNNER_PATH = Path("experiments/scripts/sift128/run_assignment_control.py")


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_assignment_control", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runs_only_the_round_arms() -> None:
    """Sequential needs the full-K branch order — 100M x 256 at the root.

    The ordering question S8 asks does not need it, so the round pair alone
    confirms the result at scale.
    """
    text = WRAPPER.read_text(encoding="utf-8")

    assert "--arms round_confidence,round_input" in text
    assert "sequential" not in text.split("ARMS=(")[1].split(")")[0]


def test_the_reproduction_arm_is_present() -> None:
    """Without round_confidence nothing checks the harness against the source."""
    runner = _load_runner()
    requested = re.search(r"--arms (\S+)", WRAPPER.read_text(encoding="utf-8")).group(1)

    assert "round_confidence" in requested.split(",")
    known = {arm.name for arm in runner.ASSIGNMENT_ARMS}
    assert set(requested.split(",")) <= known


def test_config_carries_the_tree_update_batch_size() -> None:
    """Mirrors the --batch-tree-update the build wrappers pass on the CLI.

    On CUDA this is cosmetic — the attention guard caps any explicit value to
    65535 // num_heads = 8191, which is what `auto` resolves to as well, and
    job 22750115 logged exactly that capping. The knob only changes behaviour
    on CPU. It is asserted so the control's recorded config keeps matching the
    build that produced its source index.
    """
    text = WRAPPER.read_text(encoding="utf-8")

    assert "TREE_UPDATE_BATCH_SIZE=32768" in text
    writer = Path(
        "experiments/scripts/deep/100m/label_ablation/deep100m_label_config.sh"
    ).read_text(encoding="utf-8")
    assert "tree_update_batch_size: ${TREE_UPDATE_BATCH_SIZE:-auto}" in writer


def test_never_rebuilds_and_refuses_without_a_source_index() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "build.py" not in text
    assert ': "${SOURCE_INDEX:?' in text
    assert "Source index not found" in text
    assert "exit 2" in text


def test_results_are_copied_out_before_each_search() -> None:
    """A search failure must not cost the reassignment, which is the slow part."""
    text = WRAPPER.read_text(encoding="utf-8")

    reassign = text.index("run_assignment_control.py")
    first_copy = text.index('cp -r "$CONTROL_DIR/." "$STORAGE_DIR/"')
    first_search = text.index("search.py")
    assert reassign < first_copy < first_search


def test_stays_on_the_small_gpu_queue() -> None:
    """Reassignment decodes in batches and rerank is numpy_cpu, so no big card."""
    text = WRAPPER.read_text(encoding="utf-8")

    assert "gpu_mem=16gb" in text
