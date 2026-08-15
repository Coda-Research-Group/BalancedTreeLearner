"""Time a 1-cycle, 1-tree run of each config and report resource usage.

The script patches each config to `max_alternating_cycles=1` and `num_trees=1`,
keeps the full dataset, runs build+search, and reports per-config:

  - wall time (build / search)
  - CPU time (user + sys)
  - peak RSS (process resident memory)
  - peak GPU memory (when device=cuda)
  - projected total wall time, assuming the paper-default 4-tree ensemble and
    a configurable cycle-to-convergence count

Usage:
    python experiments/scripts/estimate_runtime.py \
        experiments/configs/sift1m/sift1m_h3_paper.yaml --device cpu

    python experiments/scripts/estimate_runtime.py \
        experiments/configs/sift1m/sift1m_h2_paper.yaml \
        experiments/configs/sift1m/sift1m_h3_paper.yaml \
        experiments/configs/sift1m/sift1m_h4_paper.yaml \
        --device cuda --cycles 10
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import resource
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_AFFINITY"] = "disabled"

import psutil
import yaml


def _load(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAPER_DEFAULT_NUM_TREES = 4


def _patch_to_one_cycle(src: Path, dst: Path, device: str, queries: int) -> dict:
    cfg = yaml.safe_load(src.read_text())
    original = {
        "num_trees": cfg["model"].get("num_trees", PAPER_DEFAULT_NUM_TREES),
        "subset_size": cfg["dataset"].get("subset_size"),
        "num_queries": cfg["evaluation"].get("num_queries", queries),
    }
    cfg["model"]["num_trees"] = 1
    cfg["training"]["max_alternating_cycles"] = 1
    cfg["training"]["convergence_patience"] = 0
    cfg["training"]["device"] = device
    if device == "cpu":
        cfg["training"]["neighbor_search_backend"] = "faiss_cpu"
        cfg["training"]["tree_update_cache_embeddings"] = False
        cfg["evaluation"]["rerank_backend"] = "numpy_cpu"
    cfg["evaluation"]["num_queries"] = queries
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_probe"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return original


@dataclass
class PhaseStats:
    wall_s: float
    cpu_user_s: float
    cpu_sys_s: float
    peak_rss_mb: float
    peak_gpu_mb: float


class _RssSampler:
    """Background thread sampling process RSS every `interval` seconds."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except psutil.Error:
                break
            self._stop.wait(self.interval)

    def __enter__(self) -> _RssSampler:
        self.peak_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def _measure(fn, device: str) -> PhaseStats:
    torch_cuda = None
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            torch_cuda = torch
            torch_cuda.cuda.reset_peak_memory_stats()
            torch_cuda.cuda.synchronize()

    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    with _RssSampler() as sampler:
        t0 = time.perf_counter()
        fn()
        if torch_cuda:
            torch_cuda.cuda.synchronize()
        wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)

    peak_gpu_mb = 0.0
    if torch_cuda:
        peak_gpu_mb = torch_cuda.cuda.max_memory_allocated() / (1024**2)

    return PhaseStats(
        wall_s=wall,
        cpu_user_s=ru1.ru_utime - ru0.ru_utime,
        cpu_sys_s=ru1.ru_stime - ru0.ru_stime,
        peak_rss_mb=sampler.peak_bytes / (1024**2),
        peak_gpu_mb=peak_gpu_mb,
    )


def _probe(
    config: Path,
    device: str,
    queries: int,
    num_leaves: list[int],
    batch_train: int | None,
) -> tuple[PhaseStats, PhaseStats, dict]:
    workdir = Path(tempfile.mkdtemp(prefix="batl_probe_"))
    try:
        trial = workdir / "trial.yaml"
        original = _patch_to_one_cycle(config, trial, device, queries)
        result_dir = workdir / "results"
        index_path = workdir / "index.pkl"

        build = _load("build.py", f"probe_build_{config.stem}")
        search = _load("search.py", f"probe_search_{config.stem}")

        build_args = [str(trial), "--index-path", str(index_path), "--result-dir", str(result_dir)]
        if batch_train:
            build_args += ["--batch-train", str(batch_train)]

        build_stats = _measure(lambda: build.main(build_args), device)
        search_stats = _measure(
            lambda: search.main(
                [
                    str(trial),
                    "--index-path",
                    str(index_path),
                    "--result-dir",
                    str(result_dir),
                    "--num-leaves",
                    *[str(n) for n in num_leaves],
                    "--n-queries",
                    str(queries),
                    "--batch-search",
                    "100",
                ]
            ),
            device,
        )
        return build_stats, search_stats, original
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", type=Path, nargs="+")
    ap.add_argument("--device", required=True, choices=["cpu", "cuda", "mps"])
    ap.add_argument("--queries", type=int, default=1000, help="probe query count (default 1000)")
    ap.add_argument("--num-leaves", type=int, nargs="+", default=[10, 100])
    ap.add_argument(
        "--cycles", type=int, default=10, help="cycles-to-converge assumed for projection"
    )
    ap.add_argument("--batch-train", type=int, default=None)
    args = ap.parse_args()

    print(
        f"host: cores={psutil.cpu_count(logical=True)}  RAM={psutil.virtual_memory().total / 1024**3:.1f} GB"
    )
    if args.device == "cuda":
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU:  {props.name}  {props.total_memory / 1024**3:.1f} GB")

    rows = []
    for cfg_path in args.configs:
        print(f"\n── probing {cfg_path.name} on {args.device} ─────────────")
        build_stats, search_stats, original = _probe(
            cfg_path, args.device, args.queries, args.num_leaves, args.batch_train
        )
        num_trees = int(original["num_trees"])
        q_scale = (original["num_queries"] or args.queries) / args.queries
        proj_build_h = build_stats.wall_s * num_trees * args.cycles / 3600.0
        proj_search_h = search_stats.wall_s * q_scale / 3600.0
        total_h = proj_build_h + proj_search_h
        peak_rss = max(build_stats.peak_rss_mb, search_stats.peak_rss_mb)
        peak_gpu = max(build_stats.peak_gpu_mb, search_stats.peak_gpu_mb)
        rows.append(
            {
                "config": cfg_path.name,
                "build_wall_s": build_stats.wall_s,
                "build_cpu_s": build_stats.cpu_user_s + build_stats.cpu_sys_s,
                "search_wall_s": search_stats.wall_s,
                "search_cpu_s": search_stats.cpu_user_s + search_stats.cpu_sys_s,
                "peak_rss_mb": peak_rss,
                "peak_gpu_mb": peak_gpu,
                "num_trees": num_trees,
                "proj_total_h": total_h,
                "suggest_walltime_h": max(1, int(total_h * 1.5 + 1)),
            }
        )

    print("\n=== summary ===")
    print(
        f"device={args.device}  assumed cycles-to-converge={args.cycles}  "
        f"probe queries={args.queries}  num_leaves={args.num_leaves}"
    )
    header = (
        f"{'config':38s} {'build_w':>8s} {'build_cpu':>10s} "
        f"{'search_w':>9s} {'search_cpu':>11s} {'RSS':>8s} {'GPU':>8s} "
        f"{'trees':>5s} {'proj h':>7s} {'wall':>5s}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['config']:38s} {r['build_wall_s']:7.1f}s {r['build_cpu_s']:9.1f}s "
            f"{r['search_wall_s']:8.1f}s {r['search_cpu_s']:10.1f}s "
            f"{r['peak_rss_mb']:6.0f}MB {r['peak_gpu_mb']:6.0f}MB "
            f"{r['num_trees']:5d} {r['proj_total_h']:6.2f}h {r['suggest_walltime_h']:4d}h"
        )


if __name__ == "__main__":
    main()
