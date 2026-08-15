"""Opt-in stage timing and hardware capture for BATL benchmark runs.

Aggregate build and search wall-clock does not say whether time went to beam
decoding, leaf lookup, candidate merging, reranking, or transfers, and QPS rows
from different nodes are not comparable without knowing the GPU and thread
counts they ran on (SPEC_performance C9).

Profiling is off by default and costs nothing when off: ``StageProfiler.stage``
returns a shared no-op context manager and never synchronizes. When on, CUDA
stages are bracketed by ``torch.cuda.synchronize`` so a stage is charged the
device work it actually launched. The spec suggested CUDA events instead; a
synchronize is used because it keeps every stage on one clock, which is what
makes the totals reconcile against wall-clock, and the spec already permits
synchronizing while profiling.
"""

from __future__ import annotations

import os
import platform
import resource
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from time import perf_counter

import torch

_NULL_STAGE = nullcontext()


def peak_host_rss_bytes() -> int:
    """Peak resident set size of this process, normalized across platforms."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


class StageProfiler:
    """Accumulate wall-clock per named stage, with call counts."""

    def __init__(self, enabled: bool, device: torch.device | str = "cpu") -> None:
        self.enabled = enabled
        self.device = torch.device(device)
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._synchronize = enabled and self.device.type == "cuda" and torch.cuda.is_available()

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        if self._synchronize:
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        try:
            yield
        finally:
            if self._synchronize:
                torch.cuda.synchronize(self.device)
            elapsed = perf_counter() - start
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed
            self._calls[name] = self._calls.get(name, 0) + 1

    def stage(self, name: str):
        """Time a region, or do nothing at all when profiling is disabled."""
        if not self.enabled:
            return _NULL_STAGE
        return self._timed(name)

    def reset(self) -> None:
        self._seconds.clear()
        self._calls.clear()

    @property
    def total_s(self) -> float:
        return float(sum(self._seconds.values()))

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        """Stage totals, or an empty mapping when profiling is disabled."""
        if not self.enabled:
            return {}
        return {
            name: {"seconds": round(seconds, 6), "calls": self._calls[name]}
            for name, seconds in sorted(self._seconds.items())
        }


def stage_reconciliation(
    profiler: StageProfiler,
    wall_clock_s: float,
    tolerance: float = 0.10,
) -> dict[str, float | bool]:
    """Compare summed stage time against measured wall-clock.

    Stages never sum to the whole run: setup, logging, artifact writing, and
    the gaps between instrumented regions are all unattributed. This reports
    the residual so a profiled run can be sanity-checked rather than trusted.
    """
    covered = profiler.total_s
    unattributed = wall_clock_s - covered
    fraction = unattributed / wall_clock_s if wall_clock_s > 0 else 0.0
    return {
        "wall_clock_s": round(wall_clock_s, 6),
        "stage_total_s": round(covered, 6),
        "unattributed_s": round(unattributed, 6),
        "unattributed_fraction": round(fraction, 6),
        "within_tolerance": bool(abs(fraction) <= tolerance),
    }


def device_metadata(device: str) -> dict[str, str | int | bool]:
    """Capture what makes two QPS rows comparable, or not."""
    metadata: dict[str, str | int | bool] = {
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "torch_num_threads": torch.get_num_threads(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["cuda"] = getattr(torch.version, "cuda", "") or ""  # type: ignore[attr-defined]
        metadata["gpu_name"] = properties.name
        metadata["gpu_total_memory_bytes"] = int(properties.total_memory)
        metadata["gpu_capability"] = f"{properties.major}.{properties.minor}"
    return metadata


def memory_metadata(device: str) -> dict[str, int]:
    """Peak host and device memory for the run so far."""
    peaks = {"peak_host_rss_bytes": peak_host_rss_bytes()}
    if device == "cuda" and torch.cuda.is_available():
        peaks["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        peaks["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return peaks


def repetition_summary(times_s: list[float]) -> dict[str, float | list[float]]:
    """Median plus every individual timing, so variance stays visible."""
    if not times_s:
        return {"repetitions": 0, "median_s": 0.0, "times_s": []}
    ordered = sorted(times_s)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])
    return {
        "repetitions": len(times_s),
        "median_s": round(median, 6),
        "min_s": round(ordered[0], 6),
        "max_s": round(ordered[-1], 6),
        "times_s": [round(value, 6) for value in times_s],
    }
