"""Resource probes and conservative concurrency planning for batch phases."""
from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import threading
import time


@dataclass(frozen=True)
class GpuSnapshot:
    index: str
    total_memory_mib: int
    free_memory_mib: int
    peak_memory_used_mib: int | None = None
    utilization_percent: int | None = None


def choose_workers_per_gpu(
    snapshot: GpuSnapshot,
    *,
    cpu_workers_per_gpu: int,
    reserve_ratio: float,
    max_workers_per_gpu: int,
) -> int:
    """Choose a safe per-GPU file concurrency from one measured worker peak."""
    if not 0.0 <= reserve_ratio < 1.0:
        raise ValueError("reserve_ratio must be in [0, 1)")
    if cpu_workers_per_gpu < 1 or max_workers_per_gpu < 1:
        raise ValueError("worker caps must be positive")
    if snapshot.peak_memory_used_mib is None or snapshot.peak_memory_used_mib <= 0:
        return 1

    usable_memory = int(snapshot.total_memory_mib * (1.0 - reserve_ratio))
    by_memory = max(1, usable_memory // snapshot.peak_memory_used_mib)
    return max(1, min(by_memory, cpu_workers_per_gpu, max_workers_per_gpu))


def cpu_worker_budget(gpu_count: int) -> int:
    """Allocate all logical CPUs across GPU workers on an exclusive host."""
    return max(1, (os.cpu_count() or 1) // max(1, gpu_count))


def probe_gpus() -> list[GpuSnapshot]:
    """Read NVIDIA memory/utilization through nvidia-smi; return empty off-server."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return []

    snapshots = []
    for line in getattr(result, "stdout", "").splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        try:
            snapshots.append(GpuSnapshot(
                index=values[0], total_memory_mib=int(values[1]), free_memory_mib=int(values[2]),
                utilization_percent=int(values[3]),
            ))
        except ValueError:
            continue
    return snapshots


class GpuMemorySampler:
    """Poll GPU memory while calibration tasks are alive and retain each peak."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, GpuSnapshot] = {}
        self._peak_used: dict[str, int] = {}

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, GpuSnapshot]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 4)
        self._sample()
        return {
            index: GpuSnapshot(
                index=snapshot.index,
                total_memory_mib=snapshot.total_memory_mib,
                free_memory_mib=snapshot.free_memory_mib,
                peak_memory_used_mib=self._peak_used.get(index),
                utilization_percent=snapshot.utilization_percent,
            )
            for index, snapshot in self._latest.items()
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        for snapshot in probe_gpus():
            self._latest[snapshot.index] = snapshot
            used = max(0, snapshot.total_memory_mib - snapshot.free_memory_mib)
            self._peak_used[snapshot.index] = max(self._peak_used.get(snapshot.index, 0), used)
