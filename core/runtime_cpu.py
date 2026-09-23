"""Runtime CPU thread limiter to strictly constrain CPU usage to 1 thread per worker."""
from __future__ import annotations

import os
import sys


def enforce_single_cpu_thread() -> None:
    """Strictly set environment variables and PyTorch/ONNX runtime settings to 1 CPU thread.
    
    Prevents CPU over-saturation (>100% per core) and OpenMP/MKL thread thrashing
    across multi-worker executions on high-core server systems.
    """
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "BLIS_NUM_THREADS",
    ):
        os.environ[env_var] = "1"

    _apply_torch_threads()


def _apply_torch_threads() -> None:
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


# Auto-enforce when module is imported
enforce_single_cpu_thread()
