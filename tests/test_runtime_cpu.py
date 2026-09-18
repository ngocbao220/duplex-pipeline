import os
import pytest
from core.runtime_cpu import enforce_single_cpu_thread


def test_enforce_single_cpu_thread_env_vars():
    enforce_single_cpu_thread()
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        assert os.environ.get(env_var) == "1", f"{env_var} is not set to 1"


def test_torch_single_thread():
    enforce_single_cpu_thread()
    try:
        import torch
        assert torch.get_num_threads() == 1
    except ImportError:
        pytest.skip("PyTorch not installed")
