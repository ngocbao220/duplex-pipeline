"""Runtime compatibility helpers for PyTorch AMP and setuptools/pkg_resources."""

from __future__ import annotations

import sys
import types


def ensure_runtime_compat() -> None:
    """Ensure pkg_resources and torch.amp custom_fwd/bwd decorators work seamlessly across PyTorch versions."""
    import os
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # Some server launchers export empty cache paths.  Librosa/Numba later
    # turn those into ``os.makedirs(\"\")`` before benchmark code can sanitize
    # them, producing an unhelpful FileNotFoundError for DNSMOS/NISQA.
    for cache_var in ("NUMBA_CACHE_DIR", "LIBROSA_CACHE_DIR", "XDG_CACHE_HOME"):
        if os.environ.get(cache_var) == "":
            os.environ.pop(cache_var)
    # 1. pkg_resources fallback for Python 3.12+ environments missing setuptools
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        try:
            pr = types.ModuleType("pkg_resources")
            pr.resource_filename = lambda *args: ""
            pr.get_distribution = lambda *args: types.SimpleNamespace(version="0.0.0")
            pr.Requirement = lambda *args: None
            sys.modules["pkg_resources"] = pr
        except Exception:
            pass

    # 2. torch.amp custom_fwd / custom_bwd wrapper handling device_type kwarg variations & thread limiting
    try:
        import torch

        try:
            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
        except Exception:
            pass

        if hasattr(torch, "amp"):
            orig_fwd = getattr(torch.amp, "custom_fwd", None)
            if orig_fwd is None and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "custom_fwd"):
                orig_fwd = torch.cuda.amp.custom_fwd

            if orig_fwd is not None:
                def _custom_fwd_wrapper(*args, **kwargs):
                    try:
                        return orig_fwd(*args, **kwargs)
                    except TypeError as err:
                        err_msg = str(err)
                        if "unexpected keyword argument" in err_msg and "device_type" in err_msg:
                            kwargs.pop("device_type", None)
                            return orig_fwd(*args, **kwargs)
                        if "missing 1 required keyword-only argument" in err_msg and "device_type" in err_msg:
                            kwargs["device_type"] = "cuda"
                            return orig_fwd(*args, **kwargs)
                        raise

                torch.amp.custom_fwd = _custom_fwd_wrapper
            else:
                def _dummy_fwd(*args, **kwargs):
                    if len(args) == 1 and callable(args[0]):
                        return args[0]
                    return lambda fn: fn
                torch.amp.custom_fwd = _dummy_fwd

            orig_bwd = getattr(torch.amp, "custom_bwd", None)
            if orig_bwd is None and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "custom_bwd"):
                orig_bwd = torch.cuda.amp.custom_bwd

            if orig_bwd is not None:
                def _custom_bwd_wrapper(*args, **kwargs):
                    try:
                        return orig_bwd(*args, **kwargs)
                    except TypeError as err:
                        err_msg = str(err)
                        if "unexpected keyword argument" in err_msg and "device_type" in err_msg:
                            kwargs.pop("device_type", None)
                            return orig_bwd(*args, **kwargs)
                        if "missing 1 required keyword-only argument" in err_msg and "device_type" in err_msg:
                            kwargs["device_type"] = "cuda"
                            return orig_bwd(*args, **kwargs)
                        raise

                torch.amp.custom_bwd = _custom_bwd_wrapper
            else:
                def _dummy_bwd(*args, **kwargs):
                    if len(args) == 1 and callable(args[0]):
                        return args[0]
                    return lambda fn: fn
                torch.amp.custom_bwd = _dummy_bwd
    except Exception:
        pass
