"""Strict local-model readiness checks for ``benchmark_stereo``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.compat import ensure_runtime_compat
from .dnsmos import P808_MODEL, PRIMARY_MODEL, DNSMOSScorer, resolve_dnsmos_dir


REQUIRED_BENCHMARK_MODELS = ("dnsmos", "speechbrain_ecapa", "squim", "nisqa")


def check_benchmark_models(dnsmos_model_dir: Path | str = Path("models/dnsmos"), device: str = "cpu") -> dict:
    """Initialize and smoke-test every local benchmark metric without network access."""
    ensure_runtime_compat()
    selected_device = _resolve_device(device)
    return {
        "dnsmos": _check_dnsmos(Path(dnsmos_model_dir)),
        "speechbrain_ecapa": _check_speechbrain(selected_device),
        "squim": _check_squim(selected_device),
        "nisqa": _check_nisqa(selected_device),
    }


def benchmark_models_ready(results: dict) -> bool:
    """Return whether all metrics required by strict benchmark mode are usable."""
    return all(results.get(name, {}).get("status") == "OK" for name in REQUIRED_BENCHMARK_MODELS)


def _check_dnsmos(requested_dir: Path) -> dict:
    resolved_dir = resolve_dnsmos_dir(requested_dir)
    missing = [name for name in (PRIMARY_MODEL, P808_MODEL) if not (resolved_dir / name).is_file()]
    if missing:
        return _failure(
            "MISSING",
            f"Resolved local directory: {resolved_dir}; missing: {', '.join(missing)}. "
            "Copy both DNSMOS ONNX files locally; no download fallback is used.",
        )
    try:
        result = DNSMOSScorer(resolved_dir).score(_smoke_audio(), 16000)
        if result.get("status") != "ok":
            return _failure("ERROR", f"Resolved local directory: {resolved_dir}; smoke score failed: {result.get('reason', 'unknown error')}")
        return _success(f"Resolved local directory: {resolved_dir}; ONNX sessions initialized and smoke score completed")
    except Exception as exc:
        return _failure("ERROR", f"Resolved local directory: {resolved_dir}; {type(exc).__name__}: {exc}")


def _check_speechbrain(device: str) -> dict:
    try:
        import torch
        from core.model_utils import is_offline_mode
        from .models import _speaker_encoder, _speaker_model_path

        encoder = _speaker_encoder(device)
        with torch.inference_mode():
            encoder.encode_batch(torch.from_numpy(_smoke_audio()).unsqueeze(0).to(device))
        loc_desc = str(_speaker_model_path()) if is_offline_mode() else "speechbrain/spkrec-ecapa-voxceleb (Hub/local)"
        return _success(f"Resolved bundle: {loc_desc}; encoder initialized and smoke embedding completed")
    except Exception as exc:
        return _failure("ERROR", _model_load_failure("SpeechBrain initialization", exc))


def _check_squim(device: str) -> dict:
    try:
        import torch
        from .models import _squim_checkpoint_path, _squim_model

        path = _squim_checkpoint_path()
        model = _squim_model(device)
        with torch.inference_mode():
            model(torch.from_numpy(_smoke_audio()).unsqueeze(0).to(device))
        return _success(f"Resolved local weight: {path}; initialized on {device} and smoke score completed")
    except Exception as exc:
        return _failure("ERROR", _model_load_failure("SQUIM local initialization", exc))


def _check_nisqa(device: str) -> dict:
    try:
        from .models import _nisqa_checkpoint_path, _nisqa_one

        path = _nisqa_checkpoint_path()
        result = _nisqa_one(_smoke_audio(), device)
        if result.get("status") != "ok":
            return _failure("ERROR", f"NISQA local initialization failed: {result.get('reason', 'unknown error')}")
        return _success(f"Resolved local checkpoint: {path}; initialized and smoke score completed")
    except Exception as exc:
        return _failure("ERROR", f"NISQA local initialization failed: {type(exc).__name__}: {exc}")


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _smoke_audio() -> np.ndarray:
    """A deterministic 10-second waveform long enough for DNSMOS and all other metrics."""
    samples = np.arange(160000, dtype=np.float32)
    return (0.01 * np.sin(2 * np.pi * 220 * samples / 16000)).astype(np.float32)


def _success(details: str) -> dict:
    return {"status": "OK", "details": details}


def _failure(status: str, details: str) -> dict:
    return {"status": status, "details": details}


def _model_load_failure(stage: str, error: Exception) -> str:
    message = str(error)
    if isinstance(error, OSError) and "libtorchaudio.so" in message and "undefined symbol" in message:
        return (
            f"{stage} failed: Torch/Torchaudio ABI mismatch ({message}). "
            "Install torch and torchaudio as the same exact release and CUDA build in the active benchmark environment."
        )
    return f"{stage} failed: {type(error).__name__}: {error}"


def print_model_check_summary(results: dict) -> None:
    """Print strict readiness results in a Rich or ASCII table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Strict Local Benchmark Model Preflight", header_style="bold cyan")
        table.add_column("Model / Metric", style="bold white", justify="left")
        table.add_column("Status", justify="center")
        table.add_column("Resolution / Smoke Result", style="dim", justify="left")
        for key, info in results.items():
            status = info["status"]
            status_text = "[bold green]OK[/bold green]" if status == "OK" else f"[bold red]{status}[/bold red]"
            table.add_row(key, status_text, info["details"])
        console.print(table)
    except ImportError:
        print("\n=== Strict Local Benchmark Model Preflight ===")
        for key, info in results.items():
            print(f"[{info['status']}] {key}: {info['details']}")
        print()
