"""Utilities for handling local/offline model path resolution and fail-fast verification.

Supports running offline on servers where public Internet / HuggingFace Hub is forbidden.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def is_offline_mode() -> bool:
    """Return True if offline mode is enforced via environment variable."""
    mode = os.environ.get("MODE") or os.environ.get("PIPELINE_MODE")
    if mode:
        return str(mode).strip().lower() in ("sever", "server", "offline", "prod", "production")
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def enforce_offline_mode() -> None:
    """Set environment variables enforcing offline behavior only if offline mode is enabled."""
    if is_offline_mode():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        # Online / dev mode: allow Hugging Face downloads
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "3"


def resolve_local_model_path(
    model_identifier: str | Path | None,
    env_var: str | None = None,
    default_subpath: str | None = None,
) -> tuple[Path | str, bool]:
    """Resolve a model path from explicit identifier, environment variable, or base DUPLEX_MODEL_DIR.

    Returns:
        (resolved_path_or_identifier, is_local_dir)
    """
    # 1. Check explicit model_identifier if provided as local path
    if model_identifier is not None:
        p = Path(model_identifier)
        if p.exists():
            return p.resolve(), True

    # 2. Check specific environment variable if set
    if env_var and os.environ.get(env_var):
        env_path = Path(os.environ[env_var])
        if env_path.exists():
            return env_path.resolve(), True

    # 3. Check DUPLEX_MODEL_DIR base directory
    base_dir = os.environ.get("DUPLEX_MODEL_DIR")
    if base_dir:
        base_path = Path(base_dir)
        subs: list[str] = []
        if model_identifier:
            subs.append(str(model_identifier))
            subs.append(Path(str(model_identifier)).name)
        if default_subpath:
            subs.append(default_subpath)
            subs.append(Path(default_subpath).name)

        for sub in subs:
            if not sub:
                continue
            candidates = [
                base_path / sub,
                base_path / f"{sub}.nemo",
                base_path / f"{sub}.pt",
                base_path / f"{sub}.pth",
                base_path / f"{sub}.jit",
                base_path / f"{sub}.tar",
                base_path / f"{sub}.onnx",
            ]
            for cand in candidates:
                if cand.exists():
                    return cand.resolve(), True

    # 3.5 Check default subpath and model name in local directories (., models, ..)
    local_bases = [Path("."), Path("models"), Path(".."), Path("../models")]
    local_subs: list[str] = []
    if default_subpath:
        local_subs.append(default_subpath)
        local_subs.append(Path(default_subpath).name)
    if model_identifier:
        local_subs.append(Path(str(model_identifier)).name)

    for base in local_bases:
        for sub in local_subs:
            if not sub:
                continue
            candidates = [
                base / sub,
                base / f"{sub}.nemo",
                base / f"{sub}.pt",
                base / f"{sub}.pth",
                base / f"{sub}.jit",
                base / f"{sub}.tar",
                base / f"{sub}.onnx",
            ]
            for cand in candidates:
                if cand.exists():
                    return cand.resolve(), True

    if is_offline_mode():
        configured_path = os.environ.get(env_var, "") if env_var else ""
        if not configured_path and base_dir and default_subpath:
            configured_path = str(Path(base_dir) / default_subpath)
        if not configured_path:
            configured_path = default_subpath or str(model_identifier or "")
        raise FileNotFoundError(
            f"No found model {model_identifier or default_subpath} on path: {configured_path}. "
            "Hub and Torch Hub fallback are disabled in offline mode."
        )

    # 4. Fallback to model_identifier string (to be used with local_files_only=True)
    res_id = str(model_identifier) if model_identifier is not None else ""
    return res_id, False


def require_local_model_path(
    model_identifier: str | Path | None,
    *,
    env_var: str | None = None,
    default_subpath: str | None = None,
    model_name_hint: str = "Model",
    require_file: bool = False,
) -> Path:
    """Resolve a local model and reject identifier fallbacks before any Hub loader runs."""
    target, is_local = resolve_local_model_path(model_identifier, env_var, default_subpath)
    path = Path(target)
    expected = f"Set {env_var}" if env_var else "Provide an explicit local path"
    if not is_local or not path.exists() or (require_file and not path.is_file()):
        kind = "file" if require_file else "path"
        raise FileNotFoundError(
            f"[{model_name_hint}] A local {kind} is required; resolved '{target}' is not usable. "
            f"{expected}. Hub and network fallback are disabled."
        )
    return path.resolve()


def assert_local_model_exists(
    model_location: str | Path,
    required_files: list[str] | None = None,
    model_name_hint: str = "Model",
) -> Path:
    """Verify that a local model path exists and contains required files (Fail-Fast).

    Raises FileNotFoundError with a clear message if files are missing, ensuring
    no auto-download from HuggingFace/Internet is ever attempted.
    """
    path = Path(model_location)
    if not path.exists():
        raise FileNotFoundError(
            f"[{model_name_hint}] Local model path '{path}' does not exist. "
            f"Automatic Internet downloading is disabled in offline mode. "
            f"Please copy model files to server and set the model path via configuration or environment variable."
        )

    if required_files:
        missing = [f for f in required_files if not (path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"[{model_name_hint}] Local model path '{path}' is missing required file(s): {missing}. "
                f"Please ensure all required model weights and metadata files are copied to the server."
            )

    # Verify model files are not Git LFS text pointers
    check_files = [path] if path.is_file() else [path / f for f in (required_files or []) if (path / f).is_file()]
    for file_path in check_files:
        if file_path.stat().st_size < 2000:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if "git-lfs.github.com" in content or "version https://git-lfs" in content:
                    raise ValueError(
                        f"[{model_name_hint}] Local model file '{file_path.name}' at '{file_path}' is a Git LFS text pointer file (~{file_path.stat().st_size} bytes), "
                        f"not the full binary model weights. Copy the verified binary artifact into the local model bundle."
                    )
            except (UnicodeDecodeError, PermissionError):
                pass

    return path.resolve()


def get_offline_loader_kwargs(extra_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return dictionary of loader options specifying local_files_only=True in offline/sever mode."""
    enforce_offline_mode()
    kwargs: dict[str, Any] = {"local_files_only": is_offline_mode()}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def load_local_silero_vad(device: str = "cpu"):
    """Load Silero VAD model and helper utils supporting silero_vad.jit, hubconf.py, or PyTorch Hub cache.

    Returns:
        (vad_model, get_speech_timestamps_fn)
    """
    import torch

    enforce_offline_mode()
    local_target, is_dir = resolve_local_model_path(
        "snakers4/silero-vad", env_var="SILERO_VAD_MODEL_PATH", default_subpath="silero-vad"
    )

    path_obj = Path(local_target)
    if path_obj.is_file():
        if path_obj.suffix.lower() in (".jit", ".pt", ".onnx"):
            return torch.jit.load(str(path_obj), map_location=device), _get_speech_timestamps_fallback

    if is_dir or path_obj.is_dir():
        # 1. Check for silero_vad.jit file (PyTorch native JIT model)
        for candidate in [path_obj / "silero_vad.jit", path_obj / "files" / "silero_vad.jit", path_obj / "silero_vad.pt"]:
            if candidate.exists():
                model = torch.jit.load(str(candidate), map_location=device)
                return model, _get_speech_timestamps_fallback

        # 2. Check for hubconf.py in local folder
        if (path_obj / "hubconf.py").exists():
            vad_model, utils = torch.hub.load(
                repo_or_dir=str(path_obj), model="silero_vad", source="local", trust_repo=True, verbose=False, onnx=False
            )
            return vad_model, utils[0]

    # Fallback to torch hub / local cache
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir=str(local_target), model="silero_vad", trust_repo=True, verbose=False, onnx=False
        )
        return vad_model, utils[0]
    except Exception:
        if path_obj.exists():
            jit_files = list(path_obj.glob("**/*.jit")) or list(path_obj.glob("**/*.pt"))
            if jit_files:
                return torch.jit.load(str(jit_files[0]), map_location=device), _get_speech_timestamps_fallback
        raise


def _get_speech_timestamps_fallback(audio, model, threshold=0.5, sampling_rate=16000, min_speech_duration_ms=250, min_silence_duration_ms=100, **kwargs):
    """Fallback implementation of get_speech_timestamps for ONNX and JIT model weights."""
    import torch

    audio = torch.as_tensor(audio, dtype=torch.float32)
    try:
        audio = audio.to(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        pass

    window_size_samples = 512 if sampling_rate == 16000 else 256
    audio_length_samples = len(audio)
    speech_probs = []

    if hasattr(model, "reset_states"):
        model.reset_states()

    for current_start in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start: current_start + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
        with torch.no_grad():
            prob_tensor = model(chunk, sampling_rate)
            prob = float(prob_tensor.squeeze().item()) if torch.is_tensor(prob_tensor) else float(prob_tensor)
        speech_probs.append(prob)

    triggered = False
    speeches = []
    current_speech = {}
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000

    for i, prob in enumerate(speech_probs):
        current_sample = i * window_size_samples
        if prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = current_sample
        elif prob < (threshold - 0.15) and triggered:
            triggered = False
            current_speech["end"] = current_sample
            if current_speech["end"] - current_speech["start"] >= min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}

    if triggered:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    return speeches
