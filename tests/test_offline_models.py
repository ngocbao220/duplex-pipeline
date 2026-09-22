"""Tests for offline model resolution and fail-fast validation in duplex-pipelines."""
import hashlib
import json
import os
import zipfile
import pytest
from pathlib import Path

from core.model_utils import (
    assert_local_model_exists,
    enforce_offline_mode,
    resolve_local_model_path,
)


def _write_dialoguesidon_bundle(model_dir, *, corrupt_ssl: bool = False) -> None:
    """Create a structurally valid local DialogueSidon bundle for preflight tests."""
    model_dir.mkdir()
    hashes = {}
    for filename in ("ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2"):
        path = model_dir / filename
        if corrupt_ssl and filename == "ssl_encoder.pt2":
            path.write_bytes(b"not a zip archive")
        else:
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("version", "1")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = model_dir / "metadata.json"
    metadata.write_text(json.dumps({"latent_norm_mean": [], "latent_norm_std": [], "latent_norm_initialized": True, "ddpm_config": {}, "latent_dim": 1, "sample_rate": 24000}))
    hashes[metadata.name] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    (model_dir / "model_manifest.json").write_text(json.dumps({"files": hashes}))


def test_enforce_offline_mode(monkeypatch):
    monkeypatch.setenv("MODE", "sever")
    enforce_offline_mode()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_resolve_local_model_path_existing_dir(tmp_path):
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    
    path, is_dir = resolve_local_model_path(str(model_dir))
    assert is_dir is True
    assert path == model_dir.resolve()


def test_resolve_local_model_path_env_var(tmp_path, monkeypatch):
    model_dir = tmp_path / "env_model"
    model_dir.mkdir()
    monkeypatch.setenv("TEST_MODEL_PATH", str(model_dir))

    path, is_dir = resolve_local_model_path("non_existent_repo", env_var="TEST_MODEL_PATH")
    assert is_dir is True
    assert path == model_dir.resolve()


def test_resolve_local_model_path_base_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "models"
    base_dir.mkdir()
    target_sub = base_dir / "DialogueSidon"
    target_sub.mkdir()

    monkeypatch.setenv("DUPLEX_MODEL_DIR", str(base_dir))

    path, is_dir = resolve_local_model_path("sarulab-speech/DialogueSidon")
    assert is_dir is True
    assert path == target_sub.resolve()


def test_offline_resolution_fails_with_the_configured_model_path(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing-silero"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", str(missing_path))

    with pytest.raises(FileNotFoundError, match=rf"No found model snakers4/silero-vad on path: {missing_path}"):
        resolve_local_model_path(
            "snakers4/silero-vad", env_var="SILERO_VAD_MODEL_PATH", default_subpath="silero-vad"
        )


def test_assert_local_model_exists_missing_dir():
    with pytest.raises(FileNotFoundError, match="Local model path '.*non_existent.*' does not exist"):
        assert_local_model_exists("/non_existent_dir_12345", model_name_hint="TestModel")


def test_assert_local_model_exists_missing_file(tmp_path):
    model_dir = tmp_path / "incomplete_model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="missing required file.*ssl_encoder.pt2"):
        assert_local_model_exists(
            model_dir, required_files=["metadata.json", "ssl_encoder.pt2"], model_name_hint="DialogueSidon"
        )


def test_dialoguesidon_local_bundle_requires_matching_checksum_manifest(tmp_path):
    from core.local_model_validation import validate_dialoguesidon_model

    model_dir = tmp_path / "DialogueSidon"
    _write_dialoguesidon_bundle(model_dir)

    paths = validate_dialoguesidon_model(model_dir)

    assert paths["ssl_encoder.pt2"] == str((model_dir / "ssl_encoder.pt2").resolve())


def test_dialoguesidon_local_bundle_rejects_corrupt_export_before_loading(tmp_path):
    from core.local_model_validation import LocalModelValidationError, validate_dialoguesidon_model

    model_dir = tmp_path / "DialogueSidon"
    _write_dialoguesidon_bundle(model_dir, corrupt_ssl=True)

    with pytest.raises(LocalModelValidationError, match=r"ssl_encoder\.pt2.*not a readable ZIP"):
        validate_dialoguesidon_model(model_dir)


def test_dialoguesidon_local_bundle_rejects_checksum_mismatch(tmp_path):
    from core.local_model_validation import LocalModelValidationError, validate_dialoguesidon_model

    model_dir = tmp_path / "DialogueSidon"
    _write_dialoguesidon_bundle(model_dir)
    (model_dir / "ssl_encoder.pt2").write_bytes(b"tampered")

    with pytest.raises(LocalModelValidationError, match=r"ssl_encoder\.pt2.*SHA-256 mismatch"):
        validate_dialoguesidon_model(model_dir)


def test_dialoguesidon_manifest_writer_creates_a_valid_bundle_manifest(tmp_path):
    from core.local_model_validation import write_dialoguesidon_manifest, validate_dialoguesidon_model

    model_dir = tmp_path / "DialogueSidon"
    _write_dialoguesidon_bundle(model_dir)
    (model_dir / "model_manifest.json").unlink()

    manifest = write_dialoguesidon_manifest(model_dir)

    assert manifest.exists()
    assert validate_dialoguesidon_model(model_dir)["vae_decoder.pt2"].endswith("vae_decoder.pt2")


def test_noop_lid_filter():
    from duplexchat.language_id import NoOpLIDFilter
    import torch
    filter_obj = NoOpLIDFilter()
    is_vi, prob = filter_obj.is_vietnamese(torch.zeros(16000), 16000)
    assert is_vi is True
    assert prob == 1.0


def test_load_local_silero_vad_resolution(tmp_path, monkeypatch):
    silero_dir = tmp_path / "silero-vad"
    silero_dir.mkdir()
    onnx_file = silero_dir / "silero_vad.onnx"
    onnx_file.touch()

    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", str(onnx_file))
    from core.model_utils import resolve_local_model_path
    resolved, is_dir = resolve_local_model_path("snakers4/silero-vad", env_var="SILERO_VAD_MODEL_PATH")
    assert resolved == onnx_file.resolve()
    assert is_dir is True


def test_local_silero_vad_fallback_converts_numpy_audio_to_tensor():
    """TorchScript Silero models reject the NumPy chunks emitted by librosa."""
    import numpy as np
    import torch

    from core.model_utils import _get_speech_timestamps_fallback

    class TensorOnlyVAD(torch.nn.Module):
        def forward(self, audio, sample_rate):
            assert isinstance(audio, torch.Tensor)
            assert audio.dtype == torch.float32
            assert sample_rate == 16_000
            return torch.tensor(0.9)

    timestamps = _get_speech_timestamps_fallback(
        np.zeros(1_024, dtype=np.float32), TensorOnlyVAD(), sampling_rate=16_000
    )

    assert timestamps == [{"start": 0, "end": 1_024}]


def test_offline_silero_missing_local_model_never_calls_torch_hub(monkeypatch, tmp_path):
    import torch
    from core.model_utils import load_local_silero_vad

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", str(tmp_path / "missing-silero"))
    monkeypatch.setattr(torch.hub, "load", lambda *_args, **_kwargs: pytest.fail("Torch Hub must not run offline"))

    with pytest.raises(FileNotFoundError, match="No found model snakers4/silero-vad"):
        load_local_silero_vad()


def test_sommelier_offline_preflight_reports_every_missing_local_model(monkeypatch, tmp_path, capsys):
    """Server mode must stop before the vendor subprocess can try a remote loader."""
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "pipeline" / "sommelier" / "src"))
    from sommelier.runner import validate_offline_models

    monkeypatch.setenv("MODE", "sever")
    missing = tmp_path / "models"
    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", str(missing / "silero-vad"))
    monkeypatch.setenv("SORTFORMER_MODEL_PATH", str(missing / "sortformer"))
    monkeypatch.setenv("SPEECHBRAIN_MODEL_PATH", str(missing / "spker"))
    monkeypatch.setenv("SEPREFORMER", str(missing / "epoch.0180.pth"))

    with pytest.raises(FileNotFoundError, match="Sommelier local model preflight failed"):
        validate_offline_models()

    messages = capsys.readouterr().out
    for name, path in (
        ("Silero VAD", missing / "silero-vad"),
        ("Sortformer", missing / "sortformer"),
        ("SpeechBrain ECAPA", missing / "spker"),
        ("SepReformer", missing / "epoch.0180.pth"),
    ):
        assert f"No found model {name} on path: {path}" in messages


def test_sommelier_run_stops_before_vendor_subprocess_when_offline_models_are_missing(monkeypatch, tmp_path):
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "pipeline" / "sommelier" / "src"))
    from sommelier import runner

    monkeypatch.setenv("MODE", "sever")
    for env_var in ("SILERO_VAD_MODEL_PATH", "SORTFORMER_MODEL_PATH", "SPEECHBRAIN_MODEL_PATH", "SEPREFORMER"):
        monkeypatch.setenv(env_var, str(tmp_path / env_var.lower()))
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("vendor must not start"))

    source = tmp_path / "input.wav"
    source.touch()
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="Sommelier local model preflight failed"):
        runner.run(source, output, {})

    assert not output.exists()
