from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat import diarization_backend  # noqa: E402


@pytest.mark.parametrize("auth_parameter", ["token", "use_auth_token"])
def test_pyannote_loader_uses_supported_auth_parameter(monkeypatch, auth_parameter):
    captured = {}

    class LoadedPipeline:
        def to(self, device):
            captured["device"] = str(device)
            return self

    if auth_parameter == "token":
        class FakePipeline:
            @classmethod
            def from_pretrained(cls, checkpoint, token=None):
                captured.update(checkpoint=checkpoint, token=token)
                return LoadedPipeline()
    else:
        class FakePipeline:
            @classmethod
            def from_pretrained(cls, checkpoint, use_auth_token=None):
                captured.update(checkpoint=checkpoint, use_auth_token=use_auth_token)
                return LoadedPipeline()

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = FakePipeline

    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setattr(diarization_backend, "enforce_offline_mode", lambda: None)
    monkeypatch.setattr(diarization_backend, "resolve_local_model_path", lambda *a, **k: (Path("/model"), True))
    monkeypatch.setattr(diarization_backend, "get_token", lambda: "test-token")
    monkeypatch.setattr(diarization_backend, "_patch_pyannote_hf_token_compat", lambda: None)

    diarization_backend._load_pyannote_pipeline("pyannote/speaker-diarization-community-1", "cpu")

    assert captured["checkpoint"] == "/model"
    assert captured[auth_parameter] == "test-token"
    assert captured["device"] == "cpu"
