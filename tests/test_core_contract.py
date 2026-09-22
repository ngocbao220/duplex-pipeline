import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from core.orchestration.contract import run_sample, validate_conversation_collection, validate_duplexchat_stereo_files, validate_stereo
from core.orchestration import worker
from core.orchestration.process import stream_process


def _wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(frames), 16000)
    return path


def _stereo_wav(path, frames=1600):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((frames, 2)), 16000)
    return path


def test_core_contract_requires_one_timeline_aligned_stereo_output(tmp_path):
    mixture = _wav(tmp_path / "mixture.wav")
    stereo = _stereo_wav(tmp_path / "audio.stereo.wav")
    assert validate_stereo(mixture, stereo) == 0.1
    _stereo_wav(stereo, frames=800)
    try:
        validate_stereo(mixture, stereo)
    except ValueError as error:
        assert "timeline" in str(error)
    else:
        raise AssertionError("short track must fail validation")


def test_duplexchat_collection_contract_allows_zero_conversations(tmp_path):
    manifest = tmp_path / "conversations" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text('{"conversation_count": 0, "conversations": []}')

    assert validate_conversation_collection(manifest) == 0


def test_duplexchat_flat_contract_requires_ordered_24khz_stereo_files(tmp_path):
    first = tmp_path / "stereo_1.wav"
    second = tmp_path / "stereo_2.wav"
    sf.write(first, np.zeros((240, 2)), 24_000)
    sf.write(second, np.zeros((240, 2)), 24_000)

    assert validate_duplexchat_stereo_files([first, second]) == 2


def test_duplexchat_zero_conversation_run_accepts_an_ffmpeg_only_input(tmp_path):
    source = tmp_path / "input.m4a"
    source.write_bytes(b"not a soundfile input")

    result = run_sample(
        "duplexchat", {"key": "sample", "mixture": str(source)}, tmp_path / "output", {}, "code",
        lambda *_args: (None, {"stereo_files": [], "devices": ["cpu"]}),
    )

    assert result["status"] == "complete"
    assert result["duration_sec"] == 0
    assert result["rtf"] is None


def test_duplexchat_worker_returns_a_conversation_collection(monkeypatch, tmp_path):
    captured = {}
    runner_module = types.ModuleType("duplexchat.runner")

    def run_single_audio(*_args, **kwargs):
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"stereo_files": [tmp_path / "output" / "stereo_1.wav"], "devices": ["cpu"]}

    runner_module.run_single_audio = run_single_audio
    package = types.ModuleType("duplexchat")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "duplexchat", package)
    monkeypatch.setitem(sys.modules, "duplexchat.runner", runner_module)
    collection, metadata = worker.duplexchat(
        tmp_path / "input.wav", tmp_path / "output", {"debug": True, "num_steps": 12}
    )

    assert metadata == {
        "stereo_files": [str(tmp_path / "output" / "stereo_1.wav")],
        "devices": ["cpu"],
        "phase_times": {},
        "audio_duration_sec": 0.0,
    }
    assert captured == {"output_dir": str(tmp_path / "output"), "debug": True, "num_steps": 12}
    assert json.loads((tmp_path / "output" / "config.json").read_text()) == {"num_steps": 12}


def test_failed_sample_emits_one_concise_console_error_and_persists_traceback(tmp_path, capsys, caplog):
    mixture = _wav(tmp_path / "mixture.wav")

    def failing_adapter(source, output, config):
        raise FileNotFoundError("checkpoint missing")

    result = run_sample("sommelier", {"key": "sample", "mixture": str(mixture)}, tmp_path / "output", {}, "code", failing_adapter)

    captured = capsys.readouterr().out + "\n".join(caplog.messages)
    assert result["status"] == "failed"
    assert "FileNotFoundError: checkpoint missing" in result["traceback"]
    assert "Sample sample failed: FileNotFoundError: checkpoint missing" in captured
    assert "Traceback" not in captured


def test_sommelier_contract_resumes_a_verified_complete_output(tmp_path):
    mixture = _wav(tmp_path / "mixture.wav")
    output = tmp_path / "output"

    def successful_adapter(_source, target, _config):
        return _stereo_wav(target / "stereo_1.wav"), {}

    first = run_sample("sommelier", {"key": "sample", "mixture": str(mixture)}, output, {}, "code", successful_adapter)
    resumed = run_sample(
        "sommelier", {"key": "sample", "mixture": str(mixture)}, output, {}, "code",
        lambda *_args: (_ for _ in ()).throw(AssertionError("completed Sommelier output must not rerun")),
    )

    assert first["status"] == "complete"
    assert first["stereo_path"] == "stereo_1.wav"
    assert not (output / "audio.stereo.wav").exists()
    assert resumed["status"] == "complete"
    assert resumed["resumed"] is True


def test_worker_forces_headless_matplotlib_backend_over_notebook_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    log = tmp_path / "worker.log"

    assert stream_process(
        [sys.executable, "-c", "import os; print(os.environ['MPLBACKEND'])"], tmp_path, log
    ) == 0

    assert log.read_text(encoding="utf-8").strip() == "Agg"
