from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat import preprocess, separation_backend  # noqa: E402


def test_blend_overlap_clamps_mismatched_chunk_lengths():
    stitched = torch.zeros(2, 3_000)
    pred = torch.ones(2, 383)

    blended = separation_backend._blend_overlap(stitched, pred, 3_000)

    assert blended.shape == (2, 3_000)


def test_maybe_swap_clamps_mismatched_overlap_lengths():
    previous = torch.zeros(2, 3_000)
    current = torch.zeros(2, 383)

    result, swapped = separation_backend._maybe_swap(previous, current, 3_000)

    assert result.shape == current.shape
    assert swapped is False


def test_prepare_input_creates_16k_mono_artifact(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)

    result = preprocess.prepare_input(tmp_path / "source.mp3", tmp_path / "phases")

    assert result == tmp_path / "phases" / "phase_01_preprocess" / "audio_16k_mono.wav"
    command, kwargs = calls[0]
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[-1] == str(result)
    assert kwargs["check"] is True


def test_write_input_wav_resamples_without_downmixing(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)

    result = preprocess.write_input_wav(tmp_path / "source.m4a", tmp_path / "debug")

    command, kwargs = calls[0]
    assert result == tmp_path / "debug" / "phase_00_input" / "input.wav"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert "-ac" not in command
    assert kwargs["check"] is True


def test_run_separation_resamples_24k_input_to_16k_before_inference(monkeypatch):
    captured = {}

    def fake_resample(waveform, source_rate, target_rate):
        captured["rates"] = (source_rate, target_rate)
        return torch.zeros(1, 16_000)

    def fake_separate_chunk(waveform, _num_steps, _models):
        captured["inference_samples"] = waveform.shape[-1]
        return torch.zeros(2, 16_000)

    monkeypatch.setattr(separation_backend.F_audio, "resample", fake_resample)
    monkeypatch.setattr(separation_backend, "_separate_chunk", fake_separate_chunk)

    first, second, output_rate = separation_backend.run_separation(
        torch.zeros(1, 24_000),
        24_000,
        num_steps=30,
        models={"device": torch.device("cpu"), "sample_rate": 24_000},
        chunk_seconds=120.0,
    )

    assert captured["rates"] == (24_000, 16_000)
    assert captured["inference_samples"] == 16_320
    assert output_rate == 24_000
    assert first.shape == second.shape == (1, 24_000)
