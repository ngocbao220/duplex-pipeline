from __future__ import annotations

import sys
import wave
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat import runner  # noqa: E402


def _segment(speaker: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "start": start, "end": end}


def test_runner_exports_flat_stereo_artifacts_without_debug(monkeypatch, tmp_path):
    output_root = tmp_path / "output"
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"placeholder")
    segments = [
        _segment("A", 0.0, 6.0),
        _segment("B", 6.0, 12.0),
        _segment("A", 17.0, 23.0),
        _segment("B", 23.0, 29.0),
    ]
    calls = {"diarize": 0, "load": 0, "separate": 0}

    monkeypatch.setattr(runner, "prepare_input", lambda *_args: source_wav)

    def fake_diarize(*_args, **_kwargs):
        calls["diarize"] += 1
        return object(), segments

    monkeypatch.setattr(runner, "diarize", fake_diarize)
    monkeypatch.setattr(
        runner,
        "load_wav_tensor",
        lambda *_args: (torch.zeros(1, 29 * 16_000), 16_000),
    )

    def fake_load(*_args, **_kwargs):
        calls["load"] += 1
        return object()

    def fake_separate(waveform, sample_rate, *_args):
        calls["separate"] += 1
        output_length = round(waveform.shape[-1] * 24_000 / sample_rate)
        output = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=output_length, mode="linear", align_corners=False,
        ).squeeze(0)
        return output.clone(), output.clone() + 1, 24_000

    monkeypatch.setattr(runner, "load_separation_models", fake_load)
    monkeypatch.setattr(runner, "separate_waveform", fake_separate)

    result = runner.run_single_audio(
        str(source_wav),
        output_dir=str(output_root),
        runtime_device="cpu",
    )

    assert calls == {"diarize": 1, "load": 1, "separate": 2}
    assert result["stereo_files"] == [output_root / "stereo_1.wav", output_root / "stereo_2.wav"]
    assert not (output_root / "conversations").exists()
    assert not (output_root / "debug").exists()
    for index, stereo in enumerate(result["stereo_files"], start=1):
        with wave.open(str(stereo)) as audio:
            assert audio.getnchannels() == 2
            assert audio.getframerate() == 24_000
        assert stereo.name == f"stereo_{index}.wav"


def test_runner_keeps_debug_phase_artifacts(monkeypatch, tmp_path):
    output_root = tmp_path / "output"
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"placeholder")
    segments = [_segment("A", 0.0, 6.0), _segment("B", 6.0, 12.0)]

    monkeypatch.setattr(runner, "prepare_input", lambda *_args: source_wav)
    monkeypatch.setattr(runner, "write_input_wav", lambda *_args: None)
    monkeypatch.setattr(runner, "diarize", lambda *_args, **_kwargs: (object(), segments))
    monkeypatch.setattr(runner, "load_wav_tensor", lambda *_args: (torch.zeros(1, 12 * 16_000), 16_000))
    monkeypatch.setattr(runner, "load_separation_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner,
        "separate_waveform",
        lambda waveform, sample_rate, *_args: (
            torch.zeros(1, round(waveform.shape[-1] * 24_000 / sample_rate)),
            torch.ones(1, round(waveform.shape[-1] * 24_000 / sample_rate)),
            24_000,
        ),
    )

    result = runner.run_single_audio(str(source_wav), output_dir=str(output_root), runtime_device="cpu", debug=True)

    debug = output_root / "debug"
    assert result["stereo_files"] == [debug / "phase_03_separation" / "conversation_1" / "stereo_1.wav"]
    assert (debug / "phase_02_diarization" / "speakers.txt").read_text() == "0.000\t6.000\tA\n6.000\t12.000\tB\n"
    assert (debug / "phase_02_diarization" / "conversation.txt").read_text() == "0.000\t12.000\tconversation_1\n"
    assert (debug / "phase_03_separation" / "conversation_1" / "left_1.wav").is_file()
    assert (debug / "phase_03_separation" / "conversation_1" / "right_1.wav").is_file()
    assert not (output_root / "stereo_1.wav").exists()


def test_split_valid_dialogues_exports_manifest_and_wavs(monkeypatch, tmp_path):
    output_root = tmp_path / "output_split"
    source_wav = tmp_path / "source.wav"
    source_wav.write_bytes(b"placeholder")
    segments = [
        _segment("A", 0.0, 6.0),
        _segment("B", 6.0, 12.0),
    ]

    monkeypatch.setattr(runner, "prepare_input", lambda *_args: source_wav)
    monkeypatch.setattr(runner, "diarize", lambda *_args, **_kwargs: (object(), segments))
    monkeypatch.setattr(
        runner,
        "load_wav_tensor",
        lambda *_args: (torch.zeros(1, 12 * 16_000), 16_000),
    )

    manifest = runner.split_valid_dialogues(
        str(source_wav),
        output_dir=str(output_root),
        filter_music=False,
    )

    assert manifest["dialogue_count"] == 1
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "dialogue_1.wav").is_file()
    assert "reason" in manifest["dialogues"][0]
    assert "accepted_standard_2_speaker_dialogue" in manifest["dialogues"][0]["reason"]


def test_separate_dialogue_files(monkeypatch, tmp_path):
    input_dir = tmp_path / "dialogue_in"
    input_dir.mkdir(parents=True, exist_ok=True)
    dialogue_wav = input_dir / "dialogue_1.wav"
    dialogue_wav.write_bytes(b"placeholder")
    output_dir = tmp_path / "separate_out"

    monkeypatch.setattr(runner, "load_wav_tensor", lambda *_args: (torch.zeros(1, 10 * 16_000), 16_000))
    monkeypatch.setattr(runner, "load_separation_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner,
        "separate_waveform",
        lambda waveform, sample_rate, *_args: (
            torch.zeros(1, round(waveform.shape[-1] * 24_000 / sample_rate)),
            torch.ones(1, round(waveform.shape[-1] * 24_000 / sample_rate)),
            24_000,
        ),
    )

    result = runner.separate_dialogue_files(
        str(input_dir),
        output_dir=str(output_dir),
        runtime_device="cpu",
    )

    assert result["dialogue_count"] == 1
    assert len(result["stereo_files"]) == 1
    assert (output_dir / "stereo_1.wav").is_file()
    with wave.open(str(output_dir / "stereo_1.wav")) as audio:
        assert audio.getnchannels() == 2
        assert audio.getframerate() == 24_000


