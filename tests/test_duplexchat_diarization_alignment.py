import json
import math
import wave

import torch

from duplexchat import runner
from duplexchat.diarization_alignment import align_to_diarization

SR = 16_000


def _tone(freq, seconds):
    t = torch.arange(int(seconds * SR)) / SR
    return 0.3 * torch.sin(2 * math.pi * freq * t)


def _conversation():
    """Speaker 0 (200 Hz) and speaker 1 (500 Hz) alternate every 2 s for 20 s."""
    turns, a, b = [], [], []
    for index in range(10):
        speaker = index % 2
        turns.append({"start": 2.0 * index, "end": 2.0 * index + 1.8, "speaker": f"speaker_{speaker}"})
        voice = torch.cat([_tone(200 if speaker == 0 else 500, 1.8), torch.zeros(int(0.2 * SR))])
        silence = torch.zeros_like(voice)
        a.append(voice if speaker == 0 else silence)
        b.append(silence if speaker == 0 else voice)
    return torch.cat(a), torch.cat(b), turns


def test_alignment_undoes_a_swap_inside_the_conversation():
    first, second, turns = _conversation()
    cut = 8 * SR  # the separator swapped its outputs from 8 s on
    swapped_first = torch.cat([first[:cut], second[cut:]])
    swapped_second = torch.cat([second[:cut], first[cut:]])

    fixed_first, fixed_second, report = align_to_diarization(swapped_first, swapped_second, SR, turns)

    assert len(report.segments) == 1
    assert 7.8 <= report.segments[0][0] <= 8.0 and report.segments[0][1] == 20.0
    assert torch.allclose(fixed_first, first, atol=1e-4)
    assert torch.allclose(fixed_second, second, atol=1e-4)


def test_alignment_keeps_consistent_tracks_and_needs_two_speakers():
    first, second, turns = _conversation()
    kept_first, kept_second, report = align_to_diarization(first, second, SR, turns)
    assert report.segments == [] and kept_first is first and kept_second is second

    one_speaker = [turn for turn in turns if turn["speaker"] == "speaker_0"]
    _, _, report = align_to_diarization(second, first, SR, one_speaker)
    assert report.windows == 0


def test_separate_dialogue_files_uses_diarization_turns(monkeypatch, tmp_path):
    first, second, turns = _conversation()
    input_dir = tmp_path / "dialogue"
    input_dir.mkdir()
    (input_dir / "dialogue_1.wav").write_bytes(b"placeholder")
    (input_dir / "dialogue_1.json").write_text(json.dumps({"speaker_turns": turns}))
    calls = []

    monkeypatch.setattr(runner, "load_wav_tensor", lambda *_args: ((first + second).unsqueeze(0), SR))
    monkeypatch.setattr(runner, "load_separation_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "validate_stereo", lambda *_args: None)

    def fake_separate(waveform, sample_rate, *args):
        calls.append(args[-1])  # speaker_consistency flag
        return second.unsqueeze(0), first.unsqueeze(0), SR  # fully swapped output

    monkeypatch.setattr(runner, "separate_waveform", fake_separate)
    runner.separate_dialogue_files(str(input_dir), output_dir=str(tmp_path / "out"), runtime_device="cpu")

    assert calls == [False]
    with wave.open(str(tmp_path / "out" / "stereo_1.wav")) as audio:
        frames = torch.frombuffer(bytearray(audio.readframes(audio.getnframes())), dtype=torch.int16)
    left = frames[0::2].float()
    # 0.5 s of speaker 0 at 24 kHz must be on the left channel.
    assert left[int(0.5 * 24_000):int(1.0 * 24_000)].abs().mean() > 1000
