import json
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from core.asr.prepared_export import (export_config, export_prepared, select_voice_prompt,
                                      validate_words, words_for_trainer)

SR = 24_000


def _asr_result(duration: float, words_ch0, words_ch1):
    turns = [{"speaker": "S0", "start": 0.0, "end": 8.0}, {"speaker": "S1", "start": 9.0, "end": duration}]
    return {
        "duration": duration, "speaker_turns": turns,
        "provenance": {"channel_attribution": {"channel_speakers": ["S0", "S1"]}},
        "channels": [
            {"channel": 0, "speaker": "S0", "words": words_ch0},
            {"channel": 1, "speaker": "S1", "words": words_ch1},
        ],
    }


def _make_sample(root: Path, name="stereo_0.wav", ch0_words=None):
    duration = 18.0
    t = np.arange(int(duration * SR)) / SR
    left = np.where(t < 8.0, 0.3 * np.sin(2 * np.pi * 220 * t), 0.0)
    right = np.where(t >= 9.0, 0.3 * np.sin(2 * np.pi * 330 * t), 0.0)
    stereo = root / "vid1" / name
    stereo.parent.mkdir(parents=True, exist_ok=True)
    sf.write(stereo, np.stack([left, right], axis=1), SR, subtype="FLOAT")
    ch0 = ch0_words if ch0_words is not None else [
        {"word": "Xin", "start": 0.5, "end": 0.8, "aligned": True},
        {"word": "chào,", "start": 0.8, "end": 1.2, "aligned": True},
        {"word": "lỗi", "start": None, "end": None, "aligned": False},
    ]
    ch1 = [{"word": "Chào", "start": 9.5, "end": 9.9, "aligned": True},
           {"word": "bạn.", "start": 9.9, "end": 30.0, "aligned": True}]
    asr = stereo.with_name(stereo.stem.replace("stereo_", "dialogue_") + ".asr.json")
    asr.write_text(json.dumps(_asr_result(duration, ch0, ch1), ensure_ascii=False), encoding="utf-8")
    return stereo


def test_words_for_trainer_maps_roles_sorts_and_clamps():
    result = _asr_result(18.0, [{"word": "a", "start": 2.0, "end": 2.5, "aligned": True}],
                         [{"word": "b", "start": 1.0, "end": 99.0, "aligned": True},
                          {"word": " ", "start": 3.0, "end": 3.5, "aligned": True}])
    words = words_for_trainer(result, agent_channel=0, duration=18.0)
    assert [w["speaker"] for w in words] == ["user", "agent"]
    assert words[0]["end"] == 18.0
    assert validate_words(words, 18.0) == []
    swapped = words_for_trainer(result, agent_channel=1, duration=18.0)
    assert [w["speaker"] for w in swapped] == ["agent", "user"]


def test_validate_words_rules():
    ok = {"speaker": "agent", "word": "x", "start": 0.0, "end": 0.1}
    assert validate_words([ok], 1.0) == []
    assert any("speaker" in e for e in validate_words([{**ok, "speaker": "left"}], 1.0))
    assert any("empty" in e for e in validate_words([{**ok, "word": " "}], 1.0))
    assert any("start < 0" in e for e in validate_words([{**ok, "start": -0.1}], 1.0))
    assert any("end <= start" in e for e in validate_words([{**ok, "end": 0.0}], 1.0))
    assert any("out_of_bounds" in e for e in validate_words([{**ok, "end": 1.06}], 1.0))
    assert validate_words([{**ok, "end": 1.04}], 1.0) == []
    late = {**ok, "start": 0.5, "end": 0.6}
    assert any("sorted" in e for e in validate_words([late, ok], 1.0))
    assert "no agent words" in validate_words([{**ok, "speaker": "user"}], 1.0)


def test_voice_prompt_avoids_other_speaker():
    audio = np.full(20 * SR, 0.2, dtype=np.float32)
    clip = select_voice_prompt(audio, [(0.0, 12.0)], [(3.0, 4.0)], export_config({})["voice_prompt"])
    assert clip is not None and 5.0 <= len(clip) / SR <= 10.0
    assert select_voice_prompt(audio, [(0.0, 4.0)], [], export_config({})["voice_prompt"]) is None


def test_export_prepared_writes_trainer_contract(tmp_path):
    stereo_root = tmp_path / "duplexchat"
    _make_sample(stereo_root)
    _make_sample(stereo_root, "stereo_1.wav", ch0_words=[])  # no agent words -> rejected
    out = tmp_path / "ready"
    report = export_prepared(stereo_root, out, {})
    assert report["kept"] == 1 and report["rejected"] == 1
    entries = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert entries == [{"sample_id": "vid1__stereo_0", "sample_dir": "samples/vid1__stereo_0", "duration": 18.0}]
    sample = out / entries[0]["sample_dir"]
    with wave.open(str(sample / "conversation.wav"), "rb") as handle:
        assert handle.getnchannels() == 2 and handle.getsampwidth() == 2 and handle.getframerate() == SR
    words = json.loads((sample / "words.json").read_text(encoding="utf-8"))
    assert words[0] == {"speaker": "agent", "word": "Xin", "start": 0.5, "end": 0.8}
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["text_prompt_left"] == "Bạn là trợ lý thân thiện."
    assert metadata["agent_channel"] == "left" and metadata["swap_roles"] is False
    assert metadata["duration_sec"] == 18.0 and metadata["language"] == "vi"
    assert metadata["text_prompt_right"] == "Bạn là khách hàng hỏi thông tin."
    assert metadata["voice_prompt_left"] == metadata["voice_prompt_right"] == "same_conversation"
    assert (sample / "voice_prompt_left.wav").is_file() and (sample / "voice_prompt_right.wav").is_file()


def test_export_swap_roles_puts_right_speaker_on_left(tmp_path):
    stereo_root = tmp_path / "duplexchat"
    _make_sample(stereo_root)
    out = tmp_path / "ready"
    assert export_prepared(stereo_root, out, {"swap_roles": True})["kept"] == 1
    sample = out / "samples" / "vid1__stereo_0"
    audio, _ = sf.read(sample / "conversation.wav")
    assert np.abs(audio[: 8 * SR, 0]).max() < 1e-3 and np.abs(audio[10 * SR:, 0]).max() > 0.1
    words = json.loads((sample / "words.json").read_text(encoding="utf-8"))
    assert {w["speaker"] for w in words if w["start"] >= 9.0} == {"agent"}
    assert json.loads((sample / "metadata.json").read_text())["source_speakers"]["left"] == "S1"
