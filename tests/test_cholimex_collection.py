from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "cholimex" / "src"))

from cholimex.collection import refine_stereo_file  # noqa: E402
from cholimex.models import ActivitySegment  # noqa: E402
from core.config import Config  # noqa: E402
from core.orchestration.cli import build_pipeline_parser  # noqa: E402


def test_collection_exports_channels_and_per_channel_speech_vad_labels(tmp_path):
    source = tmp_path / "stereo_1.wav"
    mixture = tmp_path / "mixture.wav"
    separated = np.column_stack([
        np.full(24_000, 0.1, dtype=np.float32),
        np.full(24_000, -0.2, dtype=np.float32),
    ])
    sf.write(source, separated, 24_000, subtype="FLOAT")
    sf.write(mixture, np.full(16_000, 0.3, dtype=np.float32), 16_000, subtype="FLOAT")

    def fake_vad(_audio, _sample_rate, speaker, **_kwargs):
        return [ActivitySegment(0.0, 0.5, speaker)] if speaker == 0 else [ActivitySegment(0.25, 0.75, speaker)]

    output = refine_stereo_file(source, tmp_path / "cholimex", Config(), mixture_path=mixture, vad_detector=fake_vad)

    assert output == tmp_path / "cholimex" / "cholimex_stereo_1.wav"
    assert sf.info(output).subtype == "PCM_16"
    refined, sample_rate = sf.read(output, always_2d=True, dtype="float32")
    assert sample_rate == 24_000
    assert np.isclose(refined[1_000, 0], 0.3, atol=1e-3)
    assert np.allclose(refined[:6_000, 1], 0.0)
    assert np.allclose(refined[1_000:5_000, 0], 0.3, atol=1e-3)
    assert np.allclose(refined[1_000:5_000, 1], 0.0)
    assert np.allclose(refined[6_000:12_000], separated[6_000:12_000], atol=1 / 32768)
    assert np.allclose(refined[12_000:18_000, 0], 0.0)
    assert np.allclose(refined[13_000:17_000, 1], 0.3, atol=1e-3)
    assert np.allclose(refined[18_000:], 0.0)
    left = tmp_path / "cholimex" / "left.wav"
    right = tmp_path / "cholimex" / "right.wav"
    assert sf.info(left).channels == sf.info(right).channels == 1
    assert sf.info(left).subtype == sf.info(right).subtype == "PCM_16"
    left_audio, _ = sf.read(left, dtype="float32")
    right_audio, _ = sf.read(right, dtype="float32")
    assert np.isclose(left_audio[1_000], 0.3, atol=1e-3)
    assert right_audio[0] == 0.0
    assert np.allclose(right_audio[7_000:11_000], -0.2, atol=1 / 32768)
    assert np.allclose(right_audio[13_000:17_000], 0.3, atol=1e-3)
    assert (tmp_path / "cholimex" / "vad_left.txt").read_text() == "0.000\t0.500\tspeech\n"
    assert (tmp_path / "cholimex" / "vad_right.txt").read_text() == "0.250\t0.750\tspeech\n"
    assert not (tmp_path / "cholimex" / "vad.txt").exists()
    assert source.is_file()


def test_collection_cli_accepts_one_stereo_input():
    args = build_pipeline_parser("cholimex").parse_args([
        "collection", "--input", "outputs/duplexchat-easy1/stereo_1.wav",
        "--mixture", "outputs/conversations/conversation_00000/mixture.wav", "--output-dir", "out",
    ])

    assert args.input == Path("outputs/duplexchat-easy1/stereo_1.wav")
    assert args.mixture == Path("outputs/conversations/conversation_00000/mixture.wav")
