from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "cholimex" / "src"))

from cholimex.blend import blend_channel  # noqa: E402
from cholimex.models import Region  # noqa: E402


RATE = 24_000
REGIONS = [
    Region(0.0, 0.5, "single_speaker", speaker=0),
    Region(0.5, 1.0, "overlap"),
    Region(1.0, 1.5, "single_speaker", speaker=0),
    Region(1.5, 2.0, "silence"),
]


def _tone(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _rms_db(audio: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(np.square(audio))))


def test_identical_sources_pass_through_without_seams():
    tone = _tone()
    out = blend_channel(tone, tone, REGIONS, speaker=0, sample_rate=RATE)

    assert np.allclose(out[1_000:30_000], tone[1_000:30_000], atol=1e-5)
    assert np.max(np.abs(np.diff(out[:35_000]))) <= np.max(np.abs(np.diff(tone))) + 1e-4


def test_gain_match_levels_overlap_to_neighbouring_original_speech():
    tone = _tone()
    out = blend_channel(tone, 0.5 * tone, REGIONS, speaker=0, sample_rate=RATE)

    before, inside = out[6_000:11_000], out[13_000:23_000]
    assert abs(_rms_db(before) - _rms_db(inside)) < 1.0


def test_hard_switch_is_replaced_by_a_crossfade():
    mixture = np.full(2 * RATE, 0.3, dtype=np.float32)
    separated = np.full(2 * RATE, -0.3, dtype=np.float32)
    out = blend_channel(mixture, separated, REGIONS, speaker=0, sample_rate=RATE, gain_match=False)

    seam = out[11_000:13_000]
    assert np.max(np.abs(np.diff(seam))) < 0.01
    assert np.isclose(out[11_000], 0.3, atol=1e-3) and np.isclose(out[13_000], -0.3, atol=1e-3)


def test_fade_into_silence_has_no_click():
    mixture = np.full(2 * RATE, 0.3, dtype=np.float32)
    out = blend_channel(mixture, mixture, REGIONS, speaker=0, sample_rate=RATE)

    tail = out[35_000:37_500]
    assert np.max(np.abs(np.diff(tail))) < 0.01
    assert np.allclose(out[36_500:], 0.0)
