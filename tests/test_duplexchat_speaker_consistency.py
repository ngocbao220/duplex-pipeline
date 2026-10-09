from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat import separation_backend  # noqa: E402
from duplexchat.speaker_consistency import enforce_speaker_consistency  # noqa: E402

SR = 16_000
VOICE_HZ = {"A": 180.0, "B": 260.0}


def _voice(speaker: str, seconds: float, start: float = 0.0) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR + start
    # 0.5 s syllables separated by short pauses so the blocks find quiet cut points.
    gate = ((t % 0.6) < 0.5).float()
    return 0.3 * gate * torch.sin(2 * torch.pi * VOICE_HZ[speaker] * t)


def _embedder(audio: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft(audio).abs()
    freqs = torch.fft.rfftfreq(audio.numel(), 1 / SR)
    peak = float(freqs[int(spectrum.argmax())])
    # A shared component plus a small speaker-specific one, like real ECAPA vectors.
    voice_a = torch.tensor([1.0, 0.25, 0.0])
    voice_b = torch.tensor([1.0, 0.0, 0.25])
    return voice_a if abs(peak - VOICE_HZ["A"]) < abs(peak - VOICE_HZ["B"]) else voice_b


def _who(track: torch.Tensor, start: float, end: float) -> str:
    return "A" if _embedder(track.reshape(-1)[int(start * SR):int(end * SR)]) [1] > 0 else "B"


def test_swapped_middle_section_is_moved_back_to_its_channel():
    a, b = _voice("A", 60.0), _voice("B", 60.0)
    left = torch.cat([a[: 20 * SR], b[20 * SR: 40 * SR], a[40 * SR:]]).unsqueeze(0)
    right = torch.cat([b[: 20 * SR], a[20 * SR: 40 * SR], b[40 * SR:]]).unsqueeze(0)

    fixed_left, fixed_right, report = enforce_speaker_consistency(left, right, SR, _embedder)

    assert fixed_left.shape == left.shape and fixed_right.shape == right.shape
    assert report.swapped_blocks > 0
    for start in (5.0, 25.0, 33.0, 50.0):
        assert _who(fixed_left, start, start + 3.0) == "A"
        assert _who(fixed_right, start, start + 3.0) == "B"


def test_consistent_tracks_are_left_untouched():
    left, right = _voice("A", 30.0).unsqueeze(0), _voice("B", 30.0).unsqueeze(0)

    fixed_left, fixed_right, report = enforce_speaker_consistency(left, right, SR, _embedder)

    assert report.swapped_blocks == 0
    assert torch.equal(fixed_left, left) and torch.equal(fixed_right, right)


def test_chunk_seam_swap_ignores_waveform_phase():
    # Same speech envelope regenerated with a different phase, as a diffusion decoder does.
    t = torch.arange(SR * 2) / SR
    envelope_a = ((t % 0.6) < 0.3).float()
    envelope_b = 1.0 - envelope_a
    previous = torch.stack([envelope_a * torch.sin(2 * torch.pi * 200 * t),
                            envelope_b * torch.sin(2 * torch.pi * 200 * t)])
    regenerated = torch.stack([envelope_b * torch.cos(2 * torch.pi * 200 * t + 1.0),
                               envelope_a * torch.cos(2 * torch.pi * 200 * t + 1.0)])

    aligned, swapped = separation_backend._maybe_swap(previous, regenerated, previous.shape[-1])

    assert swapped
    assert torch.equal(aligned, regenerated[[1, 0]])
