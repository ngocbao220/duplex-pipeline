"""Purpose: Blend original-mixture and DialogueSidon audio without seams.

Inputs: Mono mixture, one separated channel, classified regions and smoothing settings.
Outputs: One refined channel whose region boundaries are crossfaded and level-matched.
"""
from __future__ import annotations

import numpy as np

from .models import Region


OVERLAP_TYPES = {"overlap", "overlap_backchannel"}


def blend_channel(
    mixture: np.ndarray,
    separated: np.ndarray,
    regions: list[Region],
    speaker: int,
    sample_rate: int,
    crossfade_ms: float = 40.0,
    gate_fade_ms: float = 20.0,
    gain_match: bool = True,
    gain_context_s: float = 1.0,
    gain_max: float = 2.0,
) -> np.ndarray:
    """Use the mixture where only ``speaker`` talks and the separated track in overlaps.

    Hard region masks are smoothed with raised-cosine kernels so every switch between
    sources is a linear crossfade of ``crossfade_ms`` and every switch to silence is a
    fade of ``gate_fade_ms``; overlap audio is rescaled to the level of the
    neighbouring original speech so loudness does not jump at the seam.
    """
    length = len(separated)
    original_mask = np.zeros(length, dtype=np.float64)
    separated_mask = np.zeros(length, dtype=np.float64)
    for region in regions:
        start, end = _span(region, sample_rate, length)
        if region.type == "single_speaker" and region.speaker == speaker:
            original_mask[start:end] = 1.0
        elif region.type in OVERLAP_TYPES:
            separated_mask[start:end] = 1.0

    gain = np.ones(length, dtype=np.float64)
    if gain_match:
        gain = _gain_curve(mixture, separated, regions, original_mask, sample_rate, length,
                           crossfade_ms, gain_context_s, gain_max)

    crossfade = _hann_kernel(crossfade_ms, sample_rate)
    active = original_mask + separated_mask
    share = _smooth(separated_mask, crossfade)
    support = _smooth(active, crossfade)
    separated_share = np.where(support > 1e-6, np.clip(share / np.maximum(support, 1e-6), 0.0, 1.0), 0.0)
    envelope = np.clip(_smooth(active, _hann_kernel(gate_fade_ms, sample_rate)), 0.0, 1.0)

    blended = (1.0 - separated_share) * mixture + separated_share * gain * separated
    return (envelope * blended).astype(np.float32)


def _gain_curve(
    mixture: np.ndarray,
    separated: np.ndarray,
    regions: list[Region],
    original_mask: np.ndarray,
    sample_rate: int,
    length: int,
    crossfade_ms: float,
    context_s: float,
    gain_max: float,
) -> np.ndarray:
    fallback = _level_ratio(mixture, separated, original_mask > 0, gain_max)
    gain = np.ones(length, dtype=np.float64)
    reach = int(round(crossfade_ms / 1000.0 * sample_rate))
    context = int(round(context_s * sample_rate))
    for region in regions:
        if region.type not in OVERLAP_TYPES:
            continue
        start, end = _span(region, sample_rate, length)
        neighbours = np.zeros(length, dtype=bool)
        neighbours[max(0, start - context):start] = True
        neighbours[end:min(length, end + context)] = True
        neighbours &= original_mask > 0
        ratio = _level_ratio(mixture, separated, neighbours, gain_max) if neighbours.any() else fallback
        gain[max(0, start - reach):min(length, end + reach)] = ratio
    return gain


def _level_ratio(mixture: np.ndarray, separated: np.ndarray, mask: np.ndarray, gain_max: float) -> float:
    if not mask.any():
        return 1.0
    separated_rms = float(np.sqrt(np.mean(np.square(separated[mask], dtype=np.float64))))
    if separated_rms < 1e-8:
        return 1.0
    mixture_rms = float(np.sqrt(np.mean(np.square(mixture[mask], dtype=np.float64))))
    return float(np.clip(mixture_rms / separated_rms, 1.0 / gain_max, gain_max))


def _hann_kernel(duration_ms: float, sample_rate: int) -> np.ndarray:
    size = int(round(duration_ms / 1000.0 * sample_rate))
    if size <= 1:
        return np.ones(1)
    kernel = np.hanning(size + 2)[1:-1]
    return kernel / kernel.sum()


def _smooth(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(kernel) == 1:
        return mask
    return np.convolve(mask, kernel, mode="same")


def _span(region: Region, sample_rate: int, length: int) -> tuple[int, int]:
    start = max(0, min(length, int(round(region.start * sample_rate))))
    end = max(start, min(length, int(round(region.end * sample_rate))))
    return start, end
