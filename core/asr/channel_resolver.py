"""Attribute final stereo channels to the split-dialogue speaker labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from core.orchestration.contract import validate_sommelier_channel_mapping


def sommelier_mapping(stereo: Path, speakers: set[str]) -> dict:
    order = validate_sommelier_channel_mapping(stereo)
    if set(order) != speakers:
        raise ValueError(f"Sommelier channel mapping disagrees with speaker turns: {stereo}")
    return {"channel_speakers": order, "method": "sommelier_reconstruction", "confidence": 1.0}


def _exclusive_turns(turns: list[dict], speaker: str, duration: float) -> list[tuple[float, float]]:
    own = [(max(0.0, float(t["start"])), min(duration, float(t["end"]))) for t in turns if t["speaker"] == speaker]
    others = [(float(t["start"]), float(t["end"])) for t in turns if t["speaker"] != speaker]
    result = []
    for start, end in own:
        pieces = [(start, end)]
        for other_start, other_end in others:
            remaining = []
            for a, b in pieces:
                if other_start > a:
                    remaining.append((a, min(b, other_start)))
                if other_end < b:
                    remaining.append((max(a, other_end), b))
            pieces = [(a, b) for a, b in remaining if b > a]
        result.extend((a, b) for a, b in pieces if b - a >= 0.5)
    return result


def duplexchat_mapping(stereo_audio: np.ndarray, sample_rate: int, mixture_path: Path, turns: list[dict], extractor, min_score: float = 0.4, min_margin: float = 0.05) -> dict:
    """Compare two full-track channel voices with clean source-speaker references."""
    speakers = sorted({t["speaker"] for t in turns})
    if len(speakers) != 2 or stereo_audio.shape[1] != 2:
        raise ValueError("Channel matching requires exactly two speakers and channels")
    mixture, source_rate = sf.read(mixture_path, dtype="float32", always_2d=True)
    mixture = mixture.mean(axis=1)
    duration = min(len(mixture) / source_rate, len(stereo_audio) / sample_rate)
    matrix = np.zeros((2, 2), dtype=np.float32)
    for speaker_index, speaker in enumerate(speakers):
        spans = sorted(_exclusive_turns(turns, speaker, duration), key=lambda span: span[1] - span[0], reverse=True)[:6]
        if not spans:
            raise ValueError(f"No clean speaker reference for {speaker}")
        reference_embeddings = []
        channel_embeddings = [[], []]
        for start, end in spans:
            end = min(end, start + 3.0)
            ref = mixture[round(start * source_rate):round(end * source_rate)]
            reference_embeddings.append(extractor.extract(torch.from_numpy(ref.copy()), source_rate).numpy())
            for channel in range(2):
                sample = stereo_audio[round(start * sample_rate):round(end * sample_rate), channel]
                channel_embeddings[channel].append(extractor.extract(torch.from_numpy(sample.copy()), sample_rate).numpy())
        reference = np.mean(reference_embeddings, axis=0)
        reference /= max(np.linalg.norm(reference), 1e-8)
        for channel in range(2):
            candidate = np.mean(channel_embeddings[channel], axis=0)
            candidate /= max(np.linalg.norm(candidate), 1e-8)
            matrix[channel, speaker_index] = float(np.dot(candidate, reference))
    direct = float((matrix[0, 0] + matrix[1, 1]) / 2)
    swapped = float((matrix[0, 1] + matrix[1, 0]) / 2)
    order = speakers if direct >= swapped else speakers[::-1]
    assigned = [float(matrix[channel, speakers.index(order[channel])]) for channel in range(2)]
    margin = abs(direct - swapped)
    if min(assigned) < min_score or margin < min_margin:
        raise ValueError(f"Uncertain channel mapping: scores={matrix.tolist()}, margin={margin:.3f}")
    energy_checks = []
    for channel, speaker in enumerate(order):
        spans = _exclusive_turns(turns, speaker, duration)
        assigned_energy = other_energy = 0.0
        for start, end in spans:
            first, last = round(start * sample_rate), round(end * sample_rate)
            assigned_energy += float(np.square(stereo_audio[first:last, channel]).sum())
            other_energy += float(np.square(stereo_audio[first:last, 1 - channel]).sum())
        energy_checks.append({"speaker": speaker, "assigned_channel": channel,
                              "passed": assigned_energy >= other_energy})
    return {"channel_speakers": order, "method": "speechbrain_embedding", "confidence": min(1.0, margin / 2),
            "similarities": matrix.tolist(), "assignment_margin": margin, "energy_checks": energy_checks}
