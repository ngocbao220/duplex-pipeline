"""Purpose: Refine one DuplexChat stereo conversation with VAD-guided masking.

Inputs: A numbered 24 kHz DuplexChat stereo WAV, its original mixture and Cholimex settings.
Outputs: ``<output_dir>/stereo_N.wav`` (DuplexChat layout) plus left/right WAVs, VAD labels and
regions.json under ``<output_dir>/debug/conversation_N``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F_audio

from core.config import Config
from core.outputs import write_json, write_label_file

from .blend import blend_channel
from .models import ActivitySegment
from .regions import classify_regions
from .vad import run_silero_vad


OUTPUT_SAMPLE_RATE = 24_000
VAD_SAMPLE_RATE = 16_000
_STEREO_NAME = re.compile(r"stereo_(\d+)\.wav$")


def refine_stereo_file(
    input_path: Path,
    output_dir: Path,
    cfg: Config,
    mixture_path: Path | None = None,
    vad_detector: Callable = run_silero_vad,
) -> Path:
    """Combine original non-overlap audio with DuplexChat overlap audio, smoothing every seam."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    match = _STEREO_NAME.fullmatch(input_path.name)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not match:
        raise ValueError(f"Expected a numbered DuplexChat stereo WAV, got: {input_path.name}")
    output = output_dir / f"stereo_{match.group(1)}.wav"
    if output.exists():
        raise FileExistsError(f"Cholimex stereo output already exists: {output}")

    separated, sample_rate = sf.read(input_path, always_2d=True, dtype="float32")
    if sample_rate != OUTPUT_SAMPLE_RATE or separated.shape[1] != 2:
        raise ValueError(f"Expected a 24 kHz stereo WAV: {input_path}")
    if mixture_path is None:
        raise ValueError("The original mono mixture is required for non-overlap regions")
    mixture, mixture_rate = sf.read(Path(mixture_path), dtype="float32")
    if mixture.ndim != 1 or mixture_rate <= 0:
        raise ValueError(f"Expected a mono mixture: {mixture_path}")
    if mixture_rate != sample_rate:
        mixture_tensor = torch.from_numpy(mixture).unsqueeze(0)
        mixture = F_audio.resample(mixture_tensor, mixture_rate, sample_rate).squeeze(0).numpy()
    if len(mixture) < len(separated):
        mixture = np.pad(mixture, (0, len(separated) - len(mixture)))
    elif len(mixture) > len(separated):
        mixture = mixture[:len(separated)]

    left_segments = _detect_activity(separated[:, 0], sample_rate, 0, cfg, vad_detector)
    right_segments = _detect_activity(separated[:, 1], sample_rate, 1, cfg, vad_detector)
    regions = classify_regions(
        left_segments,
        right_segments,
        duration_sec=len(separated) / sample_rate,
        backchannel_max_duration=cfg.cholimex_backchannel_max_duration,
    )
    refined = np.column_stack([
        blend_channel(
            mixture, separated[:, channel], regions, channel, sample_rate,
            crossfade_ms=cfg.cholimex_crossfade_ms,
            gate_fade_ms=cfg.cholimex_gate_fade_ms,
            gain_match=cfg.cholimex_gain_match,
            gain_context_s=cfg.cholimex_gain_context_s,
            gain_max=cfg.cholimex_gain_max,
        )
        for channel in (0, 1)
    ])

    debug_dir = output_dir / "debug" / f"conversation_{match.group(1)}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    sf.write(debug_dir / "left.wav", refined[:, 0], sample_rate, subtype="PCM_16")
    sf.write(debug_dir / "right.wav", refined[:, 1], sample_rate, subtype="PCM_16")
    write_label_file(
        debug_dir / "vad_left.txt",
        ((segment.start, segment.end, "speech") for segment in left_segments),
    )
    write_label_file(
        debug_dir / "vad_right.txt",
        ((segment.start, segment.end, "speech") for segment in right_segments),
    )
    write_json(debug_dir / "regions.json", [region.to_dict() for region in regions])
    # Written last via rename so a present stereo_N.wav always means a complete run.
    partial = output.with_name(f".{output.stem}.partial.wav")
    sf.write(partial, refined, sample_rate, subtype="PCM_16")
    partial.replace(output)
    return output


def _detect_activity(audio: np.ndarray, sample_rate: int, speaker: int, cfg: Config, vad_detector: Callable) -> list[ActivitySegment]:
    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != VAD_SAMPLE_RATE:
        waveform = F_audio.resample(waveform, sample_rate, VAD_SAMPLE_RATE)
    return vad_detector(
        waveform, VAD_SAMPLE_RATE, speaker=speaker, threshold=cfg.cholimex_vad_onset,
        offset=cfg.cholimex_vad_offset, padding_ms=cfg.cholimex_vad_padding_ms,
        min_duration=cfg.cholimex_min_vad_duration,
        merge_gap=cfg.cholimex_merge_gap,
    )

