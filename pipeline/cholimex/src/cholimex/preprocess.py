"""Purpose: Normalize Cholimex input audio to 16 kHz mono tensors.

Inputs: Source audio paths and WAV files.
Outputs: Standardized WAV artifacts and waveform tensors.
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import torch


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be installed and on PATH")


def transcode_to_wav_16k_mono(input_path: Path, output_path: Path) -> Path:
    """Standardize arbitrary input audio to the timeline used by shared models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000",
         "-vn", "-f", "wav", str(output_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_path


def load_wav_tensor(wav_path: Path) -> tuple[torch.Tensor, int]:
    """Read PCM WAV as a mono float32 tensor shaped ``(1, samples)``."""
    with wave.open(str(wav_path), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width == 2:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    elif width == 4:
        samples = torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    waveform = samples.reshape(-1, channels).T.contiguous()
    return (waveform.mean(dim=0, keepdim=True) if channels > 1 else waveform), sample_rate
