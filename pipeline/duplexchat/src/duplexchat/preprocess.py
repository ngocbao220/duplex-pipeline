"""Purpose: Preserve and normalize a DuplexChat input into phase artifacts.

Inputs: Arbitrary source audio path and phase output directory.
Outputs: A 16 kHz source-preserving WAV, a 16 kHz mono WAV, and input metadata JSON.
"""
from pathlib import Path
import subprocess

from core.outputs import write_json


def write_input_wav(audio_path: Path, output_dir: Path) -> Path:
    """Decode the supplied audio at the pipeline sample rate without downmixing."""
    target = output_dir / "phase_00_input" / "input.wav"
    write_json(
        output_dir / "phase_00_input" / "input.json",
        {"audio_path": str(audio_path), "audio_name": audio_path.name, "sample_rate": 16_000},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return target


def prepare_input(audio_path: Path, output_dir: Path) -> Path:
    """Decode and downmix the supplied audio for diarization and separation."""
    target = output_dir / "phase_01_preprocess" / "audio_16k_mono.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(target)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target


def write_wav(path: Path, audio: object, sample_rate: int = 16000) -> Path:
    """Write float numpy/tensor audio to a 16-bit PCM WAV file."""
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio_np = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), audio_np, sample_rate, subtype="PCM_16")
    return path
