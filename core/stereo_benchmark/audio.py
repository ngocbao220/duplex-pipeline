"""Input validation and channel-preserving stereo decoding."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class StereoAudio:
    path: Path
    sample_rate: int
    left: np.ndarray
    right: np.ndarray

    @property
    def frames(self) -> int:
        return len(self.left)

    @property
    def duration_sec(self) -> float:
        return self.frames / self.sample_rate


def load_stereo(path: Path) -> StereoAudio:
    """Load exactly two channels without downmixing or modifying the source file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    try:
        samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    except RuntimeError as error:
        raise ValueError(f"Cannot read audio file {path}: {error}") from error
    if samples.shape[1] != 2:
        raise ValueError(
            f"Stereo benchmark requires exactly two channels; got {samples.shape[1]}: {path}"
        )
    if samples.shape[0] == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return StereoAudio(path=path, sample_rate=int(sample_rate), left=samples[:, 0], right=samples[:, 1])
