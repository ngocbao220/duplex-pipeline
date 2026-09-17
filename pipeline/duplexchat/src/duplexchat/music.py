"""Purpose: Apply Demucs music separation to remove background music from DuplexChat dialogues.

Inputs: Waveform tensor, model settings, and target runtime device.
Outputs: Cleaned speech waveform tensor.
"""
from __future__ import annotations

import logging
import torch

LOGGER = logging.getLogger("duplexchat")


def load_music_filter(
    model_name: str = "htdemucs",
    device: str = "auto",
    enabled: bool = True,
) -> DemucsMusicFilter | NoOpMusicFilter:
    if not enabled:
        return NoOpMusicFilter()
    try:
        return DemucsMusicFilter(model_name=model_name, device=device)
    except Exception as exc:
        LOGGER.warning("Could not load Demucs music filter model '%s': %s", model_name, exc)
        return NoOpMusicFilter()


class NoOpMusicFilter:
    """Fallback filter when music filtering is disabled or Demucs is unavailable."""

    def filter_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return waveform.clone()


class DemucsMusicFilter:
    """Separate and suppress background music/accompaniment using Demucs (htdemucs)."""

    def __init__(self, model_name: str = "htdemucs", device: str = "auto") -> None:
        try:
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Demucs library is required for music filtering. Install it via `pip install demucs`."
            ) from exc

        resolved_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(resolved_device)
        self.apply_model = apply_model
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def filter_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Filter background music from a 1D or 2D (channels, time) waveform tensor."""
        import torchaudio.functional as F_audio

        orig_ndim = waveform.ndim
        if orig_ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Demucs expects stereo (2, T) or mono (1, T) -> repeat to stereo if mono
        channels = waveform.shape[0]
        stereo_wav = waveform.repeat(2, 1) if channels == 1 else waveform[:2]

        model_sr = int(getattr(self.model, "samplerate", 44100))
        if sample_rate != model_sr:
            input_tensor = F_audio.resample(stereo_wav, sample_rate, model_sr)
        else:
            input_tensor = stereo_wav

        input_batch = input_tensor.to(self.device).unsqueeze(0)  # (1, 2, T)

        with torch.inference_mode():
            sources = self.apply_model(
                self.model,
                input_batch,
                device=self.device,
                shifts=1,
                split=True,
                overlap=0.25,
            )

        # Demucs sources order: [drums, bass, other, vocals] (for htdemucs)
        # Vocals is the last source index (-1)
        vocals = sources[0, -1].detach().cpu()  # (2, T)
        if channels == 1:
            vocals = vocals.mean(dim=0, keepdim=True)  # (1, T)

        if sample_rate != model_sr:
            vocals = F_audio.resample(vocals, model_sr, sample_rate)

        # Match exact length of input
        target_len = waveform.shape[-1]
        if vocals.shape[-1] > target_len:
            vocals = vocals[..., :target_len]
        elif vocals.shape[-1] < target_len:
            vocals = torch.nn.functional.pad(vocals, (0, target_len - vocals.shape[-1]))

        return vocals if orig_ndim > 1 else vocals.squeeze(0)
