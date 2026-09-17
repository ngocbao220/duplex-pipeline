"""Whisper Language Identification (LID) for filtering Vietnamese dialogues.

Detects language probability for audio segments using Whisper/PhoWhisper models in offline mode.
"""
from __future__ import annotations

import logging
from pathlib import Path
import torch

from core.model_utils import (
    assert_local_model_exists,
    enforce_offline_mode,
    resolve_local_model_path,
)

LOGGER = logging.getLogger("duplexchat")
TARGET_SAMPLE_RATE = 16_000


def load_whisper_lid_model(
    model_name_or_path: str = "openai/whisper-small",
    device: str = "auto",
    enabled: bool = True,
) -> WhisperLIDFilter | NoOpLIDFilter:
    """Load Whisper model for Language Identification (LID)."""
    if not enabled:
        return NoOpLIDFilter()

    enforce_offline_mode()
    resolved_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    local_target, is_dir = resolve_local_model_path(
        model_name_or_path, env_var="WHISPER_MODEL_PATH", default_subpath="whisper-small"
    )

    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        target_str = str(local_target)
        if is_dir or Path(local_target).is_dir():
            assert_local_model_exists(
                local_target, required_files=["config.json"], model_name_hint="Whisper LID"
            )

        processor = WhisperProcessor.from_pretrained(target_str, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(target_str, local_files_only=True)
        model.to(torch.device(resolved_device))
        model.eval()

        return WhisperLIDFilter(model=model, processor=processor, device=resolved_device)

    except Exception as exc:
        LOGGER.warning("Could not load Transformers Whisper model '%s': %s. Trying openai-whisper package...", local_target, exc)
        try:
            import whisper

            target_str = str(local_target)
            model = whisper.load_model(target_str, device=resolved_device)
            return OpenAIWhisperLIDFilter(model=model, device=resolved_device)
        except Exception as exc2:
            raise RuntimeError(
                f"Failed to load Whisper LID model from '{local_target}'. "
                f"Ensure whisper model weights are present locally. Error: {exc2}"
            ) from exc2


class NoOpLIDFilter:
    """Fallback filter when LID filtering is disabled."""

    def is_vietnamese(self, waveform: torch.Tensor, sample_rate: int, min_prob: float = 0.5) -> tuple[bool, float]:
        return True, 1.0


class WhisperLIDFilter:
    """Filter dialogue audio using Hugging Face Transformers Whisper model."""

    def __init__(self, model, processor, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = torch.device(device)

    def is_vietnamese(self, waveform: torch.Tensor, sample_rate: int, min_prob: float = 0.5) -> tuple[bool, float]:
        """Detect Vietnamese language probability from waveform tensor."""
        import torchaudio.functional as F_audio

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = F_audio.resample(waveform.unsqueeze(0), sample_rate, TARGET_SAMPLE_RATE).squeeze(0)

        # Slice up to 30 seconds for language identification
        max_samples = 30 * TARGET_SAMPLE_RATE
        audio_np = waveform[:max_samples].detach().cpu().numpy()

        try:
            inputs = self.processor(audio_np, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
            with torch.inference_mode():
                logits = self.model(input_features, decoder_input_ids=decoder_input_ids).logits[0, -1]
                probs = torch.softmax(logits, dim=-1)

                vi_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vi|>")
                if vi_token_id is None or vi_token_id < 0:
                    vi_token_id = self.processor.tokenizer.lang_to_id.get("vi") or self.processor.tokenizer.lang_to_id.get("<|vi|>")

                vi_prob = float(probs[vi_token_id].cpu()) if vi_token_id is not None else 0.0

            return (vi_prob >= min_prob), vi_prob

        except Exception as exc:
            LOGGER.warning("Whisper LID detection failed (%s); allowing segment by default", exc)
            return True, 1.0


class OpenAIWhisperLIDFilter:
    """Filter dialogue audio using OpenAI whisper package."""

    def __init__(self, model, device: str) -> None:
        self.model = model
        self.device = torch.device(device)

    def is_vietnamese(self, waveform: torch.Tensor, sample_rate: int, min_prob: float = 0.5) -> tuple[bool, float]:
        import whisper
        import torchaudio.functional as F_audio

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = F_audio.resample(waveform.unsqueeze(0), sample_rate, TARGET_SAMPLE_RATE).squeeze(0)

        audio_np = waveform.detach().cpu().numpy()
        audio_padded = whisper.pad_or_trim(audio_np)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.device)

        with torch.inference_mode():
            _, probs = self.model.detect_language(mel)

        vi_prob = float(probs.get("vi", 0.0))
        return (vi_prob >= min_prob), vi_prob
