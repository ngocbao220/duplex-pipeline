"""Purpose: Run DuplexChat full-timeline speaker separation.

Inputs: Normalized audio, separator settings, device and chunk settings.
Outputs: Two separated waveforms and their sample rate.
"""
import threading

from core.orchestration.logging_style import get_logger

from .audio import load_wav_tensor
from .separation_backend import load_separation_models, run_separation
from .speaker_consistency import ANALYSIS_SAMPLE_RATE, enforce_speaker_consistency

LOGGER = get_logger("duplexchat")
_embedders: dict[str, object] = {}
_embedders_lock = threading.Lock()


def separate(audio_path, device, backend, model, steps, chunk, progress):
    models = load_separation_models(device=device, backend=backend, model_id=model)
    waveform, sample_rate = load_wav_tensor(audio_path)
    return separate_waveform(waveform, sample_rate, models, steps, chunk, progress)


def separate_waveform(waveform, sample_rate, models, steps, chunk, progress, speaker_consistency=True):
    """Run one already-loaded separator over a standardized waveform."""
    try:
        first, second, output_rate = run_separation(waveform, sample_rate, num_steps=steps, models=models, chunk_seconds=chunk, overlap_seconds=10.0, progress_callback=progress)
    finally:
        progress("close", 0)
    if not speaker_consistency:
        return first, second, output_rate
    extractor = _speaker_embedder(str(models["device"]))
    first, second, report = enforce_speaker_consistency(
        first, second, output_rate, lambda audio: extractor.extract(audio, ANALYSIS_SAMPLE_RATE))
    if report.swapped_blocks:
        LOGGER.info("Speaker consistency: swapped %d/%d blocks (%.1fs) back to their channel",
                    report.swapped_blocks, report.blocks, report.swapped_seconds)
    return first, second, output_rate


def _speaker_embedder(device: str):
    from .diarization_backend import SpeechBrainEmbeddingExtractor

    with _embedders_lock:
        if device not in _embedders:
            _embedders[device] = SpeechBrainEmbeddingExtractor(device)
        return _embedders[device]
