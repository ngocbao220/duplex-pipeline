"""Purpose: Persist DuplexChat separated conversation tracks.

Inputs: Two separated waveforms, output naming and model metadata.
Outputs: Flat production stereo WAVs or per-conversation debug WAVs.
"""
from pathlib import Path

import torch

from core.outputs import save_stereo_wav, save_wav


OUTPUT_SAMPLE_RATE = 24_000
def _to_output_rate(waveform, sample_rate):
    if sample_rate == OUTPUT_SAMPLE_RATE:
        return waveform
    output_length = max(1, round(waveform.shape[-1] * OUTPUT_SAMPLE_RATE / sample_rate))
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=output_length, mode="linear", align_corners=False,
    ).squeeze(0)


def write_conversation_stereo(
    output_dir,
    conversation_index,
    first_channel,
    second_channel,
    sample_rate,
    debug,
):
    """Write one conversation as flat production output or debug source tracks."""
    output_dir = Path(output_dir)
    number = conversation_index + 1
    first_channel = _to_output_rate(first_channel, sample_rate)
    second_channel = _to_output_rate(second_channel, sample_rate)
    if debug:
        conversation_dir = output_dir / "debug" / "phase_03_separation" / f"conversation_{number}"
        save_wav(conversation_dir / f"left_{number}.wav", first_channel, OUTPUT_SAMPLE_RATE)
        save_wav(conversation_dir / f"right_{number}.wav", second_channel, OUTPUT_SAMPLE_RATE)
        stereo = conversation_dir / f"stereo_{number}.wav"
    else:
        stereo = output_dir / f"stereo_{number}.wav"
    save_stereo_wav(stereo, first_channel, second_channel, OUTPUT_SAMPLE_RATE)
    return stereo
