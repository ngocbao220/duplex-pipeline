# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT

"""
Background music detection and removal utilities for podcast pipeline.
Includes PANNs-based music detection and Demucs-based music removal.
"""

import os
import shutil
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import torch
from pydub import AudioSegment as PydubAudioSegment
from utils.logger import time_logger

# Logger will be initialized from main module
logger = None

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


class DemucsModel:
    """
    Wrapper class for Demucs model to load once and reuse for multiple segments.
    This avoids the overhead of subprocess calls and model loading for each segment.
    """
    def __init__(self, model_name="htdemucs", device=None):
        """
        Initialize Demucs model.

        Args:
            model_name (str): Name of the Demucs model to use (default: "htdemucs")
            device (str or torch.device): Device to use for inference (default: auto-detect)
        """
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            self.apply_model = apply_model

            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            if logger:
                logger.info(f"Loading Demucs model '{model_name}' on device: {self.device}")

            # Load the pretrained model
            self.model = get_model(model_name)
            self.model.to(self.device)
            self.model.eval()

            if logger:
                logger.info(f"Demucs model '{model_name}' loaded successfully")

        except ImportError as e:
            if logger:
                logger.error(f"Failed to import Demucs library: {e}")
            raise
        except Exception as e:
            if logger:
                logger.error(f"Failed to load Demucs model: {e}")
            raise

    def separate_vocals(self, audio_waveform, sample_rate):
        """
        Separate vocals from music using Demucs.

        Args:
            audio_waveform (np.ndarray): Input audio waveform (mono)
            sample_rate (int): Sample rate of the audio

        Returns:
            np.ndarray: Vocal-only waveform
        """
        try:
            # Demucs expects stereo input, so duplicate mono to stereo
            if audio_waveform.ndim == 1:
                stereo_waveform = np.stack([audio_waveform, audio_waveform], axis=0)
            else:
                stereo_waveform = audio_waveform

            # Convert to torch tensor
            waveform_tensor = torch.from_numpy(stereo_waveform).float().to(self.device)

            # Add batch dimension
            waveform_tensor = waveform_tensor.unsqueeze(0)

            # Apply model
            with torch.no_grad():
                sources = self.apply_model(self.model, waveform_tensor, device=self.device, shifts=1, split=True, overlap=0.25)

            # Extract vocals (usually index 3 for htdemucs: [drums, bass, other, vocals])
            # For htdemucs with --two-stems=vocals, we get [accompaniment, vocals]
            vocals = sources[0, -1]  # Last source is vocals

            # Convert back to numpy and take first channel (mono)
            vocals_np = vocals[0].cpu().numpy()

            return vocals_np.astype(np.float32)

        except Exception as e:
            if logger:
                logger.error(f"Error during Demucs vocal separation: {e}")
            return audio_waveform


@time_logger
def detect_background_music(audio, panns_model, threshold=0.3):
    """
    Detects background music using PANNs.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        panns_model (AudioTagging): Loaded PANNs model instance.
        threshold (float): Music probability threshold. If the probability is above this value, background music is considered present.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    logger.debug("Detecting background music using PANNs")

    # PANNs expects 32kHz audio, so resample if necessary
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if sample_rate != 32000:
        waveform_32k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = waveform

    # PANNs inference (model is already loaded)
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Get labels
    labels = panns_model.labels

    # Find Music probability
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        logger.info(f"Music probability: {music_prob:.3f}")
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        logger.warning("Music label not found in PANNs output")
        return False, 0.0


def detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3):
    """
    Detects background music in segment audio.

    Args:
        segment_audio (np.ndarray): Segment audio waveform.
        sample_rate (int): Sample rate.
        panns_model (AudioTagging): Loaded PANNs model instance.
        threshold (float): Music probability threshold.

    Returns:
        tuple: (has_music: bool, music_prob: float)
    """
    if panns_model is None:
        logger.warning("PANNs model is not loaded, skipping music detection")
        return False, 0.0

    # PANNs expects 32kHz audio, so resample if necessary
    if sample_rate != 32000:
        waveform_32k = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=32000)
    else:
        waveform_32k = segment_audio

    # Check minimum length required by the PANNs model (approximately 1 second = 32000 samples)
    # A minimum length is needed to pass through the pooling layers of the Cnn14 model
    min_length = 32000  # 1 second at 32kHz
    if len(waveform_32k) < min_length:
        logger.warning(f"Segment too short for music detection ({len(waveform_32k)/32000:.2f}s < 1.0s), skipping music detection")
        return False, 0.0

    # PANNs inference (model is already loaded)
    (clipwise_output, embedding) = panns_model.inference(waveform_32k[None, :])

    # Get labels
    labels = panns_model.labels

    # Find Music probability
    music_idx = labels.index('Music') if 'Music' in labels else None
    if music_idx is not None:
        music_prob = float(clipwise_output[0, music_idx])
        has_music = music_prob > threshold
        return has_music, music_prob
    else:
        return False, 0.0


def remove_segment_background_music_demucs(segment_audio, sample_rate, demucs_model=None):
    """
    Removes background music from segment audio using Demucs and extracts vocals only.

    Args:
        segment_audio (np.ndarray): Segment audio waveform.
        sample_rate (int): Sample rate.
        demucs_model (DemucsModel, optional): Loaded Demucs model instance. If None, falls back to subprocess method.

    Returns:
        np.ndarray: Vocal-only waveform, or the original waveform on failure.
    """
    # If demucs_model is provided, use it directly (much faster!)
    if demucs_model is not None:
        try:
            vocal_waveform = demucs_model.separate_vocals(segment_audio, sample_rate)
            return vocal_waveform
        except Exception as e:
            logger.error(f"Error during Demucs processing with model instance: {e}")
            return segment_audio

    # Fallback to subprocess method (legacy, slower)
    logger.warning("Using slow subprocess method for Demucs. Consider passing demucs_model instance.")

    # Create temporary directory for demucs output
    temp_dir = tempfile.mkdtemp(prefix="demucs_seg_")

    try:
        # Save segment audio to temporary file
        temp_input = os.path.join(temp_dir, "segment.wav")
        sf.write(temp_input, segment_audio, sample_rate)

        # Run demucs to separate vocals
        demucs_output_dir = os.path.join(temp_dir, "separated")

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Running Demucs on device: {device}")

        cmd = [
            "python", "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,  # Explicitly specify device (cuda or cpu)
            "-o", demucs_output_dir,
            temp_input
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Demucs failed for segment: {result.stderr}")
            return segment_audio

        # Load separated vocal track
        vocal_path = os.path.join(demucs_output_dir, "htdemucs", "segment", "vocals.wav")

        if not os.path.exists(vocal_path):
            logger.error(f"Vocal track not found at {vocal_path}")
            return segment_audio

        # Load vocal-only audio
        vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
        return vocal_waveform.astype(np.float32)

    except Exception as e:
        logger.error(f"Error during segment Demucs processing: {e}")
        return segment_audio
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@time_logger
def preprocess_segments_with_demucs(segment_list, audio, panns_model=None, use_demucs=False, demucs_model=None, padding=0.5):
    """
    Detects background music and applies Demucs per segment before ASR.
    (Adds padding to account for potential ASR timestamp shifts)

    Args:
        segment_list (list): List of segments
        audio (dict): Audio dictionary
        panns_model: PANNs model instance
        use_demucs (bool): Whether to use Demucs
        demucs_model (DemucsModel): Loaded Demucs model instance (if None, falls back to subprocess method)
        padding (float): Padding to add to segment boundaries (in seconds)

    Returns:
        tuple: (updated_audio, segment_demucs_flags)
    """
    if not use_demucs:
        logger.info("Demucs preprocessing skipped (flag disabled)")
        return audio, [False] * len(segment_list)

    logger.info(f"Preprocessing {len(segment_list)} segments with background music detection and removal (padding={padding}s)")

    waveform = audio["waveform"].copy()
    sample_rate = audio["sample_rate"]
    total_samples = len(waveform)
    segment_demucs_flags = []

    for idx, segment in enumerate(segment_list):
        # Apply padding to cover ASR boundary shifts
        start_time = max(0, segment["start"] - padding)
        end_time = segment["end"] + padding # Exceeding the boundary is handled automatically by slicing

        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)

        # Prevent exceeding the total length
        end_frame = min(end_frame, total_samples)

        segment_audio = waveform[start_frame:end_frame]

        # Skip if segment is too short
        if len(segment_audio) < 16000: # Less than 0.5 seconds
            segment_demucs_flags.append(False)
            continue

        # Detect background music (checking the region including padding)
        has_music, music_prob = detect_segment_background_music(segment_audio, sample_rate, panns_model, threshold=0.3)

        if has_music:
            logger.info(f"Segment {idx} (with padding): Background music detected (prob={music_prob:.3f}), applying Demucs...")
            # Apply demucs with model instance (much faster!)
            vocal_audio = remove_segment_background_music_demucs(segment_audio, sample_rate, demucs_model=demucs_model)

            # Replace the segment in the waveform
            # vocal_audio length may differ from segment_audio, so match the length
            target_length = len(segment_audio)
            if len(vocal_audio) >= target_length:
                waveform[start_frame:end_frame] = vocal_audio[:target_length]
            else:
                # Rare case where vocal_audio is shorter, pad accordingly
                waveform[start_frame : start_frame + len(vocal_audio)] = vocal_audio

            segment_demucs_flags.append(True)
            logger.info(f"Segment {idx}: Demucs applied successfully")
        else:
            segment_demucs_flags.append(False)

    # Update audio dictionary with processed waveform
    updated_audio = audio.copy()
    updated_audio["waveform"] = waveform

    # Also update audio_segment for export
    waveform_clipped = np.clip(waveform, -1.0, 1.0)
    waveform_int16 = (waveform_clipped * 32767).astype(np.int16)
    updated_audio_segment = PydubAudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    updated_audio["audio_segment"] = updated_audio_segment

    logger.info(f"Demucs preprocessing completed: {sum(segment_demucs_flags)}/{len(segment_list)} segments processed")
    return updated_audio, segment_demucs_flags
