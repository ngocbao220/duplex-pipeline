# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT
"""
Export and data conversion utilities for podcast pipeline.
Includes audio export, caption addition, and FlowSE denoising integration.
"""

import os
import time
import requests
import numpy as np
import soundfile as sf
from pydub import AudioSegment as PydubAudioSegment

# Logger will be initialized from main module
logger = None

# Constants
QWEN_3_OMNI_PORT = "10856"

def set_logger(log_instance):
    """Set logger instance from main module."""
    global logger
    logger = log_instance


import os
import time
import soundfile as sf
import requests

def add_qwen3omni_caption(filtered_list, audio, save_path, use_context=False):
    """
    Add Qwen3-Omni captions to each ASR segment.
    - Target for captioning is ALWAYS the current segment only.
    - Previous segments (up to 2) are provided as CONTEXT ONLY.
    """
    mode_str = "context-aware" if use_context else "standard"
    logger.info(f"Adding Qwen3-Omni captions ({mode_str} mode) to {len(filtered_list)} segments...")
    caption_start_time = time.time()

    url = f"http://localhost:{QWEN_3_OMNI_PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Strong guardrail: "caption only the target" + "output only the caption text"
    system_prompt = (
        "You are an audio captioning model.\n"
        "You may receive multiple audio clips.\n"
        "IMPORTANT:\n"
        "- Only the clip labeled [TARGET] must be captioned.\n"
        "- Clips labeled [CONTEXT] are for understanding only (e.g., sarcasm, intent). Do NOT caption them.\n"
        "- Output ONLY the caption text for [TARGET].\n"
        "- Do not add labels, numbering, quotes, or extra commentary.\n"
        "- Return a single caption in one paragraph."
    )

    def _build_content_list(context_paths, target_path):
        content = []
        content.append({
            "type": "text",
            "text": (
                "You will receive audio clips in order.\n"
                "Caption ONLY the one marked [TARGET].\n"
                "Do NOT caption any [CONTEXT] clips.\n"
                "Output only the target caption text."
            )
        })

        for i, p in enumerate(context_paths, start=1):
            content.append({"type": "text", "text": f"[CONTEXT {i}]"})
            content.append({"type": "audio_url", "audio_url": {"url": f"file://{p}"}})

        content.append({"type": "text", "text": "[TARGET]"})
        content.append({"type": "audio_url", "audio_url": {"url": f"file://{target_path}"}})
        return content

    for idx, segment in enumerate(filtered_list):
        temp_audio_paths = []
        try:
            context_paths = []

            # --- Prepare context audio (up to 2 segments) ---
            if use_context:
                start_c = max(0, idx - 2)
                for context_idx in range(start_c, idx):
                    context_segment = filtered_list[context_idx]
                    context_audio = _extract_segment_audio(context_segment, audio)
                    temp_path = os.path.join(save_path, f"temp_context_{idx:05d}_{context_idx:05d}.wav")
                    sf.write(temp_path, context_audio, audio["sample_rate"])
                    temp_audio_paths.append(temp_path)
                    context_paths.append(temp_path)

            # --- Prepare target (current) audio ---
            segment_audio = _extract_segment_audio(segment, audio)
            target_path = os.path.join(save_path, f"temp_target_{idx:05d}.wav")
            sf.write(target_path, segment_audio, audio["sample_rate"])
            temp_audio_paths.append(target_path)

            # --- Construct message ---
            content_list = _build_content_list(context_paths, target_path)

            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list},
                ],
                # Add options if needed (only when the server supports them)
                # "temperature": 0.2,
                # "max_tokens": 128,
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                caption = result.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                caption = caption.strip()

                # Safety measure: if the model violates the rules and outputs multiple lines/labels, use only the last paragraph (optional)
                # (Remove if not desired)
                if "\n" in caption:
                    caption = caption.split("\n")[-1].strip()

                segment["qwen3omni_caption"] = caption
                logger.debug(f"Segment {idx}: caption added (context={use_context and idx>0}, context_n={len(context_paths)})")
            else:
                logger.warning(f"Segment {idx}: API call failed with status {response.status_code} | {response.text[:200]}")
                segment["qwen3omni_caption"] = ""

        except Exception as e:
            logger.error(f"Segment {idx}: Error calling Qwen3-Omni API: {e}")
            segment["qwen3omni_caption"] = ""

        finally:
            # Clean up temporary files in all cases
            for p in temp_audio_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    caption_processing_time = time.time() - caption_start_time
    logger.info(f"Qwen3-Omni caption addition completed ({mode_str} mode)")
    return filtered_list, caption_processing_time



def _extract_segment_audio(segment, audio):
    """
    Extract audio from a segment.
    If enhanced_audio processed by SepReformer exists, use it preferentially.

    Args:
        segment (dict): Segment information
        audio (dict): Full audio information (including waveform, sample_rate)

    Returns:
        numpy.ndarray: Extracted audio data
    """
    # [CRITICAL] Use SepReformer-processed audio preferentially (to match the file that will be saved)
    if "enhanced_audio" in segment:
        return segment["enhanced_audio"]
    else:
        start_time = segment["start"]
        end_time = segment["end"]
        sample_rate = audio["sample_rate"]
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        return audio["waveform"][start_frame:end_frame]


def apply_flowse_denoising(filtered_list, audio, save_path, denoiser=None, use_asr_moe=False):
    """
    Apply FlowSE denoising to segments where sepreformer==True.

    Args:
        filtered_list (list): List of ASR result segments
        audio (dict): Audio dictionary (including waveform, sample_rate)
        save_path (str): Path to save temporary audio files
        denoiser: FlowSE denoiser object (skipped if None)
        use_asr_moe (bool): Whether to use ASR MoE (affects metadata key selection)

    Returns:
        tuple: (List of FlowSE-processed segments, processing time in seconds)
    """
    if denoiser is None:
        logger.info("FlowSE denoiser not provided, skipping denoising")
        return filtered_list, 0.0

    logger.info(f"Applying FlowSE denoising to sepreformer segments...")
    denoising_start_time = time.time()

    denoised_count = 0

    for idx, segment in enumerate(filtered_list):
        # Only process segments where sepreformer was applied
        sepreformer_key = "sepreformer" if use_asr_moe else "sepreformer"
        if not segment.get(sepreformer_key, False):
            continue

        # Check if enhanced_audio exists
        if "enhanced_audio" not in segment:
            logger.warning(f"Segment {idx}: sepreformer=True but no enhanced_audio found")
            continue

        temp_input_path = None
        temp_output_path = None

        try:
            # Extract segment audio
            segment_audio = segment["enhanced_audio"]
            sample_rate = audio["sample_rate"]

            # Save as temporary file (for FlowSE input)
            temp_input_path = os.path.join(save_path, f"temp_sepreformer_{idx:05d}.wav")
            sf.write(temp_input_path, segment_audio, sample_rate)

            # Get ASR text (FlowSE performs text-based denoising)
            text = segment.get("text", "")
            if not text:
                logger.warning(f"Segment {idx}: No text found, skipping FlowSE denoising")
                segment["flowse_denoised"] = False
            else:
                # Perform FlowSE denoising
                temp_output_path = os.path.join(save_path, f"temp_denoised_{idx:05d}.wav")
                denoiser.denoise(temp_input_path, text, temp_output_path)

                # Check if denoised audio was generated
                if os.path.exists(temp_output_path):
                    # Add denoising flag to metadata
                    segment["flowse_denoised"] = True
                    segment["denoised_audio_path"] = temp_output_path
                    denoised_count += 1
                    logger.debug(f"Segment {idx}: FlowSE denoising completed")
                else:
                    logger.warning(f"Segment {idx}: FlowSE output not found")
                    segment["flowse_denoised"] = False

        except Exception as e:
            logger.error(f"Segment {idx}: FlowSE denoising failed: {e}")
            segment["flowse_denoised"] = False

        finally:
            # Clean up temporary files (input file is always deleted)
            if temp_input_path and os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    denoising_end_time = time.time()
    denoising_processing_time = denoising_end_time - denoising_start_time

    logger.info(f"FlowSE denoising completed: {denoised_count}/{len(filtered_list)} segments processed")
    return filtered_list, denoising_processing_time


def export_segments_with_enhanced_audio(audio_info, segment_list, save_dir, audio_name):
    """
    Export segments to MP3 files.
    If 'enhanced_audio' exists in the segment (processed by SepReformer), use it.
    Otherwise, slice from the original audio.
    """
    # Create folder for saving segments
    segments_dir = os.path.join(save_dir, audio_name)
    os.makedirs(segments_dir, exist_ok=True)

    # Original full audio (Pydub object)
    full_audio_segment = audio_info.get("audio_segment")
    sample_rate = audio_info["sample_rate"]

    if full_audio_segment is None:
        # If audio_segment is not available, create from waveform (fallback)
        waveform_int16 = (audio_info["waveform"] * 32767).astype(np.int16)
        full_audio_segment = PydubAudioSegment(
            waveform_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

    logger.info(f"Exporting {len(segment_list)} segments with enhanced audio check...")

    for i, seg in enumerate(segment_list):
        # Generate filename (e.g., 00001_SPEAKER_01.mp3)
        idx_str = seg.get("index", f"{i:05d}")
        spk = seg.get("speaker", "Unknown")
        filename = f"{idx_str}_{spk}.mp3"
        file_path = os.path.join(segments_dir, filename)

        # 1. Check if FlowSE denoised audio exists (highest priority)
        if seg.get("flowse_denoised", False) and "denoised_audio_path" in seg:
            denoised_path = seg["denoised_audio_path"]
            if os.path.exists(denoised_path):
                # Load the denoised audio file
                denoised_waveform, denoised_sr = sf.read(denoised_path)

                # Convert float32 (-1.0 ~ 1.0) -> int16 range
                denoised_waveform = np.clip(denoised_waveform, -1.0, 1.0)
                wav_int16 = (denoised_waveform * 32767).astype(np.int16)

                target_segment = PydubAudioSegment(
                    wav_int16.tobytes(),
                    frame_rate=denoised_sr,
                    sample_width=2,
                    channels=1
                )
                logger.debug(f"Segment {idx_str}: Saved using FlowSE denoised output.")

                # Export as MP3 and then delete the temporary file
                target_segment.export(file_path, format="mp3")
                os.remove(denoised_path)
                logger.debug(f"Segment {idx_str}: Cleaned up temporary denoised file.")
                continue

        # 2. Check if 'enhanced_audio' processed by SepReformer exists
        elif seg.get("is_separated", False) and "enhanced_audio" in seg:
            # Convert Numpy array -> Pydub AudioSegment
            enhanced_waveform = seg["enhanced_audio"]

            # Convert float32 (-1.0 ~ 1.0) -> int16 range
            # Apply clip to prevent clipping artifacts
            enhanced_waveform = np.clip(enhanced_waveform, -1.0, 1.0)
            wav_int16 = (enhanced_waveform * 32767).astype(np.int16)

            target_segment = PydubAudioSegment(
                wav_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            # logger.debug(f"Segment {idx_str}: Saved using SepReformer output.")

        else:
            # 3. If not applied, extract from the original audio
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            target_segment = full_audio_segment[start_ms:end_ms]

        # Save as MP3
        target_segment.export(file_path, format="mp3")
