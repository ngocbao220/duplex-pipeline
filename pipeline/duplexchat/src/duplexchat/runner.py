"""Purpose: Build flat DuplexChat stereo conversation WAVs from one episode.

Inputs: One source recording and DuplexChat diarization/separation settings.
Outputs: Numbered 24 kHz stereo WAVs, with optional phase-debug artifacts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import wave

import torch

from core.orchestration.logging_style import StepTimer, get_logger, section
from core.outputs import write_label_file

from .audio import load_wav_tensor
from .devices import resolve_device, validate_multi_gpu
from .diarization import diarize
from .dialogue import dialogue_filter_summary, extract_valid_dialogues
from .music import load_music_filter
from .preprocess import prepare_input, write_input_wav, write_wav
from .reconstruct import OUTPUT_SAMPLE_RATE, write_conversation_stereo
from .separation import separate_waveform
from .separation_backend import SAMPLE_RATE_IN, load_separation_models


def resolve_output_dir(output_prefix: str, output_dir: str | None = None) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path("data/processed") / output_prefix


def _no_progress(_event: str, _value: int) -> None:
    pass


def _fit_channel(channel: torch.Tensor, length: int) -> torch.Tensor:
    if channel.shape[-1] < length:
        return torch.nn.functional.pad(channel, (0, length - channel.shape[-1]))
    return channel[..., :length]


def _resample(waveform: torch.Tensor, input_rate: int, output_rate: int) -> torch.Tensor:
    if input_rate == output_rate:
        return waveform
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=max(1, round(waveform.shape[-1] * output_rate / input_rate)),
        mode="linear", align_corners=False,
    ).squeeze(0)


def _release(model: object) -> None:
    if isinstance(model, dict):
        for sub in model.values():
            _release(sub)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return

    if hasattr(model, "model") and hasattr(model.model, "to"):
        try:
            model.model.to(torch.device("cpu"))
        except Exception:
            pass
    elif hasattr(model, "pipeline") and hasattr(model.pipeline, "to"):
        try:
            model.pipeline.to(torch.device("cpu"))
        except Exception:
            pass
    elif hasattr(model, "to"):
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _devices(runtime_device: str, device_ids: list[list[int]] | list[int] | None) -> list[str]:
    ids = validate_multi_gpu(bool(device_ids), device_ids)
    return [f"cuda:{index}" for index in ids] if ids else [resolve_device(runtime_device, allow_cpu_fallback=True)]


def _device_label(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def _compact_path(value, width: int = 46) -> str:
    text = str(value)
    return text if len(text) <= width else "..." + text[-(width - 3):]


def _log_run_summary(logger, audio_path, normalized, output_root, phase_dir, separation_output, devices, conversations, segment_count, phase_times, audio_duration_sec: float = 0.0):
    total_time = sum(t for t in phase_times.values() if t is not None) or 1e-6
    speed_x = (audio_duration_sec / total_time) if (total_time > 0 and audio_duration_sec > 0) else 0.0
    rtf = (total_time / audio_duration_sec) if audio_duration_sec > 0 else 0.0
    mm, ss = divmod(int(audio_duration_sec), 60)
    time_fmt = f"{mm:02d}:{ss:02d}"

    stage_names = {
        "preprocess": "1. Preprocessing",
        "diarization": "2. Split Dialogue",
        "music_filter": "3. Remove music background",
        "separation": "2. DuplexChat Separation",
    }

    logger.info("")
    logger.info("Detected conversations: %d | Device: %s", conversations, _device_label(devices[0]))
    for key, elapsed in phase_times.items():
        label = stage_names.get(key, key.replace("_", " ").capitalize())
        if elapsed is not None:
            rtf_val = (elapsed / audio_duration_sec) if audio_duration_sec > 0 else 0.0
            logger.info("%s — Time: %.2fs | RTF: %.4f", label, elapsed, rtf_val)
        else:
            logger.info("%s — SKIPPED", label)
    if audio_duration_sec > 0:
        logger.info("Audio: %.2fs (%s) | Total: %.2fs | Speed: %.2fx RT", audio_duration_sec, time_fmt, total_time, speed_x)
    logger.info("===> Result saved to: %s", output_root)

    try:
        table_lines = [
            "| Stage | Processing Time (s) | RTF |",
            "| :--- | :---: | :---: |",
            f"| Audio Duration | {audio_duration_sec:.2f} | — |",
        ]
        for key, elapsed in phase_times.items():
            if elapsed is not None:
                label = stage_names.get(key, key.replace("_", " ").capitalize())
                rtf_val = (elapsed / audio_duration_sec) if audio_duration_sec > 0 else 0.0
                table_lines.append(f"| {label} | {elapsed:.2f} | {rtf_val:.4f} |")
        total_rtf = (total_time / audio_duration_sec) if audio_duration_sec > 0 else 0.0
        table_lines.append(f"| **Total** | **{total_time:.2f}** | **{total_rtf:.4f}** |")
        speech_content = "# DuplexChat Speed & Performance Summary\n\n" + "\n".join(table_lines) + "\n"
        (Path(output_root) / "speech.md").write_text(speech_content, encoding="utf-8")
        try:
            Path("speech.md").write_text(speech_content, encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass


def _separate_dialogues(waveform, sample_rate, dialogues, devices, num_steps, chunk_seconds, music_filter=None, phase_dir=None, debug=False, separation_model=None):
    models = {device: load_separation_models(device=device, model_id=separation_model) for device in devices}

    import queue
    device_queue = queue.Queue()
    for d in devices:
        device_queue.put(d)

    def run(pair):
        index, dialogue = pair
        start = max(0, min(waveform.shape[-1], round(dialogue.start * sample_rate)))
        end = max(start, min(waveform.shape[-1], round(dialogue.end * sample_rate)))
        crop = waveform[..., start:end].clone()

        if music_filter is not None:
            crop = music_filter.filter_waveform(crop, sample_rate)
            if debug and phase_dir is not None:
                music_debug_dir = phase_dir / "phase_02_music_filter"
                music_debug_dir.mkdir(parents=True, exist_ok=True)
                write_wav(music_debug_dir / f"dialogue_{index + 1}.wav", crop.squeeze(0).cpu().numpy(), sample_rate)

        device = device_queue.get()
        try:
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.set_device(torch.device(device))
            first, second, output_rate = separate_waveform(crop, sample_rate, models[device], num_steps, chunk_seconds, _no_progress)
            return index, dialogue, crop, first, second, output_rate
        finally:
            device_queue.put(device)

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            return sorted(pool.map(run, enumerate(dialogues)), key=lambda row: row[0])
    finally:
        for model in models.values():
            _release(model)


def run_single_audio(
    audio_path_str: str,
    diarize_chunk: float | str | None = None,
    separate_chunk: float = 120.0,
    separation_model: str | None = None,
    diarization_backend: str = "auto",
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    output_dir: str | None = None,
    output_prefix: str = "output",
    runtime_device: str = "auto",
    num_steps: int = 30,
    device_ids: list[int] | None = None,
    filter_music: bool = False,
    music_model: str = "htdemucs",
    debug: bool = False,
) -> dict:
    audio_path = Path(audio_path_str)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    devices = _devices(runtime_device, device_ids)
    logger = get_logger("duplexchat")
    output_root = Path(output_dir) if output_dir else resolve_output_dir(output_prefix)
    phase_dir = output_root / "debug" if debug else output_root / ".work"
    phase_dir.mkdir(parents=True, exist_ok=True)

    phase_times = {}
    if debug:
        write_input_wav(audio_path, phase_dir)

    # Estimate audio duration if audio_path is readable via wave/sf, or default to None until normalized
    audio_duration_sec = None
    try:
        with wave.open(str(audio_path), "rb") as stream:
            audio_duration_sec = stream.getnframes() / float(stream.getframerate())
    except Exception:
        pass

    with StepTimer(logger, "Step 0: Preprocess", duration_sec=audio_duration_sec) as timer:
        normalized = prepare_input(audio_path, phase_dir)
    phase_times["preprocess"] = timer.elapsed

    if audio_duration_sec is None:
        try:
            with wave.open(str(normalized), "rb") as stream:
                audio_duration_sec = stream.getnframes() / float(stream.getframerate())
        except Exception:
            pass

    logger.info("Device: diarization=%s | separation=%s", _device_label(devices[0]), [_device_label(d) for d in devices])
    with StepTimer(logger, "2. Split Dialogue", duration_sec=audio_duration_sec) as timer:
        diarizer, segments = diarize(normalized, phase_dir, diarization_model, diarization_backend, devices[0], diarize_chunk, _no_progress)
    phase_times["diarization"] = timer.elapsed
    _release(diarizer)
    if debug:
        write_label_file(
            phase_dir / "phase_02_diarization" / "speakers.txt",
            ((segment["start"], segment["end"], segment["speaker"]) for segment in segments),
        )

    summary = dialogue_filter_summary(segments)
    dialogues = extract_valid_dialogues(segments)
    logger.info("Found %d clips", len(dialogues))
    logger.info("Ignore: %d Imbalance | %d Short", summary.get("rejected_imbalanced", 0), summary.get("rejected_short", 0))
    logger.info("Detected conversations: %d", len(dialogues))
    if debug:
        write_label_file(
            phase_dir / "phase_02_diarization" / "conversation.txt",
            ((dialogue.start, dialogue.end, f"conversation_{index + 1}") for index, dialogue in enumerate(dialogues)),
        )
    waveform, sample_rate = load_wav_tensor(normalized)
    audio_duration_sec = float(waveform.shape[-1] / sample_rate)
    music_filter = load_music_filter(model_name=music_model, device=devices[0], enabled=filter_music) if filter_music else None
    rows = []
    if dialogues:
        with StepTimer(logger, "2. DuplexChat Separation", duration_sec=audio_duration_sec) as timer:
            logger.info("DialogueSidon | input=%d Hz -> output=%d Hz", SAMPLE_RATE_IN, OUTPUT_SAMPLE_RATE)
            separated = _separate_dialogues(
                waveform, sample_rate, dialogues, devices, num_steps, separate_chunk,
                music_filter=music_filter, phase_dir=phase_dir, debug=debug, separation_model=separation_model,
            )
        phase_times["separation"] = timer.elapsed
        for index, dialogue, crop, first, second, output_rate in separated:
            first, second, mixture = (_resample(first, output_rate, OUTPUT_SAMPLE_RATE), _resample(second, output_rate, OUTPUT_SAMPLE_RATE), _resample(crop, sample_rate, OUTPUT_SAMPLE_RATE))
            length = min(first.shape[-1], second.shape[-1], mixture.shape[-1])
            first, second, mixture = (_fit_channel(first, length), _fit_channel(second, length), _fit_channel(mixture, length))
            stereo = write_conversation_stereo(output_root, index, first, second, OUTPUT_SAMPLE_RATE, debug)
            rows.append(stereo)
    if not debug:
        shutil.rmtree(phase_dir)
    _log_run_summary(logger, audio_path, normalized, output_root, phase_dir, output_root, devices, len(rows), len(segments), phase_times, audio_duration_sec=audio_duration_sec)
    return {
        "stereo_files": rows,
        "devices": devices,
        "phase_times": phase_times,
        "audio_duration_sec": audio_duration_sec,
    }


def split_valid_dialogues(
    audio_path_str: str,
    output_dir: str,
    diarize_chunk: float | str | None = None,
    diarization_backend: str = "auto",
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    runtime_device: str = "auto",
    device_ids: list[int] | None = None,
    filter_music: bool = True,
    music_model: str = "htdemucs",
    filter_vietnamese: bool = True,
    lid_model: str = "openai/whisper-small",
    min_vi_prob: float = 0.5,
    debug: bool = False,
) -> dict:
    """Preprocesses audio, diarizes, extracts valid 2-speaker dialogues, applies music filtering, runs Whisper LID for Vietnamese filtering, and saves dialogue WAVs."""
    import json
    from .language_id import load_whisper_lid_model

    audio_path = Path(audio_path_str)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    devices = _devices(runtime_device, device_ids)
    logger = get_logger("duplexchat")
    output_root = Path(output_dir)
    phase_dir = output_root / "debug" if debug else output_root / ".work"
    phase_dir.mkdir(parents=True, exist_ok=True)

    phase_times = {}
    if debug:
        write_input_wav(audio_path, phase_dir)

    audio_duration_sec = None
    try:
        with wave.open(str(audio_path), "rb") as stream:
            audio_duration_sec = stream.getnframes() / float(stream.getframerate())
    except Exception:
        pass

    print(section("Dialogue Filtering"))
    logger.info("Dataset : %s", audio_path)
    logger.info("Output  : %s", output_root)
    logger.info("Device  : %s", _device_label(devices[0]))

    with StepTimer(logger, "1. Preprocessing", duration_sec=audio_duration_sec) as timer:
        normalized = prepare_input(audio_path, phase_dir)
    phase_times["preprocess"] = timer.elapsed

    if audio_duration_sec is None:
        try:
            with wave.open(str(normalized), "rb") as stream:
                audio_duration_sec = stream.getnframes() / float(stream.getframerate())
        except Exception:
            pass

    with StepTimer(logger, "2. Split Dialogue", duration_sec=audio_duration_sec) as timer:
        diarizer, segments = diarize(normalized, phase_dir, diarization_model, diarization_backend, devices[0], diarize_chunk, _no_progress)
    phase_times["diarization"] = timer.elapsed
    _release(diarizer)

    if debug:
        write_label_file(
            phase_dir / "phase_02_diarization" / "speakers.txt",
            ((segment["start"], segment["end"], segment["speaker"]) for segment in segments),
        )

    summary = dialogue_filter_summary(segments)
    dialogues = extract_valid_dialogues(segments)
    logger.info("+ Sample: %s", audio_path.name)
    logger.info("+ Duration: %.2fs", audio_duration_sec or 0.0)
    logger.info("+ Found %d clips", len(dialogues))
    logger.info("+ Ignore:")
    logger.info("++++ %d Imbalance (>80%% single speaker)", summary.get("rejected_imbalanced", 0))
    logger.info("++++ %d Short (<10s)", summary.get("rejected_short", 0))
    if summary.get("speakers", 0) < 2:
        logger.info("++++ %d Monologue (<2 distinct speakers detected by diarization)", 1 if summary.get("speakers", 0) <= 1 else 0)

    waveform, sample_rate = load_wav_tensor(normalized)
    audio_duration_sec = float(waveform.shape[-1] / sample_rate)
    music_filter = load_music_filter(model_name=music_model, device=devices[0], enabled=filter_music) if filter_music else None
    lid_filter = load_whisper_lid_model(model_name_or_path=lid_model, device=devices[0], enabled=filter_vietnamese) if filter_vietnamese else None

    output_root.mkdir(parents=True, exist_ok=True)
    dialogue_info = []
    candidate_dialogues = []
    output_dialogue_idx = 1
    lid_rejected_count = 0

    for index, dialogue in enumerate(dialogues):
        start = max(0, min(waveform.shape[-1], round(dialogue.start * sample_rate)))
        end = max(start, min(waveform.shape[-1], round(dialogue.end * sample_rate)))
        crop = waveform[..., start:end].clone()

        if music_filter is not None:
            crop = music_filter.filter_waveform(crop, sample_rate)

        vi_prob = 1.0
        if lid_filter is not None:
            is_vi, vi_prob = lid_filter.is_vietnamese(crop, sample_rate, min_prob=min_vi_prob)
            if not is_vi:
                lid_rejected_count += 1
                candidate_dialogues.append({
                    "candidate_index": index + 1,
                    "start": dialogue.start,
                    "end": dialogue.end,
                    "duration": dialogue.end - dialogue.start,
                    "vietnamese_probability": round(vi_prob, 4),
                    "reason": dialogue.reason,
                    "decision": "rejected_by_lid",
                })
                logger.info("++++ 1 Skipped by Whisper LID (Vietnamese probability %.3f < %.3f)", vi_prob, min_vi_prob)
                continue

        dialogue_filename = f"dialogue_{output_dialogue_idx}.wav"
        out_wav_path = output_root / dialogue_filename
        write_wav(out_wav_path, crop.squeeze(0).cpu().numpy(), sample_rate)

        dialogue_info.append({
            "index": output_dialogue_idx,
            "filename": dialogue_filename,
            "path": str(out_wav_path),
            "start": dialogue.start,
            "end": dialogue.end,
            "duration": dialogue.end - dialogue.start,
            "vietnamese_probability": round(vi_prob, 4),
            "reason": dialogue.reason,
        })
        candidate_dialogues.append({
            "candidate_index": index + 1,
            "start": dialogue.start,
            "end": dialogue.end,
            "duration": dialogue.end - dialogue.start,
            "vietnamese_probability": round(vi_prob, 4),
            "reason": dialogue.reason,
            "decision": "exported",
            "filename": dialogue_filename,
        })
        output_dialogue_idx += 1

    skip_reason = None
    if not dialogue_info:
        if summary.get("speakers", 0) < 2:
            skip_reason = f"Diarization detected only {summary.get('speakers', 0)} speaker (Monologue - minimum 2 speakers required)"
        elif len(dialogues) == 0:
            reasons = []
            if summary.get("rejected_imbalanced", 0) > 0:
                reasons.append(f"{summary['rejected_imbalanced']} clips rejected due to speaker imbalance (>80%)")
            if summary.get("rejected_short", 0) > 0:
                reasons.append(f"{summary['rejected_short']} clips rejected for being too short (<10s)")
            skip_reason = "; ".join(reasons) if reasons else "No two-speaker dialogue segments found"
        elif lid_rejected_count > 0:
            skip_reason = f"All {lid_rejected_count} candidate clips rejected by Whisper LID (language probability < {min_vi_prob})"
        else:
            skip_reason = "No candidate clips passed filtering criteria"

        logger.warning("===> [SKIPPED] Audio '%s' produced 0 valid dialogue clips!", audio_path.name)
        logger.warning("     Reason: %s", skip_reason)
    else:
        logger.info("===> Collect %d dialogue from \"%s\", save to \"%s\"", len(dialogue_info), audio_path, output_root)

    manifest = {
        "source_audio": str(audio_path),
        "audio_duration_sec": audio_duration_sec,
        "dialogue_count": len(dialogue_info),
        "skip_reason": skip_reason,
        "filter_music": filter_music,
        "music_model": music_model if filter_music else None,
        "filter_vietnamese": filter_vietnamese,
        "lid_model": lid_model if filter_vietnamese else None,
        "min_vi_prob": min_vi_prob if filter_vietnamese else None,
        "filter_summary": {
            "candidate_dialogue_count": len(dialogues),
            "exported_dialogue_count": len(dialogue_info),
            "rejected_by_lid": lid_rejected_count,
            "rejected_short": summary.get("rejected_short", 0),
            "rejected_imbalanced": summary.get("rejected_imbalanced", 0),
            "diarized_speaker_count": summary.get("speakers", 0),
            "lid": {
                "enabled": filter_vietnamese,
                "model": lid_model if filter_vietnamese else None,
                "min_vi_probability": min_vi_prob if filter_vietnamese else None,
            },
        },
        "candidate_dialogues": candidate_dialogues,
        "dialogues": dialogue_info,
        "phase_times": phase_times,
    }

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not debug and phase_dir.exists():
        shutil.rmtree(phase_dir, ignore_errors=True)

    return manifest


def separate_dialogue_files(
    input_path_str: str,
    output_dir: str,
    runtime_device: str = "auto",
    device_ids: list[int] | None = None,
    num_steps: int = 30,
    separate_chunk: float = 120.0,
    separation_model: str | None = None,
    debug: bool = False,
) -> dict:
    """Takes input dialogue directory (or WAV file) and runs DialogueSidon separation model, generating 24kHz stereo outputs."""
    import time
    from tqdm import tqdm

    logger = get_logger("duplexchat")
    input_path = Path(input_path_str)
    def _dialogue_key(p: Path) -> int:
        m = re.search(r"dialogue_(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    if input_path.is_dir():
        dialogue_files = sorted(list(input_path.glob("dialogue_*.wav")), key=_dialogue_key)
    elif input_path.is_file():
        dialogue_files = [input_path]
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path_str}")

    devices = _devices(runtime_device, device_ids)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not dialogue_files:
        logger.warning("No dialogue files (dialogue_*.wav) found in '%s' (0 clips from dialogue filtering). Skipping DuplexChat separation.", input_path.name)
        return {
            "stereo_files": [],
            "devices": devices,
            "dialogue_count": 0,
        }

    total_input_dur = 0.0
    for df in dialogue_files:
        try:
            with wave.open(str(df), "rb") as wf:
                total_input_dur += wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass

    # ===== Section header theo logging.md =====
    print(section("DuplexChat"))
    model_loc = separation_model or os.environ.get("DUPLEX_MODEL_DIR", "local")
    logger.info("Separation by \"DialogueSidon\" (load from %s)", model_loc)
    logger.info("+ Input duration: %.2fs", total_input_dur)
    logger.info("+ Output sample rate: 24khz")
    logger.info("")

    models = {device: load_separation_models(device=device, model_id=separation_model) for device in devices}

    rows = []
    total_audio_sec = 0.0
    t0 = time.perf_counter()

    try:
        pbar = tqdm(dialogue_files, desc="Separating dialogues", unit="clip")
        for index, dialogue_wav in enumerate(pbar):
            crop, sample_rate = load_wav_tensor(dialogue_wav)
            clip_dur = float(crop.shape[-1] / sample_rate)
            total_audio_sec += clip_dur

            device = devices[index % len(devices)]
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.set_device(torch.device(device))

            first, second, output_rate = separate_waveform(
                crop, sample_rate, models[device], num_steps, separate_chunk, _no_progress
            )

            first = _resample(first, output_rate, OUTPUT_SAMPLE_RATE)
            second = _resample(second, output_rate, OUTPUT_SAMPLE_RATE)
            mixture = _resample(crop, sample_rate, OUTPUT_SAMPLE_RATE)
            length = min(first.shape[-1], second.shape[-1], mixture.shape[-1])
            first = _fit_channel(first, length)
            second = _fit_channel(second, length)
            mixture = _fit_channel(mixture, length)

            # Preserve dialogue index in stereo file name (e.g. dialogue_10 -> stereo_10)
            m = re.search(r"dialogue_(\d+)", dialogue_wav.stem)
            conv_index = int(m.group(1)) - 1 if m else index
            stereo = write_conversation_stereo(output_root, conv_index, first, second, OUTPUT_SAMPLE_RATE, debug)
            rows.append(stereo)
    finally:
        for model in models.values():
            _release(model)

    elapsed = time.perf_counter() - t0
    rtf = (elapsed / total_audio_sec) if total_audio_sec > 0 else 0.0
    logger.info("Time: %.2fs | RTF: %.4f", elapsed, rtf)
    logger.info("Done, result save to %s", output_root)

    try:
        speed_x = (total_audio_sec / elapsed) if (elapsed > 0 and total_audio_sec > 0) else 0.0
        table_lines = [
            "| Stage | Processing Time (s) | RTF |",
            "| :--- | :---: | :---: |",
            f"| Total Dialogue Audio | {total_audio_sec:.2f} | — |",
            f"| DialogueSidon Separation | {elapsed:.2f} | {rtf:.4f} |",
            f"| **Total** | **{elapsed:.2f}** | **{rtf:.4f}** |",
        ]
        speech_content = f"# DuplexChat Speed & Performance Summary\n\nDialogues: {len(rows)} | Speed: {speed_x:.2f}x RT\n\n" + "\n".join(table_lines) + "\n"
        (output_root / "speech.md").write_text(speech_content, encoding="utf-8")
    except Exception:
        pass

    return {
        "stereo_files": rows,
        "devices": devices,
        "dialogue_count": len(dialogue_files),
        "total_audio_sec": total_audio_sec,
        "elapsed_sec": elapsed,
        "rtf": rtf,
    }

