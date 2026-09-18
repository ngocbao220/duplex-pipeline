"""Purpose: Bridge vendored Sommelier artifacts to the common stereo contract.

Inputs: One mixture path, output directory and fixed Sommelier profile.
Outputs: Native Sommelier JSON/MP3 artifacts and one full-duration stereo WAV.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from core.orchestration.logging_style import get_logger, section  # noqa: E402


def run(source: Path, output: Path, config: dict):
    import re
    logger = get_logger("sommelier")
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or "hf_offline_local_token"
    _stage_sepreformer_checkpoint()
    native = output / f"native_{source.stem}"
    input_dir = native / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    copied = input_dir / source.name
    shutil.copy2(source, copied)

    # ===== Section header theo logging.md =====
    print(section("Sommelier"))
    logger.info("Input  : %s", source)
    logger.info("Output : %s", output)
    logger.info("")

    cfg = {
        "huggingface_token": token,
        "language": {"multilingual": False, "supported": ["en", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "ru", "ar", "hi"]},
        "entrypoint": {"input_folder_path": str(input_dir), "SAMPLE_RATE": int(config.get("sample_rate", 16000))},
        "separate": {"step1": {"model_path": "", "denoise": True, "margin": 44100, "chunks": 15, "n_fft": 6144, "dim_t": 8, "dim_f": 3072}},
    }
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "podcast_pipeline"
    env = os.environ.copy()
    if config.get("debug", False):
        env["SOMMELIER_LOG_LEVEL"] = "DEBUG"
    else:
        env["SOMMELIER_LOG_LEVEL"] = env.get("SOMMELIER_LOG_LEVEL", "INFO")

    sortformer_loc = os.environ.get("DUPLEX_MODEL_DIR", "local")
    sepreformer_loc = os.environ.get("SEPREFORMER", "local")

    with tempfile.TemporaryDirectory(prefix="sommelier-config-") as temporary:
        config_path = Path(temporary) / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        command = [sys.executable, str(vendor / "main_original_ASR_MoE.py"), "--input_folder_path", str(input_dir), "--config_path", str(config_path), "--sepreformer", "--no-demucs", "--no-ASRMoE", "--no-qwen3omni", "--until-pre-asr", "--expected-speakers", "2", "--LLM", "case_0", "--overlap_threshold", str(config.get("overlap_threshold", 1.0)), "--speaker-link-threshold", str(config.get("speaker_link_threshold", 0.75))]
        subprocess.run(command, cwd=vendor, env=env, check=True)
    manifests = sorted(input_dir.rglob(f"{source.stem}.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise RuntimeError("Sommelier completed without its JSON manifest")
    manifest_path = str(manifests[-1])

    stereo = _reconstruct_tracks(source, manifests[-1], output)

    # Log summary theo logging.md: Time/RTF mỗi bước
    try:
        manifest_data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        metadata = manifest_data.get("metadata", {})
        audio_dur = metadata.get("audio_duration_seconds", 0.0)
        t_dia = metadata.get("vad_sortformer", {}).get("processing_time_seconds", 0.0)
        t_sep = metadata.get("sepreformer_separation", {}).get("processing_time_seconds", 0.0)
        t_constrain = metadata.get("constrain_speakers", {}).get("processing_time_seconds", 0.0)
        t_pre = metadata.get("step0_preprocess", {}).get("processing_time_seconds", 0.0)
        total_time = t_pre + t_dia + t_sep + t_constrain

        # Count overlap segments
        segments = manifest_data.get("segments", [])
        overlap_count = 0
        for i in range(len(segments) - 1):
            if segments[i].get("end", 0) > segments[i + 1].get("start", 0):
                overlap_count += 1

        def _rtf_str(t):
            rtf_val = (t / audio_dur) if audio_dur > 0 else 0.0
            return f"Time: {t:.2f}s | RTF: {rtf_val:.4f}"

        logger.info("Overlap detection: Diarization by \"sortformer\" (load from %s)", sortformer_loc)
        logger.info("Found %d overlap part", overlap_count)
        if t_dia > 0:
            logger.info("%s", _rtf_str(t_dia))
        logger.info("")

        logger.info("Overlap separation by \"SepReformer\" (load from %s)", sepreformer_loc)
        if t_sep > 0:
            logger.info("%s", _rtf_str(t_sep))
        logger.info("")

        logger.info("Reconstruction")
        if t_constrain > 0:
            logger.info("%s", _rtf_str(t_constrain))
        if audio_dur > 0:
            speed_x = audio_dur / total_time if total_time > 0 else 0.0
            logger.info("Audio: %.2fs | Total: %.2fs | Speed: %.2fx RT", audio_dur, total_time, speed_x)
        logger.info("Done, result save to %s", output)

        table_lines = [
            "| Stage | Processing Time (s) | RTF |",
            "| :--- | :---: | :---: |",
            f"| Audio Duration | {audio_dur:.2f} | — |",
            f"| Overlap Detection (Sortformer) | {t_dia:.2f} | {(t_dia / audio_dur if audio_dur > 0 else 0.0):.4f} |",
            f"| Overlap Separation (SepReformer) | {t_sep:.2f} | {(t_sep / audio_dur if audio_dur > 0 else 0.0):.4f} |",
        ]
        if t_constrain > 0:
            table_lines.append(f"| Reconstruction | {t_constrain:.2f} | {(t_constrain / audio_dur if audio_dur > 0 else 0.0):.4f} |")
        total_rtf = total_time / audio_dur if audio_dur > 0 else 0.0
        table_lines.append(f"| **Total** | **{total_time:.2f}** | **{total_rtf:.4f}** |")
        speech_content = "# Sommelier Speed & Performance Summary\n\n" + "\n".join(table_lines) + "\n"
        (output / "speech.md").write_text(speech_content, encoding="utf-8")
        try:
            Path("speech.md").write_text(speech_content, encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass

    if not config.get("debug", False):
        shutil.rmtree(native, ignore_errors=True)
        return stereo, {}
    return stereo, {"native_manifest": manifest_path}


def _stage_sepreformer_checkpoint() -> None:
    """Expose the shared checkpoint through original Sommelier's fixed lookup path."""
    configured = os.environ.get("SEPREFORMER") or os.environ.get("VILIER_SEPREFORMER_CHECKPOINT", "")
    if not configured:
        raise RuntimeError("Set SEPREFORMER to a trusted SepReformer .pt/.pth file")
    checkpoint = Path(configured).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() not in {".pt", ".pth"}:
        raise RuntimeError(f"Invalid SEPREFORMER: {checkpoint}")
    weights_dir = Path(__file__).resolve().parents[2] / "vendor" / "SepReformer" / "models" / "SepReformer_Base_WSJ0" / "log" / "pretrain_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    link = weights_dir / checkpoint.name
    if link.exists() or link.is_symlink():
        if link.resolve() != checkpoint:
            raise RuntimeError(f"Sommelier checkpoint link already targets another file: {link}")
        return
    link.symlink_to(checkpoint)


def _reconstruct_tracks(source: Path, manifest_path: Path, output: Path) -> Path:
    import re
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rate = int(payload.get("metadata", {}).get("sample_rate", 16000))
    original = AudioSegment.from_file(source).set_channels(1).set_frame_rate(rate)
    total = len(original.get_array_of_samples())
    tracks = {}
    segment_dir = manifest_path.parent / source.stem
    for index, segment in enumerate(payload.get("segments", [])):
        speaker = str(segment.get("speaker", ""))
        if not speaker:
            continue
        path = segment_dir / f"{segment.get('index', f'{index:05d}')}_{speaker}.mp3"
        if not path.exists():
            continue
        audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(rate)
        samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        start = max(0, int(round(float(segment["start"]) * rate)))
        end = min(total, start + len(samples))
        track = tracks.setdefault(speaker, np.zeros(total, dtype=np.float32))
        track[start:end] += samples[: end - start]
    if len(tracks) != 2:
        raise ValueError(f"Sommelier must produce exactly two speakers; got {len(tracks)}")

    m = re.search(r"dialogue_(\d+)", source.stem)
    if m:
        stereo = output / f"stereo_{int(m.group(1))}.wav"
    else:
        stereo = output / "audio.stereo.wav"

    target_rate = 24000
    import torch
    stacked = np.stack([audio for _, audio in sorted(tracks.items())], axis=0)
    if rate != target_rate:
        tensor = torch.from_numpy(stacked)
        resampled = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=max(1, round(tensor.shape[-1] * target_rate / rate)),
            mode="linear", align_corners=False
        ).squeeze(0)
        out_samples = resampled.numpy().T.clip(-1.0, 1.0)
        out_rate = target_rate
    else:
        out_samples = stacked.T.clip(-1.0, 1.0)
        out_rate = rate

    sf.write(stereo, out_samples, out_rate, subtype="PCM_16")
    return stereo


