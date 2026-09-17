"""Purpose: Execute the complete Cholimex separation pipeline.

Inputs: One mixture audio path, output directory, and Cholimex configuration.
Outputs: One stereo WAV, run manifest, and optional debug phase artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio.functional as F_audio

from core import outputs
from core.config import Config
from core.orchestration.logging_style import StepTimer, get_logger
from . import preprocess as audio
from . import separation as separate
from .devices import resolve_device

from .reconstruct import reconstruct_tracks
from .regions import classify_regions
from .speaker_assignment import SpeechBrainEmbeddingExtractor, build_reference_embeddings
from .vad import run_silero_vad, write_vad_artifacts


LOGGER = get_logger("cholimex")


def _make_progress(desc: str, unit: str = "chunk"):
    def _callback(event: str, value: int) -> None:
        return None

    return _callback


def run_cholimex_file(input_path: Path, output_dir: Path, cfg: Config) -> dict:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Cholimex input audio file does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    original_path = debug_dir / "original.wav"
    audio.ensure_ffmpeg()
    audio.transcode_to_wav_16k_mono(input_path, original_path)
    original, sample_rate = audio.load_wav_tensor(original_path)
    duration_sec = original.shape[-1] / sample_rate
    device = resolve_device(cfg.runtime_device, cfg.allow_cpu_fallback)

    with StepTimer(LOGGER, "Step 1: Proposal separation", duration_sec=duration_sec,
                   details=f"backend={cfg.cholimex_proposal_backend} device={device}"):
        proposal_models = separate.load_separation_models(device, cfg.cholimex_proposal_backend, cfg.cholimex_proposal_model)
        sidon_0, sidon_1, sidon_sr = separate.run_separation(
            original, sample_rate, cfg.separation_num_steps, proposal_models,
            progress_callback=_make_progress(f"cholimex proposal {input_path.stem}"))
    sidon_0 = _align_proposal_track(sidon_0, sidon_sr, sample_rate, original.shape[-1], "sidon_track_0")
    sidon_1 = _align_proposal_track(sidon_1, sidon_sr, sample_rate, original.shape[-1], "sidon_track_1")
    sidon_sr = sample_rate
    outputs.save_wav(debug_dir / "sidon_track_0.wav", sidon_0, sidon_sr)
    outputs.save_wav(debug_dir / "sidon_track_1.wav", sidon_1, sidon_sr)

    LOGGER.info("Step 2: VAD masking on provisional tracks")
    mask_0 = run_silero_vad(
        sidon_0,
        sidon_sr,
        speaker=0,
        threshold=cfg.cholimex_vad_onset,
        offset=cfg.cholimex_vad_offset,
        padding_ms=cfg.cholimex_vad_padding_ms,
        min_duration=cfg.cholimex_min_vad_duration,
        merge_gap=cfg.cholimex_merge_gap,
    )
    mask_1 = run_silero_vad(
        sidon_1,
        sidon_sr,
        speaker=1,
        threshold=cfg.cholimex_vad_onset,
        offset=cfg.cholimex_vad_offset,
        padding_ms=cfg.cholimex_vad_padding_ms,
        min_duration=cfg.cholimex_min_vad_duration,
        merge_gap=cfg.cholimex_merge_gap,
    )
    write_vad_artifacts(debug_dir / "vad_track_0.json", debug_dir / "vad_track_0.txt", mask_0)
    write_vad_artifacts(debug_dir / "vad_track_1.json", debug_dir / "vad_track_1.txt", mask_1)

    LOGGER.info("Step 3: Region classification")
    regions = classify_regions(
        mask_0,
        mask_1,
        duration_sec=duration_sec,
        backchannel_max_duration=cfg.cholimex_backchannel_max_duration,
    )
    outputs.write_json(debug_dir / "regions.json", [region.to_dict() for region in regions])

    overlap_regions = [region for region in regions if region.type in {"overlap", "overlap_backchannel"}]
    embedder = None
    references = {}
    reference_audio = {}
    if overlap_regions:
        LOGGER.info("Step 4: Speaker references and overlap separation")
        embedder = SpeechBrainEmbeddingExtractor(cfg.cholimex_speaker_embedding_model, device=device)
        references = build_reference_embeddings(
            original,
            sample_rate,
            regions,
            embedder,
            cfg.cholimex_min_reference_duration,
        )
        reference_audio = _write_reference_audio(debug_dir, original, sample_rate, regions, cfg.cholimex_min_reference_duration)

    overlap_models = proposal_models
    if (
        overlap_regions
        and (
            cfg.cholimex_overlap_separator_backend != cfg.cholimex_proposal_backend
            or cfg.cholimex_overlap_separator_model != cfg.cholimex_proposal_model
        )
    ):
        overlap_models = separate.load_separation_models(
            device,
            cfg.cholimex_overlap_separator_backend,
            cfg.cholimex_overlap_separator_model,
        )

    def _separator(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return separate.run_separation(
            wav,
            sr,
            cfg.separation_num_steps,
            overlap_models,
            progress_callback=_make_progress(f"cholimex overlap {input_path.stem}"),
        )

    def _write_overlap_debug(record: dict, mixture: torch.Tensor, candidate_0: torch.Tensor,
                             candidate_1: torch.Tensor, speaker_0: torch.Tensor, speaker_1: torch.Tensor, sr: int) -> None:
        directory = debug_dir / "overlaps" / record["id"]
        directory.mkdir(parents=True, exist_ok=True)
        outputs.save_wav(directory / "mixture.wav", mixture, sr)
        outputs.save_wav(directory / "candidate_0.wav", candidate_0, sr)
        outputs.save_wav(directory / "candidate_1.wav", candidate_1, sr)
        outputs.save_stereo_wav(directory / "audio.stereo.wav", speaker_0, speaker_1, sr)
        record["reference_audio"] = reference_audio
        outputs.write_json(directory / "metadata.json", record)
        assignment = record["assignment"]
        LOGGER.info("Overlap %s assignment mode=%s mapping=%s direct=%.3f swapped=%.3f margin=%.3f",
                    record["id"], assignment["mode"], assignment["mapping"], assignment["direct_score"],
                    assignment["swapped_score"], assignment["margin"])

    with StepTimer(LOGGER, "Step 5: Reconstruction", duration_sec=duration_sec,
                   details=f"overlap_regions={len(overlap_regions)}"):
        final_0, final_1, overlap_records = reconstruct_tracks(
            original, sample_rate, regions, _separator if overlap_regions else None, references, embedder,
            cfg.cholimex_cosine_similarity_threshold, cfg.cholimex_speaker_assignment_mode,
            cfg.cholimex_overlap_padding,
            debug_callback=_write_overlap_debug if overlap_regions else None)
    outputs.write_json(debug_dir / "overlaps.json", overlap_records)
    stereo_path = output_dir / "audio.stereo.wav"
    outputs.save_stereo_wav(stereo_path, final_0, final_1, sample_rate)
    outputs.save_wav(debug_dir / "final_track_0.wav", final_0, sample_rate)
    outputs.save_wav(debug_dir / "final_track_1.wav", final_1, sample_rate)
    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "stereo": str(stereo_path),
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "regions": len(regions),
        "overlap_regions": len(overlap_records),
        "models": {
            "proposal": {
                "backend": cfg.cholimex_proposal_backend,
                "model": cfg.cholimex_proposal_model,
            },
            "overlap_separator": {
                "backend": cfg.cholimex_overlap_separator_backend,
                "model": cfg.cholimex_overlap_separator_model,
            },
            "speaker_embedding": cfg.cholimex_speaker_embedding_model,
        },
        "speaker_assignment": {
            "mode": cfg.cholimex_speaker_assignment_mode,
            "cosine_similarity_threshold": cfg.cholimex_cosine_similarity_threshold
            if cfg.cholimex_speaker_assignment_mode == "strict_threshold" else None,
        },
        "debug_dir": str(debug_dir),
    }
    (output_dir / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_reference_audio(
    debug_dir: Path,
    original: torch.Tensor,
    sample_rate: int,
    regions: list,
    min_reference_duration: float,
) -> dict[str, str]:
    paths = {}
    for speaker in [0, 1]:
        parts = []
        for region in regions:
            if region.type == "single_speaker" and region.speaker == speaker and region.duration >= min_reference_duration:
                start = int(round(region.start * sample_rate))
                end = int(round(region.end * sample_rate))
                parts.append(original[:, start:end])
        if parts:
            path = debug_dir / f"speaker_reference_{speaker}.wav"
            outputs.save_wav(path, torch.cat(parts, dim=-1), sample_rate)
            paths[f"speaker_{speaker}"] = str(path)
    return paths


def _align_proposal_track(
    wav: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
    target_samples: int,
    label: str,
) -> torch.Tensor:
    aligned = wav.detach().cpu().float()
    if aligned.ndim == 1:
        aligned = aligned.unsqueeze(0)
    if aligned.shape[0] > 1:
        aligned = aligned.mean(dim=0, keepdim=True)
    if source_sample_rate != target_sample_rate:
        LOGGER.warning(
            "Resampling %s from %s Hz to %s Hz for timeline alignment",
            label,
            source_sample_rate,
            target_sample_rate,
        )
        aligned = F_audio.resample(aligned, source_sample_rate, target_sample_rate)
    diff = aligned.shape[-1] - target_samples
    if diff > 0:
        LOGGER.warning("Cropping %s by %d samples to match original timeline", label, diff)
        return aligned[:, :target_samples]
    if diff < 0:
        LOGGER.warning("Padding %s by %d samples to match original timeline", label, -diff)
        return torch.nn.functional.pad(aligned, (0, -diff))
    return aligned
