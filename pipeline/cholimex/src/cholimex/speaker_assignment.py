"""Purpose: Assign separated overlap candidates to Cholimex speakers.

Inputs: Reference and candidate waveforms plus embedding configuration.
Outputs: Speaker embeddings and deterministic candidate assignments.
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio

from .devices import cuda_device_index

from .models import Region

from pathlib import Path
from core.model_utils import enforce_offline_mode, resolve_local_model_path

MIN_EMBEDDING_SAMPLES = 8000


class EmbeddingExtractor(Protocol):
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...


class SpeechBrainEmbeddingExtractor:
    def __init__(self, model_id: str, device: str = "cpu") -> None:
        enforce_offline_mode()
        EncoderClassifier = _load_encoder_classifier()

        local_target, _ = resolve_local_model_path(
            model_id, env_var="SPEECHBRAIN_MODEL_PATH", default_subpath="spkrec-ecapa-voxceleb"
        )
        gpu_index = cuda_device_index(device)
        self.device = f"cuda:{gpu_index}" if gpu_index is not None else device
        self.sample_rate = 16000
        self.model = EncoderClassifier.from_hparams(
            source=str(local_target),
            savedir=str(local_target) if Path(local_target).is_dir() else None,
            run_opts={"device": self.device},
            local_files_only=True,
        )

    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        prepared = wav.detach().cpu().float()
        if prepared.ndim > 1:
            prepared = prepared.mean(dim=0)
        if sample_rate != self.sample_rate:
            prepared = F_audio.resample(prepared.unsqueeze(0), sample_rate, self.sample_rate).squeeze(0)
        if prepared.numel() == 0:
            prepared = torch.zeros(MIN_EMBEDDING_SAMPLES, dtype=torch.float32)
        elif prepared.shape[-1] < MIN_EMBEDDING_SAMPLES:
            prepared = F.pad(prepared, (0, MIN_EMBEDDING_SAMPLES - prepared.shape[-1]))
        with torch.inference_mode():
            emb = self.model.encode_batch(prepared.unsqueeze(0).to(self.device))
        return emb.detach().cpu().reshape(-1).float()


def build_reference_embeddings(
    original: torch.Tensor,
    sample_rate: int,
    regions: list[Region],
    extractor: EmbeddingExtractor,
    min_reference_duration: float,
) -> dict[int, torch.Tensor]:
    refs: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for region in regions:
        if region.type != "single_speaker" or region.speaker is None:
            continue
        if region.duration < min_reference_duration:
            continue
        start = int(round(region.start * sample_rate))
        end = int(round(region.end * sample_rate))
        if end <= start:
            continue
        refs[region.speaker].append(extractor.extract(original[:, start:end], sample_rate))
    return {
        speaker: torch.stack(embeddings).mean(dim=0)
        for speaker, embeddings in refs.items()
        if embeddings
    }


def assign_candidates(
    candidate_0: torch.Tensor,
    candidate_1: torch.Tensor,
    sample_rate: int,
    references: dict[int, torch.Tensor],
    extractor: EmbeddingExtractor | None,
    threshold: float,
    mode: str = "relative_similarity",
) -> tuple[dict[int, torch.Tensor], dict]:
    if extractor is None or set(references) != {0, 1}:
        raise RuntimeError("Cholimex overlap reconstruction requires embedding references for both speakers")
    if mode not in {"relative_similarity", "strict_threshold"}:
        raise ValueError("Cholimex speaker_assignment_mode must be relative_similarity or strict_threshold")

    emb_0 = extractor.extract(candidate_0, sample_rate)
    emb_1 = extractor.extract(candidate_1, sample_rate)

    scores = {
        "candidate_0": {"speaker_0": _score(emb_0, references[0]), "speaker_1": _score(emb_0, references[1])},
        "candidate_1": {"speaker_0": _score(emb_1, references[0]), "speaker_1": _score(emb_1, references[1])},
    }
    direct = (scores["candidate_0"]["speaker_0"] + scores["candidate_1"]["speaker_1"]) / 2.0
    swapped = (scores["candidate_0"]["speaker_1"] + scores["candidate_1"]["speaker_0"]) / 2.0
    best = max(direct, swapped)
    if mode == "strict_threshold" and best < threshold:
        raise RuntimeError(
            f"Cholimex overlap source assignment confidence {best:.3f} is below threshold {threshold:.3f}"
        )
    details = {
        "method": "relative_cosine", "mode": mode, "score": best,
        "direct_score": direct, "swapped_score": swapped,
        "margin": abs(direct - swapped), "candidate_scores": scores,
        "threshold": threshold if mode == "strict_threshold" else None,
    }
    if swapped > direct:
        return {0: candidate_1, 1: candidate_0}, {**details, "swapped": True, "mapping": {"candidate_0": 1, "candidate_1": 0}}
    return {0: candidate_0, 1: candidate_1}, {**details, "swapped": False, "mapping": {"candidate_0": 0, "candidate_1": 1}}


def _score(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.reshape(-1), right.reshape(-1), dim=0))


def _load_encoder_classifier():
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        return EncoderClassifier
    except ModuleNotFoundError as exc:
        if not _is_speechbrain_import_error(exc):
            raise

    try:
        from speechbrain.inference.classifiers import EncoderClassifier

        return EncoderClassifier
    except ModuleNotFoundError as exc:
        if not _is_speechbrain_import_error(exc):
            raise
        raise RuntimeError(
            "Cholimex overlap speaker assignment requires a compatible speechbrain install. "
            "Install it with `pip install speechbrain` or `pip install -r requirements.txt` "
            "in the active environment."
        ) from exc


def _is_speechbrain_import_error(exc: ModuleNotFoundError) -> bool:
    return exc.name is not None and (exc.name == "speechbrain" or exc.name.startswith("speechbrain."))
