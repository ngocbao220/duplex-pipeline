"""Purpose: Resolve DuplexChat diarization and separation model options.

Inputs: Backend aliases and user-selected model identifiers.
Outputs: Concrete backend and model settings.
"""
from __future__ import annotations

DIARIZATION_MODELS = {
    "sortformer": "nvidia/diar_sortformer_4spk-v1",
    "sortformer-streaming-v2.1": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "pyannote-3.1": "pyannote/speaker-diarization-3.1",
    "diarizen": "BUT-FIT/diarizen-wavlm-large-s80-md",
}

SEPARATION_MODELS = {
    "dialoguesidon": "sarulab-speech/DialogueSidon",
}


def resolve_model_alias(model: str | None, aliases: dict[str, str]) -> str | None:
    if model is None:
        return None
    return aliases.get(model.strip().lower(), model)


def infer_diarization_backend(model: str) -> str:
    model_norm = model.strip().lower()
    if model_norm.startswith(("nvidia/diar_sortformer", "nvidia/diar_streaming_sortformer")):
        return "sortformer"
    if model_norm.startswith("but-fit/diarizen"):
        return "diarizen"
    return "pyannote"
