"""Configuration shared by orchestration and the Cholimex runtime."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    runtime_device: str = "auto"
    allow_cpu_fallback: bool = True
    separation_num_steps: int = 30
    benchmark_output_dir: Path = Path("reports")
    cholimex_backchannel_max_duration: float = 1.0
    cholimex_min_vad_duration: float = 0.0
    cholimex_vad_onset: float | None = None
    cholimex_vad_offset: float | None = None
    cholimex_vad_padding_ms: int = 30
    cholimex_merge_gap: float = 0.0
    cholimex_min_reference_duration: float = 2.0
    cholimex_speaker_assignment_mode: str = "relative_similarity"
    cholimex_cosine_similarity_threshold: float = 0.5
    cholimex_overlap_padding: float = 0.10
    cholimex_proposal_backend: str = "dialoguesidon"
    cholimex_proposal_model: str | None = "sarulab-speech/DialogueSidon"
    cholimex_overlap_separator_backend: str = "dialoguesidon"
    cholimex_overlap_separator_model: str | None = "sarulab-speech/DialogueSidon"
    cholimex_speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"


FIELD_ALIASES = {
    "runtime.device": "runtime_device",
    "runtime.allow_cpu_fallback": "allow_cpu_fallback",
    "separation.num_steps": "separation_num_steps",
    "benchmark.output_dir": "benchmark_output_dir",
    "cholimex.backchannel_max_duration": "cholimex_backchannel_max_duration",
    "cholimex.min_vad_duration": "cholimex_min_vad_duration",
    "cholimex.vad_onset": "cholimex_vad_onset",
    "cholimex.vad_offset": "cholimex_vad_offset",
    "cholimex.vad_padding_ms": "cholimex_vad_padding_ms",
    "cholimex.merge_gap": "cholimex_merge_gap",
    "cholimex.min_reference_duration": "cholimex_min_reference_duration",
    "cholimex.speaker_assignment_mode": "cholimex_speaker_assignment_mode",
    "cholimex.cosine_similarity_threshold": "cholimex_cosine_similarity_threshold",
    "cholimex.overlap_padding": "cholimex_overlap_padding",
    "cholimex.proposal_backend": "cholimex_proposal_backend",
    "cholimex.proposal_model": "cholimex_proposal_model",
    "cholimex.overlap_separator_backend": "cholimex_overlap_separator_backend",
    "cholimex.overlap_separator_model": "cholimex_overlap_separator_model",
    "cholimex.speaker_embedding_model": "cholimex_speaker_embedding_model",
}
PATH_FIELDS = {"benchmark_output_dir"}


def _flatten_mapping(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, full_key))
        else:
            flat[full_key] = value
    return flat


def apply_config_data(cfg: Config, data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise TypeError("config.json must contain a JSON object")
    for config_key, value in _flatten_mapping(data).items():
        field_name = FIELD_ALIASES.get(config_key)
        if field_name is None:
            raise ValueError(f"unknown config key: {config_key}")
        setattr(cfg, field_name, Path(value) if field_name in PATH_FIELDS and value is not None else value)
    return cfg


def load_config(path: Path = Path("configs/config.json")) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        return apply_config_data(Config(), json.load(handle))
