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
    cholimex_crossfade_ms: float = 40.0
    cholimex_gate_fade_ms: float = 20.0
    cholimex_gain_match: bool = True
    cholimex_gain_context_s: float = 1.0
    cholimex_gain_max: float = 2.0


FIELD_ALIASES = {
    "runtime.device": "runtime_device",
    "runtime_device": "runtime_device",
    "device": "runtime_device",
    "runtime.allow_cpu_fallback": "allow_cpu_fallback",
    "allow_cpu_fallback": "allow_cpu_fallback",
    "separation.num_steps": "separation_num_steps",
    "pipeline.separation.num_steps": "separation_num_steps",
    "separation_num_steps": "separation_num_steps",
    "num_steps": "separation_num_steps",
    "benchmark.output_dir": "benchmark_output_dir",
    "benchmark_output_dir": "benchmark_output_dir",
}

CHOLIMEX_KEYS = [
    "backchannel_max_duration",
    "min_vad_duration",
    "vad_onset",
    "vad_offset",
    "vad_padding_ms",
    "merge_gap",
    "min_reference_duration",
    "speaker_assignment_mode",
    "cosine_similarity_threshold",
    "overlap_padding",
    "proposal_backend",
    "proposal_model",
    "overlap_separator_backend",
    "overlap_separator_model",
    "speaker_embedding_model",
    "crossfade_ms",
    "gate_fade_ms",
    "gain_match",
    "gain_context_s",
    "gain_max",
]

for _k in CHOLIMEX_KEYS:
    FIELD_ALIASES[_k] = f"cholimex_{_k}"
    FIELD_ALIASES[f"cholimex.{_k}"] = f"cholimex_{_k}"
    FIELD_ALIASES[f"pipeline.{_k}"] = f"cholimex_{_k}"
    FIELD_ALIASES[f"cholimex_{_k}"] = f"cholimex_{_k}"

IGNORED_KEYS = {"name", "pipeline.name", "_target_"}
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
        raise TypeError("Cholimex config must contain a mapping")
    for config_key, value in _flatten_mapping(data).items():
        if config_key in IGNORED_KEYS:
            continue
        field_name = FIELD_ALIASES.get(config_key)
        if field_name is None:
            raise ValueError(f"unknown config key: {config_key}")
        setattr(cfg, field_name, Path(value) if field_name in PATH_FIELDS and value is not None else value)
    return cfg


def load_config(path: Path | str | None = None) -> Config:
    """Load a resolved Cholimex config (JSON or YAML); defaults when no path is given."""
    if path is None:
        return Config()
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Cholimex config does not exist: {path_obj}")

    if path_obj.suffix in {".yaml", ".yml"}:
        try:
            from omegaconf import OmegaConf
            loaded = OmegaConf.load(path_obj)
            raw_data = OmegaConf.to_container(loaded, resolve=True)
        except ImportError:
            import yaml
            with path_obj.open("r", encoding="utf-8") as handle:
                raw_data = yaml.safe_load(handle)
        return apply_config_data(Config(), raw_data)

    with path_obj.open("r", encoding="utf-8") as handle:
        return apply_config_data(Config(), json.load(handle))


def config_from_omegaconf(cfg_dict: Any) -> Config:
    """Convert an OmegaConf DictConfig or dict to a Config dataclass."""
    try:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(cfg_dict, DictConfig):
            data = OmegaConf.to_container(cfg_dict, resolve=True)
        else:
            data = dict(cfg_dict)
    except ImportError:
        data = dict(cfg_dict)
    return apply_config_data(Config(), data)

