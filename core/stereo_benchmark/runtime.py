"""Runtime paths for the standalone stereo benchmark CLI."""
from __future__ import annotations

import os
from pathlib import Path


def apply_environment_config(path: Path) -> dict[str, str]:
    """Export benchmark model paths from one Hydra environment YAML."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = data.get("paths", {})
    base_models = str(paths.get("base_models", ""))

    def resolve(value: object) -> str:
        return str(value or "").replace("${env.paths.base_models}", base_models)

    configured = {
        "SPEECHBRAIN_MODEL_PATH": resolve(paths.get("speechbrain")),
        "SQUIM_MODEL_PATH": resolve(paths.get("squim")),
        "NISQA_MODEL_PATH": resolve(paths.get("nisqa")),
        "DNSMOS_MODEL_PATH": resolve(paths.get("dnsmos")),
    }
    for variable, value in configured.items():
        if value:
            os.environ[variable] = value
    if data.get("offline", False):
        os.environ["MODE"] = str(data.get("name", "sever"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    return configured
