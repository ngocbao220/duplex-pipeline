"""Validate local DialogueSidon bundles without importing GPU/audio runtime code."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from .model_utils import assert_local_model_exists

MODEL_FILES = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
MODEL_MANIFEST = "model_manifest.json"


class LocalModelValidationError(RuntimeError):
    """A local DialogueSidon bundle is incomplete, modified, or unreadable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dialoguesidon_manifest(model_dir: str | Path) -> Path:
    """Write a checksum manifest; run only where the source bundle is trusted."""
    target_dir = assert_local_model_exists(model_dir, required_files=MODEL_FILES, model_name_hint="DialogueSidon")
    manifest_path = target_dir / MODEL_MANIFEST
    manifest_path.write_text(
        json.dumps({"files": {filename: _sha256(target_dir / filename) for filename in MODEL_FILES}}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def validate_dialoguesidon_model(model_dir: str | Path) -> dict[str, str]:
    """Validate a checksummed DialogueSidon local bundle without Hugging Face."""
    target_dir = assert_local_model_exists(
        model_dir, required_files=MODEL_FILES + [MODEL_MANIFEST], model_name_hint="DialogueSidon"
    )
    manifest_path = target_dir / MODEL_MANIFEST
    try:
        expected_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LocalModelValidationError(
            f"DialogueSidon checksum manifest '{manifest_path}' must contain a 'files' object mapping every model file to SHA-256."
        ) from exc
    if not isinstance(expected_hashes, dict):
        raise LocalModelValidationError(f"DialogueSidon checksum manifest '{manifest_path}' has an invalid 'files' value.")

    paths: dict[str, str] = {}
    for filename in MODEL_FILES:
        file_path = target_dir / filename
        expected = expected_hashes.get(filename)
        if not isinstance(expected, str) or len(expected) != 64:
            raise LocalModelValidationError(f"DialogueSidon checksum manifest '{manifest_path}' is missing a SHA-256 for '{filename}'.")
        actual = _sha256(file_path)
        if actual.lower() != expected.lower():
            raise LocalModelValidationError(
                f"DialogueSidon local artifact '{file_path}' SHA-256 mismatch: expected {expected}, got {actual}. "
                "Do not run this bundle; copy the verified file again."
            )
        paths[filename] = str(file_path)

    try:
        metadata = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
        required = {"latent_norm_mean", "latent_norm_std", "latent_norm_initialized", "ddpm_config", "latent_dim", "sample_rate"}
        missing = sorted(required - metadata.keys())
        if missing:
            raise KeyError(", ".join(missing))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LocalModelValidationError(f"DialogueSidon metadata '{target_dir / 'metadata.json'}' is invalid or missing required fields: {exc}.") from exc

    for filename in MODEL_FILES[:-1]:
        try:
            with zipfile.ZipFile(target_dir / filename, "r") as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise LocalModelValidationError(f"DialogueSidon local artifact '{target_dir / filename}' has a corrupt ZIP member '{bad_member}'.")
        except zipfile.BadZipFile as exc:
            raise LocalModelValidationError(
                f"DialogueSidon local artifact '{target_dir / filename}' is not a readable ZIP/PT2 archive. "
                "It is truncated or not the exported model file; copy it again from the verified local bundle."
            ) from exc
    return paths
