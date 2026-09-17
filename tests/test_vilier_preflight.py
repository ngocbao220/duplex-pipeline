from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VILIER_SRC = ROOT / "pipeline" / "vilier" / "src"
sys.path.insert(0, str(VILIER_SRC))

from vilier.separation import preflight_overlap_separator, validate_checkpoint_file
from core.orchestration.process import is_console_noise


def test_sepreformer_preflight_fails_before_diarization_with_actionable_checkpoint_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="VILIER_SEPREFORMER_CHECKPOINT"):
        preflight_overlap_separator(
            {
                "enabled": True,
                "backend": "sepreformer",
                "sepreformer_path": str(tmp_path / "SepReformer"),
                "model_name": "SepReformer_Base_WSJ0",
            }
        )


def test_console_noise_is_kept_in_worker_log_but_not_rendered_as_progress():
    assert is_console_noise("OneLogger: Setting error_handling_strategy\n")
    assert is_console_noise("[NeMo W 2026-09-08] ignored configuration\n")
    assert is_console_noise("Diarizing: 1it [00:01, 1.86s/it]\n")
    assert not is_console_noise("PipelineRunError: missing checkpoint\n")


def test_git_lfs_pointer_is_rejected_before_torch_load(tmp_path):
    checkpoint = tmp_path / "epoch.0180.pth"
    checkpoint.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Git-LFS pointer"):
        validate_checkpoint_file(checkpoint)


def test_preflight_accepts_explicit_trusted_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "trusted.pth"
    checkpoint.write_bytes(b"torch checkpoint bytes")

    preflight_overlap_separator(
        {
            "enabled": True,
            "backend": "sepreformer",
            "sepreformer_path": str(tmp_path),
            "checkpoint_path": str(checkpoint),
        }
    )
