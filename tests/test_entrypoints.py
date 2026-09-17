from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from core.orchestration.cli import build_pipeline_parser


def test_each_pipeline_cli_exposes_single_command():
    for name in ("vilier", "duplexchat", "cholimex"):
        parser = build_pipeline_parser(name)
        single_args = parser.parse_args(["single", "--input", "mixture.wav", "--output-dir", "out"])
        assert single_args.command == "single"


def test_cholimex_cli_exposes_collection_refinement_command():
    args = build_pipeline_parser("cholimex").parse_args([
        "collection", "--input", "duplexchat/stereo_1.wav",
        "--mixture", "conversations/conversation_00000/mixture.wav", "--output-dir", "out",
    ])

    assert args.command == "collection"
    assert args.input == Path("duplexchat/stereo_1.wav")
    assert args.mixture == Path("conversations/conversation_00000/mixture.wav")


def test_duplexchat_cli_uses_conversation_mode_by_default():
    parser = build_pipeline_parser("duplexchat")

    args = parser.parse_args(["single", "--input", "mixture.wav", "--output-dir", "out"])
    assert args.separate_chunk == 120.0


def test_duplexchat_cli_exposes_separation_chunk():
    parser = build_pipeline_parser("duplexchat")

    args = parser.parse_args([
        "single", "--input", "mixture.wav", "--output-dir", "out",
        "--separation-chunk", "30",
    ])

    assert args.separate_chunk == 30.0


def test_duplexchat_cli_accepts_explicit_local_separation_model():
    parser = build_pipeline_parser("duplexchat")

    args = parser.parse_args([
        "separate_dialogue", "--input", "dialogues", "--output-dir", "out",
        "--separation-model", "/models/DialogueSidon",
    ])

    assert args.separation_model == Path("/models/DialogueSidon")


def test_duplexchat_cli_exposes_local_model_validation_command():
    args = build_pipeline_parser("duplexchat").parse_args([
        "validate_model", "--separation-model", "/models/DialogueSidon",
    ])

    assert args.command == "validate_model"
    assert args.separation_model == Path("/models/DialogueSidon")


def test_duplexchat_cli_keeps_separate_chunk_alias():
    parser = build_pipeline_parser("duplexchat")

    args = parser.parse_args([
        "single", "--input", "mixture.wav", "--output-dir", "out",
        "--separate-chunk", "30",
    ])

    assert args.separate_chunk == 30.0


def test_duplexchat_cli_accepts_explicit_gpu_ids():
    args = build_pipeline_parser("duplexchat").parse_args([
        "single", "--input", "mixture.wav", "--output-dir", "out", "--device-ids", "0", "1",
    ])
    assert args.device_ids == [0, 1]


def test_duplexchat_cli_does_not_accept_ground_truth_flags():
    with pytest.raises(SystemExit):
        build_pipeline_parser("duplexchat").parse_args([
            "single", "--input", "mixture.wav", "--output-dir", "out", "--gt-speaker-a", "a.wav",
        ])


def test_duplexchat_cli_rejects_removed_conversation_scale_option():
    parser = build_pipeline_parser("duplexchat")

    with pytest.raises(SystemExit):
        parser.parse_args(["single", "--input", "mixture.wav", "--output-dir", "out", "--scale", "true"])


def test_duplexchat_cli_accepts_diarization_backend_and_model():
    parser = build_pipeline_parser("duplexchat")
    args = parser.parse_args([
        "single", "--input", "mixture.wav", "--output-dir", "out",
        "--diarization-backend", "sortformer",
        "--diarization-model", "nvidia/diar_streaming_sortformer_4spk-v2.1",
    ])
    assert args.diarization_backend == "sortformer"
    assert args.diarization_model == "nvidia/diar_streaming_sortformer_4spk-v2.1"


def test_vilier_reconstruct_imports_the_renamed_preprocess_module():
    source_root = Path(__file__).resolve().parents[1] / "pipeline" / "vilier" / "src"
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module("vilier.reconstruct")
    finally:
        sys.path.remove(str(source_root))
    assert callable(module.export_segments_and_tracks)


def test_each_pipeline_owns_named_source_modules_with_contract_headers():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "vilier": {"preprocess.py", "vad.py", "diarization.py", "separation.py", "reconstruct.py", "runner.py"},
        "duplexchat": {"preprocess.py", "diarization.py", "dialogues.py", "separation.py", "reconstruct.py", "runner.py"},
        "cholimex": {"preprocess.py", "vad.py", "regions.py", "speaker_assignment.py", "separation.py", "reconstruct.py", "runner.py"},
    }
    for name, modules in expected.items():
        phase_dir = root / "pipeline" / name / "src" / name
        assert modules <= {path.name for path in phase_dir.glob("*.py")}
        for source in phase_dir.glob("*.py"):
            docstring = source.read_text(encoding="utf-8").lstrip()
            assert docstring.startswith('"""')
            assert all(label in docstring.split('"""', 2)[1] for label in ("Purpose:", "Inputs:", "Outputs:"))
