from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from core.orchestration.cli import build_pipeline_parser


def test_each_pipeline_cli_exposes_single_command():
    for name in ("duplexchat", "cholimex"):
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


def test_sommelier_single_does_not_load_hydra_config_as_runtime_config(tmp_path, monkeypatch):
    import json

    import core.config
    import core.orchestration.single as single

    def reject_shared_config(*_args, **_kwargs):
        raise AssertionError("single pipeline must not parse Hydra's root config as runtime config")

    monkeypatch.setattr(core.config, "load_config", reject_shared_config)

    def complete_launch(_name, request, _run_dir):
        result_path = Path(request["results"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps([{"status": "complete"}]), encoding="utf-8")
        assert request["config"]["name"] == "sommelier"
        return 0

    monkeypatch.setattr(single, "launch_pipeline", complete_launch)
    result = single.run_single(
        "sommelier", tmp_path / "dialogue_2.wav", tmp_path / "out" / "dialogue_2",
        debug=False, separate_chunk=120.0, device_ids=None,
    )

    assert result == 0


def test_worker_always_uses_current_interpreter_instead_of_uv(tmp_path, monkeypatch):
    import core.orchestration.runner as runner

    root = tmp_path / "repo"
    (root / "pipeline" / "sommelier").mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    observed = {}

    def capture(command, cwd, log_path, env):
        observed.update(command=command, cwd=cwd, log_path=log_path, env=env)
        return 0

    monkeypatch.setattr(runner, "stream_process", capture)
    run_dir = tmp_path / "run"
    assert runner.launch_pipeline("sommelier", {"pipeline": "sommelier"}, run_dir) == 0

    assert observed["command"][0] == sys.executable
    assert observed["cwd"] == root / "pipeline" / "sommelier"
    assert "UV_PROJECT_ENVIRONMENT" not in observed["env"]


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


def test_duplexchat_split_cli_accepts_dialogue_filter_thresholds():
    args = build_pipeline_parser("duplexchat").parse_args([
        "split_valid_dialogue", "--input", "source.wav", "--output-dir", "out",
        "--dialogue-gap-seconds", "4.0", "--min-dialogue-duration-seconds", "8.0",
        "--max-dialogue-duration-seconds", "500", "--max-single-speaker-ratio", "0.75",
        "--preferred-split-pause-seconds", "3.5", "--min-split-pause-seconds", "1.5",
    ])

    assert args.dialogue_gap_seconds == 4.0
    assert args.min_dialogue_duration_seconds == 8.0
    assert args.max_dialogue_duration_seconds == 500.0
    assert args.max_single_speaker_ratio == 0.75
    assert args.preferred_split_pause_seconds == 3.5
    assert args.min_split_pause_seconds == 1.5


def test_each_pipeline_owns_named_source_modules_with_contract_headers():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "duplexchat": {"preprocess.py", "diarization.py", "dialogue.py", "separation.py", "reconstruct.py", "runner.py"},
        "cholimex": {"preprocess.py", "vad.py", "regions.py", "speaker_assignment.py", "separation.py", "reconstruct.py", "runner.py"},
    }
    for name, modules in expected.items():
        phase_dir = root / "pipeline" / name / "src" / name
        assert modules <= {path.name for path in phase_dir.glob("*.py")}
        for source in phase_dir.glob("*.py"):
            if source.name in modules:
                docstring = source.read_text(encoding="utf-8").lstrip()
                assert docstring.startswith('"""')
                assert all(label in docstring.split('"""', 2)[1] for label in ("Purpose:", "Inputs:", "Outputs:"))
