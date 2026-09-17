import subprocess
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _batch_module():
    spec = importlib.util.spec_from_file_location("run_pipeline_batch_test", ROOT / "scripts" / "run_pipeline_batch.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_pipeline_batch_dry_run(tmp_path):
    input_dir = tmp_path / "raw_in"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "sample_1.wav").write_bytes(b"placeholder")
    output_dir = tmp_path / "processed_out"

    cmd = [
        "python", str(ROOT / "scripts" / "run_pipeline_batch.py"),
        "--step", "split_dialogue",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--gpu", "0",
        "--dry-run"
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert "Running split_dialogue on 1 files" in res.stdout
    assert "sample_1.wav" in res.stdout


def test_batch_timing_aggregates_steps_and_records_audio_speed(monkeypatch, tmp_path):
    batch = _batch_module()
    raw_dir = tmp_path / "raw"
    dialogue_dir = tmp_path / "dialogue"
    raw_dir.mkdir()
    dialogue_dir.mkdir()
    raw = raw_dir / "source.wav"
    dialogue = dialogue_dir / "dialogue_1.wav"
    raw.write_bytes(b"raw")
    dialogue.write_bytes(b"dialogue")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_RUN_ID", "workflow-test")
    monkeypatch.setattr(batch.subprocess, "run", lambda *_args, **_kwargs: type("Result", (), {"returncode": 0})())
    monkeypatch.setattr(batch, "_audio_duration_seconds", lambda _path: 10.0)

    monkeypatch.setattr(sys, "argv", ["run_pipeline_batch.py", "--step", "split_dialogue", "--input-dir", str(raw_dir), "--output-dir", str(tmp_path / "split")])
    batch.run_batch()
    monkeypatch.setattr(sys, "argv", ["run_pipeline_batch.py", "--step", "separate_dialogue", "--input-dir", str(dialogue_dir), "--output-dir", str(tmp_path / "separated")])
    batch.run_batch()

    report = json.loads((tmp_path / "outputs" / "pipeline_timing.json").read_text())
    assert report["workflow_id"] == "workflow-test"
    assert [phase["step"] for phase in report["phases"]] == ["split_dialogue", "separate_dialogue"]
    assert report["summary"]["phase_count"] == 2
    assert report["phases"][0]["items"][0]["audio_seconds"] == 10.0
    assert report["phases"][0]["items"][0]["audio_seconds_per_wall_second"] is not None


def test_batch_timing_records_failed_subprocess_and_dry_run_writes_nothing(monkeypatch, tmp_path):
    batch = _batch_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "source.wav").write_bytes(b"raw")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIPELINE_RUN_ID", "failure-test")
    monkeypatch.setattr(batch.subprocess, "run", lambda *_args, **_kwargs: type("Result", (), {"returncode": 9})())
    monkeypatch.setattr(sys, "argv", ["run_pipeline_batch.py", "--step", "split_dialogue", "--input-dir", str(raw_dir), "--output-dir", str(tmp_path / "split")])
    batch.run_batch()

    report = json.loads((tmp_path / "outputs" / "pipeline_timing.json").read_text())
    assert report["summary"]["failed_item_count"] == 1
    assert report["phases"][0]["items"][0]["exit_code"] == 9

    (tmp_path / "outputs" / "pipeline_timing.json").unlink()
    monkeypatch.setattr(sys, "argv", ["run_pipeline_batch.py", "--step", "split_dialogue", "--input-dir", str(raw_dir), "--output-dir", str(tmp_path / "split"), "--dry-run"])
    batch.run_batch()
    assert not (tmp_path / "outputs" / "pipeline_timing.json").exists()
