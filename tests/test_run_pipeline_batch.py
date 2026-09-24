import subprocess
import importlib.util
import json
import sys
from pathlib import Path

from core.resource_tuning import GpuSnapshot

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


def test_sommelier_publish_keeps_one_numbered_stereo_without_copying_audio(tmp_path):
    batch = _batch_module()
    run_output = tmp_path / ".runs" / "dialogue_2"
    collection_output = tmp_path / "sommelier" / "youtube_100"
    run_output.mkdir(parents=True)
    collection_output.mkdir(parents=True)
    dialogue = tmp_path / "dialogue_2.wav"
    dialogue.touch()
    generated = run_output / "stereo_2.wav"
    generated.write_bytes(b"stereo")
    (collection_output / "audio.stereo.wav").write_bytes(b"legacy duplicate")

    published = batch._publish_sommelier_stereo(run_output, collection_output, dialogue)

    assert published == collection_output / "stereo_2.wav"
    assert published.read_bytes() == b"stereo"
    assert published.stat().st_ino == generated.stat().st_ino
    assert not (collection_output / "audio.stereo.wav").exists()


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


def test_split_dialogue_report_counts_lid_decisions_and_lists_rejected_audio(tmp_path):
    batch = _batch_module()
    output_dir = tmp_path / "dialogues"
    accepted = output_dir / "accepted"
    rejected = output_dir / "rejected"
    accepted.mkdir(parents=True)
    rejected.mkdir()
    (accepted / "manifest.json").write_text(json.dumps({
        "source_audio": "/input/accepted.wav",
        "audio_duration_sec": 3600,
        "dialogue_count": 1,
        "skip_reason": None,
        "candidate_dialogues": [{"duration": 120}, {"duration": 60, "decision": "rejected_by_lid"}],
        "dialogues": [{"duration": 120}],
        "filter_summary": {
            "candidate_dialogue_count": 2,
            "exported_dialogue_count": 1,
            "rejected_by_lid": 1,
            "rejected_short": 3,
            "rejected_imbalanced": 2,
            "diarized_speaker_count": 2,
            "lid": {"enabled": True, "model": "/models/whisper", "min_vi_probability": 0.5},
        },
    }), encoding="utf-8")
    (rejected / "manifest.json").write_text(json.dumps({
        "source_audio": "/input/rejected.wav",
        "audio_duration_sec": 1800,
        "dialogue_count": 0,
        "skip_reason": "All 1 candidate clips rejected by Whisper LID",
        "candidate_dialogues": [{"duration": 90, "decision": "rejected_by_lid"}],
        "dialogues": [],
        "filter_summary": {
            "candidate_dialogue_count": 1,
            "exported_dialogue_count": 0,
            "rejected_by_lid": 1,
            "rejected_short": 0,
            "rejected_imbalanced": 0,
            "diarized_speaker_count": 2,
            "lid": {"enabled": True, "model": "/models/whisper", "min_vi_probability": 0.5},
        },
    }), encoding="utf-8")

    report = batch._summarize_split_dialogue_manifests(output_dir, input_audio_count=2)

    assert report["lid"]["enabled"] is True
    assert report["lid"]["model"] == "/models/whisper"
    assert report["counts"] == {
        "input_audio": 2,
        "manifest_audio": 2,
        "candidate_dialogues": 3,
        "exported_dialogues": 1,
        "rejected_by_lid": 2,
        "rejected_short": 3,
        "rejected_imbalanced": 2,
        "monologue_audio": 0,
        "zero_dialogue_audio": 1,
    }
    assert report["audio"][1]["source_audio"] == "/input/rejected.wav"
    assert report["audio"][1]["skip_reason"] == "All 1 candidate clips rejected by Whisper LID"
    assert report["retention"]["source_hours"] == 1.5
    assert report["retention"]["processed_source_hours"] == 1.5
    assert report["retention"]["after_dialogue"]["hours"] == 0.075
    assert report["retention"]["after_dialogue"]["retention_percent_of_source"] == 5.0
    assert report["retention"]["after_dialogue"]["removed_hours"] == 1.425
    assert report["retention"]["after_lid"]["hours"] == 120 / 3600
    assert report["retention"]["after_lid"]["removed_hours"] == 150 / 3600
    assert report["retention"]["largest_duration_drop_phase"] == "dialogue"
    assert report["audio"][0]["retention"]["after_lid"]["retention_percent_of_source"] == 120 / 3600 * 100


def test_split_dialogue_retention_marks_unprocessed_audio_and_lid_disabled(tmp_path):
    batch = _batch_module()
    output_dir = tmp_path / "dialogues"
    source = tmp_path / "input" / "complete.wav"
    manifest_dir = output_dir / "complete"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({
        "source_audio": str(source),
        "audio_duration_sec": 3600,
        "dialogue_count": 1,
        "candidate_dialogues": [{"duration": 300}],
        "dialogues": [{"duration": 300}],
        "filter_summary": {
            "candidate_dialogue_count": 1,
            "exported_dialogue_count": 1,
            "rejected_by_lid": 0,
            "rejected_short": 0,
            "rejected_imbalanced": 0,
            "diarized_speaker_count": 2,
            "lid": {"enabled": False, "model": None, "min_vi_probability": None},
        },
    }), encoding="utf-8")

    report = batch._summarize_split_dialogue_manifests(
        output_dir,
        input_audio_count=2,
        input_audio_durations={str(source): 3600, str(tmp_path / "input" / "missing.wav"): 1800},
    )

    assert report["lid"]["enabled"] is False
    assert report["retention"]["source_hours"] == 1.5
    assert report["retention"]["unprocessed_source_hours"] == 0.5
    assert report["retention"]["after_dialogue"]["hours"] == 300 / 3600
    assert report["retention"]["after_lid"]["hours"] == 300 / 3600
    assert report["retention"]["after_lid"]["removed_hours"] == 0
    missing = next(item for item in report["audio"] if item["skip_reason"] == "No completed manifest")
    assert missing["retention"]["after_dialogue"]["hours"] is None


def test_split_dialogue_batch_writes_a_lid_audit_with_the_configured_threshold(monkeypatch, tmp_path):
    batch = _batch_module()
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "dialogues"
    raw_dir.mkdir()
    source = raw_dir / "source.wav"
    source.write_bytes(b"raw")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        worker_output = Path(command[command.index("--output-dir") + 1])
        worker_output.mkdir(parents=True)
        (worker_output / "manifest.json").write_text(json.dumps({
            "source_audio": str(source),
            "dialogue_count": 0,
            "audio_duration_sec": 10,
            "skip_reason": "All 1 candidate clips rejected by Whisper LID",
            "candidate_dialogues": [{"duration": 8, "decision": "rejected_by_lid"}],
            "dialogues": [],
            "filter_summary": {
                "candidate_dialogue_count": 1,
                "exported_dialogue_count": 0,
                "rejected_by_lid": 1,
                "rejected_short": 0,
                "rejected_imbalanced": 0,
                "diarized_speaker_count": 2,
                "lid": {"enabled": True, "model": "local-whisper", "min_vi_probability": 0.7},
            },
        }), encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setattr(batch, "_audio_duration_seconds", lambda _path: 10.0)
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline_batch.py", "--step", "split_dialogue", "--input-dir", str(raw_dir),
        "--output-dir", str(output_dir), "--lid", "vi", "--min-lid-prob", "0.7",
    ])

    batch.run_batch()

    command = next(command for command in commands if "--lid" in command)
    assert ["--lid", "vi", "--min-lid-prob", "0.7"] == command[command.index("--lid"):command.index("--lid") + 4]
    report = json.loads((output_dir / "split_dialogue_report.json").read_text())
    assert report["lid"] == {"enabled": True, "model": "local-whisper", "min_vi_probability": 0.7}
    assert report["counts"]["rejected_by_lid"] == 1
    assert report["retention"]["source_hours"] == 10 / 3600
    assert report["retention"]["after_dialogue"]["hours"] == 8 / 3600
    assert report["retention"]["after_lid"]["hours"] == 0


def test_cholimex_dry_run_pairs_each_duplexchat_stereo_with_its_dialogue_mixture(monkeypatch, capsys, tmp_path):
    batch = _batch_module()
    duplex_dir = tmp_path / "duplexchat"
    dialogue_dir = tmp_path / "dialogues"
    (duplex_dir / "episode").mkdir(parents=True)
    (dialogue_dir / "episode").mkdir(parents=True)
    (duplex_dir / "episode" / "stereo_1.wav").write_bytes(b"stereo")
    (dialogue_dir / "episode" / "dialogue_1.wav").write_bytes(b"mixture")

    monkeypatch.setattr(sys, "argv", [
        "run_pipeline_batch.py",
        "--step", "cholimex", "--input-dir", str(duplex_dir),
        "--mixture-dir", str(dialogue_dir), "--output-dir", str(tmp_path / "cholimex"), "--dry-run",
    ])
    batch.run_batch()
    output = capsys.readouterr().out

    assert "stereo_1.wav" in output
    assert "dialogue_1.wav" in output
    assert "-m cholimex collection" in output


def test_auto_tuning_calibrates_before_expanding_gpu_workers(monkeypatch):
    batch = _batch_module()
    snapshot = GpuSnapshot(index="0", total_memory_mib=40_000, free_memory_mib=38_000)

    class _Sampler:
        def start(self):
            return None

        def stop(self):
            return {"0": GpuSnapshot(index="0", total_memory_mib=40_000, free_memory_mib=34_000, peak_memory_used_mib=6_000)}

    monkeypatch.setattr(batch, "probe_gpus", lambda: [snapshot])
    monkeypatch.setattr(batch, "GpuMemorySampler", _Sampler)
    calls = []

    results, plan = batch._run_gpu_items(
        [1, 2, 3], ["0"], lambda item, gpu: calls.append((item, gpu)) or item,
        resource_mode="auto", configured_workers_per_gpu=1, reserve_ratio=0.10, max_workers_per_gpu=8,
    )

    assert results == [1, 2, 3]
    assert calls[0] == (1, "0")
    assert plan["workers_per_gpu"] == {"0": 6}


def test_explicit_gpu_identifier_is_not_wrapped_by_visible_gpu_count(monkeypatch, tmp_path):
    batch = _batch_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "source.wav").write_bytes(b"raw")
    captured = []
    monkeypatch.setattr(batch, "_audio_duration_seconds", lambda _path: 1.0)
    monkeypatch.setattr(batch.subprocess, "run", lambda _cmd, **kwargs: captured.append(kwargs["env"]) or type("Result", (), {"returncode": 0})())
    monkeypatch.setattr(batch, "_available_gpu_count", lambda: 8)
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline_batch.py", "--step", "split_dialogue", "--input-dir", str(raw_dir),
        "--output-dir", str(tmp_path / "out"), "--gpu", "5",
    ])

    batch.run_batch()

    assert captured[0]["CUDA_VISIBLE_DEVICES"] == "5"


def test_resume_helpers_only_accept_complete_phase_artifacts(tmp_path):
    batch = _batch_module()
    split_out = tmp_path / "split"
    split_out.mkdir()
    (split_out / "manifest.json").write_text(json.dumps({"dialogue_count": 1, "dialogues": [{"filename": "dialogue_1.wav"}]}))
    assert not batch._split_dialogue_complete(split_out)
    (split_out / "dialogue_1.wav").write_bytes(b"audio")
    assert batch._split_dialogue_complete(split_out)

    dialogue_dir = tmp_path / "dialogues"
    dialogue_dir.mkdir()
    (dialogue_dir / "dialogue_1.wav").write_bytes(b"audio")
    duplex_out = tmp_path / "duplex"
    duplex_out.mkdir()
    assert not batch._duplexchat_complete(dialogue_dir, duplex_out)
    (duplex_out / "stereo_1.wav").write_bytes(b"stereo")
    assert batch._duplexchat_complete(dialogue_dir, duplex_out)

    cholimex_out = tmp_path / "cholimex_1"
    cholimex_out.mkdir()
    assert not batch._cholimex_complete(cholimex_out, 1)
    (cholimex_out / "cholimex_stereo_1.wav").write_bytes(b"stereo")
    assert batch._cholimex_complete(cholimex_out, 1)


def test_speechbrain_preflight_requires_the_complete_local_bundle(monkeypatch, tmp_path):
    batch = _batch_module()
    model_dir = tmp_path / "spker"
    model_dir.mkdir()
    monkeypatch.setenv("SPEECHBRAIN_MODEL_PATH", str(model_dir))

    _path, status, valid = batch._check_model_status(
        "Speaker Embedding", "speechbrain/spkrec-ecapa-voxceleb", "SPEECHBRAIN_MODEL_PATH",
        "spkrec-ecapa-voxceleb", ["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt", "mean_var_norm_emb.ckpt"],
    )

    assert valid is False
    assert "INCOMPLETE" in status
