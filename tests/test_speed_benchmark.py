from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_speed  # noqa: E402
import summarize_speed  # noqa: E402
from core.orchestration.logging_style import StepTimer  # noqa: E402


def test_sampling_is_reproducible_and_links_the_chosen_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(20):
        sf.write(raw / f"ep_{index:02d}.wav", np.zeros(1_600, dtype=np.float32), 16_000)

    first = benchmark_speed.sample_raw_files(raw, "*.wav", 5, seed=7)
    assert first == benchmark_speed.sample_raw_files(raw, "*.wav", 5, seed=7)
    assert first != benchmark_speed.sample_raw_files(raw, "*.wav", 5, seed=8)

    sample = benchmark_speed.build_sample(tmp_path / "bench", raw, "*.wav", 5, seed=7)
    assert sample["count"] == 5
    assert np.isclose(sample["audio_seconds"], 0.5)
    assert sorted(p.name for p in (tmp_path / "bench" / "raw").iterdir()) == [p.name for p in first]


def test_phase_commands_point_every_stage_into_the_bench_directory(tmp_path):
    cmd = benchmark_speed.phase_command("cholimex", tmp_path, "0", 4, ["env=dev"], dry_run=True)
    assert "step=cholimex" in cmd and "pipeline=cholimex" in cmd
    assert f"data.duplex_out_dir={tmp_path / 'duplexchat'}" in cmd
    assert "optimization.workers=4" in cmd and "env=dev" in cmd and cmd[-1] == "dry_run=true"


def _phase(step, elapsed, items):
    audio = sum(item["audio_seconds"] for item in items)
    return {"step": step, "elapsed_seconds": elapsed, "audio_seconds": audio, "item_count": len(items),
            "failed_item_count": sum(item["status"] != "complete" for item in items), "items": items, "gpu": "0"}


def _item(audio, elapsed, status="complete"):
    return {"audio_seconds": audio, "elapsed_seconds": elapsed, "real_time_factor": elapsed / audio, "status": status}


def test_summary_reports_seconds_per_hour_and_end_to_end_pipelines(tmp_path):
    (tmp_path / "sample.json").write_text(json.dumps({"count": 2, "seed": 0, "audio_seconds": 7200.0, "audio_hours": 2.0}))
    (tmp_path / "timing.json").write_text(json.dumps({"workflow_id": "bench-x", "phases": [
        _phase("split_dialogue", 720.0, [_item(3600, 400), _item(3600, 400)]),
        _phase("separate_dialogue", 360.0, [_item(1800, 360)]),
        _phase("sommelier", 900.0, [_item(1800, 900)]),
    ]}))
    (tmp_path / "steps.jsonl").write_text(
        json.dumps({"phase": "split", "step": "2. Split Dialogue", "elapsed_seconds": 90.0, "audio_seconds": 3600.0}) + "\n"
        + json.dumps({"phase": "split", "step": "2. Split Dialogue", "elapsed_seconds": 90.0, "audio_seconds": 3600.0}) + "\n"
    )

    summarize_speed.summarize(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    rows = {row["step"]: row for row in report["phases"]}

    assert rows["split_dialogue"]["seconds_per_audio_hour"] == 360.0
    assert rows["split_dialogue"]["rtf"] == 0.1
    assert rows["separate_dialogue"]["seconds_per_audio_hour"] == 720.0
    assert rows["separate_dialogue"]["seconds_per_raw_hour"] == 180.0
    pipelines = {row["pipeline"]: row["seconds_per_raw_hour"] for row in report["pipelines"]}
    assert pipelines == {"DuplexChat": 540.0, "Sommelier": 810.0}
    assert report["substeps"][0]["seconds_per_audio_hour"] == 90.0
    assert "| Split dialogue |" in (tmp_path / "report.md").read_text()


def test_step_timer_appends_jsonl_only_when_enabled(tmp_path, monkeypatch):
    log = logging.getLogger("bench-test")
    with StepTimer(log, "quiet", duration_sec=1.0):
        pass
    path = tmp_path / "steps.jsonl"
    monkeypatch.setenv("PIPELINE_STEP_TIMINGS", str(path))
    monkeypatch.setenv("PIPELINE_BENCH_PHASE", "split")
    with StepTimer(log, "1. Preprocessing", duration_sec=10.0):
        pass
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["phase"], row["step"], row["audio_seconds"]) for row in rows] == [("split", "1. Preprocessing", 10.0)]
