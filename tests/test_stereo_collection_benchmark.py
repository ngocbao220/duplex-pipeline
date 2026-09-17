from __future__ import annotations

import json

from core.stereo_benchmark.collection import run_collection_benchmark


def test_collection_benchmark_writes_one_row_per_conversation(monkeypatch, tmp_path):
    first = tmp_path / "conversations" / "conversation_00000" / "audio.stereo.wav"
    second = tmp_path / "conversations" / "conversation_00001" / "audio.stereo.wav"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_benchmark(audio, output_dir, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {"acoustic_quality": {"dnsmos": {"left": {"status": "ok"}}, "squim": {"left": {"status": "ok"}}},
                  "speaker_identity": {"itc": {"mean": {"status": "ok"}}, "itd": {"status": "ok"}}}
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report))
        return report, report_path

    monkeypatch.setattr("core.stereo_benchmark.collection.run_benchmark", fake_benchmark)
    manifest = tmp_path / "conversations" / "manifest.json"
    manifest.write_text(json.dumps({"conversations": [
        {"conversation_idx": 0, "duration": 2.0, "stereo": str(first)},
        {"conversation_idx": 1, "duration": 3.0, "stereo": str(second)},
    ]}))

    summary = json.loads(run_collection_benchmark(manifest, tmp_path).read_text())
    assert summary["mode"] == "reference_free"
    assert summary["aggregate"]["duration_sec"] == 5.0
    assert len(summary["conversations"]) == 2
    assert summary["runtime"]["total_seconds"] >= 0
