from __future__ import annotations

import numpy as np
import pytest

from core.stereo_benchmark.models import acoustic_metrics, _nisqa_one
from core.stereo_benchmark.report import summarize_reports, render_tables


def test_nisqa_one_returns_unavailable_when_package_missing_or_audio_empty():
    empty = np.array([], dtype=np.float32)
    res = _nisqa_one(empty, "cpu")
    assert res["status"] == "unavailable"

    audio = np.random.randn(16000).astype(np.float32)
    res_audio = _nisqa_one(audio, "cpu")
    # Should be unavailable or ok depending on whether NISQA is installed
    assert res_audio["status"] in ("ok", "unavailable")


def test_nisqa_scores_long_audio_in_bounded_chunks(monkeypatch, tmp_path):
    from pathlib import Path
    import soundfile as sf
    import core.stereo_benchmark.models as models

    monkeypatch.setattr(models, "_ensure_nisqa_installed", lambda: True)
    monkeypatch.setattr(models, "_nisqa_checkpoint_path", lambda: tmp_path / "nisqa.tar")

    def predict(args, _device):
        files = sorted(Path(args["data_dir"]).glob("*.wav"))
        assert len(files) == 5
        assert all(sf.info(path).duration <= 45 for path in files)

        class Result:
            def iterrows(self):
                for index, path in enumerate(files):
                    yield index, {"deg": path.name, "mos_pred": 2.0 if path.name.startswith("left") else 4.0}

        return Result()

    monkeypatch.setattr(models, "_predict_nisqa", predict)
    result = models._nisqa_metrics(
        np.zeros(61 * models.SAMPLE_RATE, dtype=np.float32),
        np.zeros(125 * models.SAMPLE_RATE, dtype=np.float32),
        "cpu",
    )
    assert result["left"]["nisqa_mos"] == 2.0
    assert result["right"]["nisqa_mos"] == 4.0


def test_acoustic_metrics_includes_nisqa_dict():
    left = np.random.randn(16000).astype(np.float32)
    right = np.random.randn(16000).astype(np.float32)
    metrics = acoustic_metrics(left, right, "cpu")
    assert "nisqa" in metrics
    assert "left" in metrics["nisqa"]
    assert "right" in metrics["nisqa"]


def test_report_summary_and_tables_render_nisqa():
    report = {
        "acoustic_quality": {
            "dnsmos": {"mean": {"ovrl": 3.8}},
            "nisqa": {"mean": {"nisqa_mos": 4.1}},
            "squim": {"mean": {"sq_stoi": 0.85, "sq_pesq": 3.2}},
        },
        "speaker_identity": {
            "itc": {"mean": {"itc": 0.9}},
            "itd": {"itd": 0.7},
        },
        "speech_activity": {
            "overlap": {"percentage": 12.5},
        },
        "turn_taking": {
            "turn_exchanges_per_min": 10.0,
            "overlapping_transition_rate": 0.1,
            "backchannels_per_min": 2.0,
        },
        "leakage_proxy": {
            "left_to_right": {"median_db": -25.0},
            "right_to_left": {"median_db": -26.0},
        },
    }

    summary = summarize_reports([report])
    assert summary["metrics"]["nisqa_mos"] == 4.1

    rendered = render_tables(summary)
    assert "NISQA-MOS" in rendered
    assert "4.1000" in rendered
