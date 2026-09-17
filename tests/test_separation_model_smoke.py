"""Contract tests for the standalone separation-model smoke runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "test_separation_models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_separation_models", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_selection_accepts_all_and_rejects_unknown_models():
    module = _load_module()

    assert set(module.select_models("all")) == set(module.MODELS)
    assert module.select_models("sepformer-wsj02mix,sepreformer-base-wsj0") == [
        "sepformer-wsj02mix",
        "sepreformer-base-wsj0",
    ]

    try:
        module.select_models("not-a-model")
    except ValueError as exc:
        assert "Unknown model" in str(exc)
    else:
        raise AssertionError("unknown model must be rejected")


def test_normalize_sources_preserves_source_count_and_length():
    module = _load_module()
    sources = np.arange(30, dtype=np.float32).reshape(3, 10)

    normalized = module.normalize_sources(sources, expected_length=16)

    assert normalized.shape == (3, 16)
    assert np.array_equal(normalized[:, :10], sources)
    assert np.all(normalized[:, 10:] == 0)


def test_complete_report_writes_two_track_aliases_only_for_two_source_model(tmp_path):
    module = _load_module()
    paths = module.write_outputs(
        tmp_path,
        np.zeros(16, dtype=np.float32),
        16_000,
        np.zeros((3, 16), dtype=np.float32),
    )
    report = module.build_report("rahma89-voice-separation-model", paths, 3, 0.2)
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    assert not (tmp_path / "audio.stereo.wav").exists()
    assert len(list(tmp_path.glob("source_*.wav"))) == 3
    assert report["two_track_output"] is False


def test_two_source_model_writes_one_stereo_output(tmp_path):
    module = _load_module()

    paths = module.write_outputs(
        tmp_path,
        np.zeros(16, dtype=np.float32),
        16_000,
        np.zeros((2, 16), dtype=np.float32),
    )

    assert paths["stereo"] == "audio.stereo.wav"
    info = sf.info(tmp_path / "audio.stereo.wav")
    assert info.channels == 2


def test_overlap_manifest_resolves_debug_mixtures_in_manifest_order(tmp_path):
    module = _load_module()
    debug_dir = tmp_path / "debug"
    first = debug_dir / "overlaps" / "overlap_00002"
    second = debug_dir / "overlaps" / "overlap_00001"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "mixture.wav").touch()
    (second / "mixture.wav").touch()
    manifest = debug_dir / "overlaps.json"
    manifest.write_text(json.dumps([
        {"id": "overlap_00002", "start": 2.0, "end": 3.0},
        {"id": "overlap_00001", "start": 1.0, "end": 2.0},
    ]), encoding="utf-8")

    overlaps = module.load_overlap_inputs(manifest)

    assert [item["id"] for item in overlaps] == ["overlap_00002", "overlap_00001"]
    assert [item["path"] for item in overlaps] == [first / "mixture.wav", second / "mixture.wav"]


def test_mossformer2_source_accepts_the_upstream_checkout_or_standalone_subdir(tmp_path):
    module = _load_module()
    standalone = tmp_path / "MossFormer2_standalone"
    (standalone / "model").mkdir(parents=True)
    (standalone / "model" / "mossformer2.py").touch()

    assert module.resolve_mossformer2_source(tmp_path) == standalone
    assert module.resolve_mossformer2_source(standalone) == standalone
