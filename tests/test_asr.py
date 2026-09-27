import json
import hashlib
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

from core.outputs import turns_from_diarization
from core.asr.channel_resolver import duplexchat_mapping, sommelier_mapping
from core.asr.normalizer import normalize_text
from core.asr.pipeline import ASRPipeline, input_fingerprints, speaker_turns_for_asr
from core.asr.region_extractor import asr_regions, uncovered_speech
from core.asr.zipformer_engine import ZipformerEngine
from core.asr.aligner_engine import WhisperXAligner
from scripts.run_asr import _offline_preflight


def test_region_merge_gap_padding_and_clamp():
    turns = [
        {"speaker": "A", "start": 0.1, "end": 0.4},
        {"speaker": "A", "start": 0.899, "end": 1.0},
        {"speaker": "A", "start": 1.5, "end": 1.7},
        {"speaker": "B", "start": 0.6, "end": 0.7},
    ]
    # 0.499s merges before padding; the third region joins only after padding.
    assert asr_regions(turns, "A", 1.8) == [(0.0, 1.8)]
    assert asr_regions(turns[:2], "A", 2.0, pad=0) == [(0.1, 1.0)]
    exact = [{"speaker": "A", "start": 0.1, "end": 0.4}, {"speaker": "A", "start": 0.9, "end": 1.0}]
    assert asr_regions(exact, "A", 2.0, pad=0) == [(0.1, 0.4), (0.9, 1.0)]


def test_phase_one_turns_keep_same_speaker_silence():
    segments = [{"speaker": "A", "start": 0.0, "end": 0.3},
                {"speaker": "A", "start": 0.6, "end": 0.9}]
    assert len(turns_from_diarization(segments)) == 2


def test_legacy_merged_turns_recover_gaps_from_segments():
    segments = [{"speaker": "A", "start": 0.0, "end": 0.4},
                {"speaker": "A", "start": 0.8, "end": 1.0},
                {"speaker": "B", "start": 1.1, "end": 1.5}]
    original = [{"speaker": "A", "start": 0.0, "end": 1.0}, segments[2]]
    metadata = {"speaker_turns": original, "segments": segments}
    turns, source = speaker_turns_for_asr(metadata)
    assert source == "segments_legacy_recovery"
    assert turns == segments
    assert metadata["speaker_turns"] == original


def test_normalization_retains_vietnamese_tones_and_filters_by_vocabulary():
    assert normalize_text("  XIN   CHÀO! ", set("xin chào")) == "xin chào"


def test_coverage_audit_uses_unpadded_turns():
    ratio, missing = uncovered_speech([(0.0, 1.0)], [(0.2, 0.8)])
    assert ratio == pytest.approx(0.6)
    assert missing[0] == pytest.approx((0.0, 0.2))
    assert missing[1] == pytest.approx((0.8, 1.0))
    assert uncovered_speech([], [(0.0, 1.0)]) == (None, [])


class _SignedEmbedding:
    def extract(self, wav, sample_rate):
        mean = float(wav.mean())
        return torch.tensor([max(mean, 0.0), max(-mean, 0.0)])


def test_duplexchat_mapping_matches_embedded_voices_not_channel_index(tmp_path):
    rate = 24_000
    mixture = np.r_[np.full(rate, 0.5), np.full(rate, -0.5)]
    stereo = np.column_stack([np.r_[np.zeros(rate), np.full(rate, -0.5)],
                              np.r_[np.full(rate, 0.5), np.zeros(rate)]])
    path = tmp_path / "dialogue_1.wav"
    sf.write(path, mixture, rate)
    turns = [{"speaker": "A", "start": 0.0, "end": 1.0},
             {"speaker": "B", "start": 1.0, "end": 2.0}]
    mapping = duplexchat_mapping(stereo.astype("float32"), rate, path, turns, _SignedEmbedding())
    assert mapping["channel_speakers"] == ["B", "A"]
    with pytest.raises(ValueError, match="Uncertain"):
        duplexchat_mapping(stereo.astype("float32"), rate, path, turns, _SignedEmbedding(), min_margin=1.1)


def test_sommelier_mapping_uses_reconstruction_sidecar(tmp_path):
    stereo = tmp_path / "stereo_1.wav"
    stereo.with_suffix(".channels.json").write_text('{"channel_speakers":["B","A"]}')
    assert sommelier_mapping(stereo, {"A", "B"})["channel_speakers"] == ["B", "A"]


def test_server_preflight_requires_local_models(monkeypatch, tmp_path):
    monkeypatch.setenv("SILERO_VAD_MODEL_PATH", str(tmp_path / "missing-vad"))
    monkeypatch.setenv("SPEECHBRAIN_MODEL_PATH", str(tmp_path / "missing-speaker"))
    cfg = {"zipformer": {"local_dir": str(tmp_path / "missing-zip"),
                         "encoder": "encoder.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx", "bpe_model": "bpe.model"},
           "aligner": {"local_dir": str(tmp_path / "missing-aligner")}}
    with pytest.raises(FileNotFoundError, match="ASR offline preflight failed") as error:
        _offline_preflight(cfg, "duplexchat")
    assert "missing-zip" in str(error.value) and "missing-aligner" in str(error.value)


@pytest.mark.parametrize("offline,method", [(False, "greedy_search"), (True, "modified_beam_search")])
def test_zipformer_model_loading_respects_environment(monkeypatch, tmp_path, offline, method):
    captured = {}
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "bpe.model"):
        (tmp_path / name).write_bytes(b"model")

    class Tokenizer:
        def __init__(self, model_file):
            captured["bpe"] = model_file

        def id_to_piece(self, index):
            return ("<blk>", "▁a")[index]

        def vocab_size(self):
            return 2

    class Recognizer:
        @staticmethod
        def from_transducer(**kwargs):
            captured["recognizer"] = kwargs
            return object()

    monkeypatch.setattr("sentencepiece.SentencePieceProcessor", Tokenizer)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", types.SimpleNamespace(OfflineRecognizer=Recognizer))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: captured.setdefault("download", kwargs) and str(tmp_path))
    cfg = {"model_id": "hynt/Zipformer-30M-RNNT-6000h", "local_dir": str(tmp_path),
           "encoder": "encoder.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx", "bpe_model": "bpe.model",
           "decoding_method": method, "max_active_paths": 8}
    engine = ZipformerEngine(cfg, offline)
    assert captured["recognizer"]["tokens"].endswith("tokens.txt")
    assert ("download" in captured) is not offline
    assert captured["recognizer"]["sample_rate"] == 16_000
    assert captured["recognizer"]["decoding_method"] == method
    assert captured["recognizer"]["max_active_paths"] == 8
    engine._temporary.cleanup()


def test_whisperx_offline_path_and_nullable_score(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setitem(sys.modules, "whisperx", types.SimpleNamespace(
        load_align_model=lambda **kwargs: (captured.setdefault("name", kwargs["model_name"]) and object(),
                                           {"dictionary": {"a": 0, "b": 1}}),
        align=lambda *args, **kwargs: {"word_segments": [{"word": "ab", "start": 0.1, "end": 0.2}]},
    ))
    aligner = WhisperXAligner({"model_id": "model/id", "local_dir": str(tmp_path)}, True, "cpu")
    text, words, status, error = aligner.align("AB", np.zeros(16_000, dtype="float32"))
    assert captured["name"] == str(tmp_path)
    assert (text, status, error) == ("ab", "aligned", None)
    assert words == [{"word": "ab", "start": 0.1, "end": 0.2, "score": None, "aligned": True}]


class _FakeASR:
    model_id = "fake-zipformer"
    decoding_method = "greedy_search"
    max_active_paths = 4

    def transcribe(self, audio):
        return "Xin chào"


class _FakeAligner:
    model_id = "fake-aligner"

    def align(self, text, audio):
        return "xin chào", [
            {"word": "xin", "start": 0.1, "end": 0.3, "score": None, "aligned": True},
            {"word": "chào", "start": None, "end": None, "score": None, "aligned": False},
        ], "partial", None


def test_stereo_sample_end_to_end_projects_words_and_keeps_partial_alignment(tmp_path):
    rate = 24_000
    dialogue = tmp_path / "dialogue_1.json"
    turns = [{"speaker": "A", "start": 0.5, "end": 1.0},
             {"speaker": "B", "start": 1.5, "end": 2.0}]
    dialogue.write_text(json.dumps({"speaker_turns": turns}))
    stereo = tmp_path / "stereo_1.wav"
    sf.write(stereo, np.zeros((rate * 3, 2), dtype="float32"), rate)
    stereo.with_suffix(".channels.json").write_text('{"channel_speakers":["B","A"]}')
    engine = ASRPipeline.__new__(ASRPipeline)
    engine.cfg = {"regions": {"merge_gap_seconds": 0.5, "pad_seconds": 0.25}}
    engine.asr = _FakeASR()
    engine.aligner = _FakeAligner()
    engine.vad_model = object()
    engine.vad_timestamps = lambda audio, model, **kwargs: [{"start": 0, "end": 16_000 * 3}]
    engine.config_signature = "test-config"

    result = engine.process(stereo, tmp_path, "sommelier")
    assert result["channels"][0]["speaker"] == "B"
    assert result["channels"][0]["regions"][0]["asr_text_raw"] == "Xin chào"
    assert result["channels"][0]["regions"][0]["aligned_word_ratio"] == 0.5
    assert result["channels"][0]["regions"][0]["words"][1]["score"] is None
    assert result["channels"][0]["regions"][0]["words"][1]["aligned"] is False
    assert result["merged_timeline"][0]["start"] == pytest.approx(0.35)
    assert result["merged_timeline"][1]["start"] == pytest.approx(1.35)
    assert [row["start"] for row in result["merged_timeline"]] == sorted(row["start"] for row in result["merged_timeline"])
    assert result["coverage_ratio"] == pytest.approx(1 / 6)


def test_batch_resume_skips_model_loading_for_matching_stereo(monkeypatch, tmp_path):
    from scripts import run_asr

    stereo_root = tmp_path / "stereo"
    stereo_root.mkdir()
    stereo = stereo_root / "stereo_1.wav"
    sf.write(stereo, np.zeros((24_000, 2), dtype="float32"), 24_000)
    stereo.with_suffix(".channels.json").write_text('{"channel_speakers":["A","B"]}')
    dialogue_dir = tmp_path / "dialogue"
    dialogue_dir.mkdir()
    (dialogue_dir / "dialogue_1.json").write_text('{"speaker_turns":[]}')
    cfg = {"model": "unchanged"}
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    fingerprints = input_fingerprints(stereo, dialogue_dir, "sommelier")
    (stereo_root / "dialogue_1.asr.json").write_text(json.dumps({"provenance": {
        **fingerprints, "config_sha256": signature,
    }}))
    monkeypatch.setattr(run_asr, "ASRPipeline", lambda *args: pytest.fail("resumed run loaded models"))
    monkeypatch.setattr(sys, "argv", ["run_asr.py", "--pipeline", "sommelier", "--stereo-root", str(stereo_root),
                                  "--dialogue-root", str(dialogue_dir), "--config-json", json.dumps(cfg)])
    assert run_asr.run() == 0
    assert json.loads((stereo_root / "asr_report.json").read_text())["items"][0]["status"] == "resumed"
    (dialogue_dir / "dialogue_1.json").write_text('{"speaker_turns":[{"speaker":"A"}]}')
    assert input_fingerprints(stereo, dialogue_dir, "sommelier")["dialogue_sha256"] != fingerprints["dialogue_sha256"]

    class FakeEngine:
        def process(self, path, source_dir, pipeline_name):
            return {"provenance": {**input_fingerprints(path, source_dir, pipeline_name),
                                   "config_sha256": signature}}

    monkeypatch.setattr(run_asr, "ASRPipeline", lambda *args: FakeEngine())
    assert run_asr.run() == 0
    assert json.loads((stereo_root / "asr_report.json").read_text())["items"][0]["status"] == "complete"
    assert len(list((stereo_root / ".history").rglob("dialogue_1.asr.json"))) == 1
