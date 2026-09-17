from __future__ import annotations

import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat.dialogue import Dialogue, _split_long_dialogue, extract_valid_dialogues, split_into_dialogues  # noqa: E402
from duplexchat.diarization_backend import GlobalSpeakerLinker  # noqa: E402
from duplexchat import diarization as diarization_phase  # noqa: E402


def _segment(speaker: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "start": start, "end": end}


def test_overlapping_turn_uses_latest_active_end_for_grouping_and_crop_end():
    segments = [
        _segment("A", 40.582, 54.233),
        _segment("B", 52.394, 52.500),
        _segment("A", 58.381, 59.000),
    ]

    dialogues = split_into_dialogues(segments, gap_seconds=5.0)

    assert len(dialogues) == 1
    assert dialogues[0].end == 59.0
    assert dialogues[0].duration == 18.418


def test_long_dialogue_splits_at_dual_silence_near_target():
    dialogue = Dialogue(
        segments=[
            _segment("A", 0.0, 599.0),
            _segment("B", 600.0, 601.0),
            _segment("A", 601.0, 900.0),
        ],
        start=0.0,
        end=900.0,
    )

    chunks = _split_long_dialogue(dialogue, max_duration=600.0, min_duration=10.0)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 599.0), (600.0, 900.0)]


def test_long_dialogue_is_not_cut_without_a_dual_silence_boundary():
    dialogue = Dialogue(
        segments=[
            _segment("A", 0.0, 700.0),
            _segment("B", 0.0, 700.0),
        ],
        start=0.0,
        end=700.0,
    )

    assert _split_long_dialogue(dialogue, max_duration=600.0, min_duration=10.0) == [dialogue]


def test_long_dialogue_expands_the_silence_search_when_target_window_has_none():
    dialogue = Dialogue(
        segments=[
            _segment("A", 0.0, 100.0),
            _segment("B", 700.0, 1_300.0),
        ],
        start=0.0,
        end=1_300.0,
    )

    chunks = _split_long_dialogue(dialogue, max_duration=600.0, min_duration=10.0)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 100.0), (700.0, 1_300.0)]


def test_long_dialogue_keeps_a_short_tail_after_a_silence_boundary():
    dialogue = Dialogue(
        segments=[
            _segment("A", 0.0, 599.5),
            _segment("B", 600.0, 605.0),
        ],
        start=0.0,
        end=605.0,
    )

    chunks = _split_long_dialogue(dialogue, max_duration=600.0, min_duration=10.0)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 599.5), (600.0, 605.0)]


def test_valid_dialogues_drop_short_chunks_created_by_long_split():
    segments = [
        _segment("A", 0.0, 300.0),
        _segment("B", 300.0, 599.5),
        _segment("A", 600.0, 602.0),
        _segment("B", 602.0, 605.0),
    ]

    dialogues = extract_valid_dialogues(segments)

    assert [(dialogue.start, dialogue.end) for dialogue in dialogues] == [(0.0, 599.5)]


class _MeanEmbeddingExtractor:
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return torch.tensor([wav.float().mean()])


def test_global_linker_keeps_one_identity_when_chunk_local_labels_reset():
    linker = GlobalSpeakerLinker(_MeanEmbeddingExtractor(), similarity_threshold=0.7)
    segment = [_segment("SPEAKER_00", 0.0, 1.0)]

    first = linker.link(segment, torch.ones((1, 16_000)), 16_000)
    second = linker.link([_segment("SPEAKER_01", 0.0, 1.0)], torch.ones((1, 16_000)), 16_000)

    assert first == {"SPEAKER_00": "SPEAKER_00"}
    assert second == {"SPEAKER_01": "SPEAKER_00"}
    assert linker.diagnostics()["embedding_labels"] == 2
    assert linker.diagnostics()["matched_labels"] == 1
    assert linker.diagnostics()["new_labels"] == 1


def test_global_linker_never_merges_distinct_local_speakers_in_one_chunk():
    linker = GlobalSpeakerLinker(_MeanEmbeddingExtractor(), similarity_threshold=0.7)
    local_speakers = [
        _segment("SPEAKER_00", 0.0, 1.0),
        _segment("SPEAKER_01", 1.0, 2.0),
    ]

    mapping = linker.link(local_speakers, torch.ones((1, 32_000)), 16_000)

    assert mapping["SPEAKER_00"] != mapping["SPEAKER_01"]
    assert linker.diagnostics()["global_speakers"] == 2


def test_global_linker_caps_a_two_person_recording_at_two_global_speakers():
    linker = GlobalSpeakerLinker(_MeanEmbeddingExtractor(), similarity_threshold=0.7)
    first_chunk = [
        _segment("SPEAKER_00", 0.0, 1.0),
        _segment("SPEAKER_01", 1.0, 2.0),
    ]
    noisy_chunk = [
        _segment("SPEAKER_00", 0.0, 1.0),
        _segment("SPEAKER_01", 1.0, 2.0),
        _segment("SPEAKER_02", 2.0, 3.0),
    ]

    linker.link(first_chunk, torch.ones((1, 32_000)), 16_000)
    mapping = linker.link(noisy_chunk, torch.ones((1, 48_000)), 16_000)

    assert set(mapping.values()) <= {"SPEAKER_00", "SPEAKER_01"}
    assert linker.diagnostics()["global_speakers"] == 2


def test_diarization_returns_segments_without_writing_legacy_phase_artifacts(monkeypatch, tmp_path):
    pipeline = object()
    diagnostics = {"method": "speechbrain_ecapa_cosine", "embedding_labels": 2}

    monkeypatch.setattr(diarization_phase, "load_diarization_pipeline", lambda *_args, **_kwargs: pipeline)

    # The fake replaces runtime diagnostics with deterministic linker details.
    def fake_run(*_args, diagnostics=None, **_kwargs):
        diagnostics.update({"method": "speechbrain_ecapa_cosine", "embedding_labels": 2})
        return []

    monkeypatch.setattr(diarization_phase, "run_diarization", fake_run)
    result, segments = diarization_phase.diarize(
        tmp_path / "audio.wav", tmp_path, "model", "pyannote", "cpu", 60.0, lambda *_args: None
    )

    assert result is pipeline
    assert segments == []
    assert not (tmp_path / "phase_02_diarization").exists()


def test_extract_valid_dialogues_includes_reason():
    segments_standard = [
        _segment("A", 0.0, 6.0),
        _segment("B", 6.0, 12.0),
    ]
    dialogues_std = extract_valid_dialogues(segments_standard)
    assert len(dialogues_std) == 1
    assert "accepted_standard_2_speaker_dialogue" in dialogues_std[0].reason

    segments_long = [
        _segment("A", 0.0, 300.0),
        _segment("B", 300.0, 599.0),
        _segment("A", 600.0, 900.0),
        _segment("B", 900.0, 1200.0),
    ]
    dialogues_long = extract_valid_dialogues(segments_long, max_duration_seconds=600.0)
    assert len(dialogues_long) == 2
    assert "split_from_long_dialogue" in dialogues_long[0].reason
    assert "chunk 1/2" in dialogues_long[0].reason
    assert "split_from_long_dialogue" in dialogues_long[1].reason
    assert "chunk 2/2" in dialogues_long[1].reason

