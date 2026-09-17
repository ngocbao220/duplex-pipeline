from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "pipeline" / "vilier" / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vilier.diarization import SortformerDiarizer, build_silence_diarization_chunks  # noqa: E402
from vilier.preprocess import sommelier_normalize  # noqa: E402
from vilier.schema import SpeakerSegment  # noqa: E402
from vilier.separation import PyannoteOverlapAssigner, apply_overlap_separation  # noqa: E402


def test_sommelier_normalize_peak_normalizes_non_silent_waveform():
    waveform = np.array([-0.25, 0.5, -1.0], dtype=np.float32)

    normalized = sommelier_normalize(waveform)

    np.testing.assert_allclose(normalized, waveform)


def test_sommelier_normalize_preserves_silence():
    waveform = np.zeros(8, dtype=np.float32)

    normalized = sommelier_normalize(waveform)

    np.testing.assert_array_equal(normalized, waveform)


def test_silence_chunking_cuts_at_silence_midpoint_and_preserves_source_time(tmp_path):
    waveform = np.arange(100, dtype=np.float32) / 100
    speech = [{"start": 0.0, "end": 2.0}, {"start": 4.0, "end": 10.0}]

    chunks = build_silence_diarization_chunks(
        waveform, sample_rate=10, speech_segments=speech, output_dir=tmp_path,
        max_chunk_seconds=7.0, min_silence_seconds=0.3,
    )

    assert [(chunk["source_start"], chunk["source_end"]) for chunk in chunks] == [(0.0, 3.0), (3.0, 10.0)]
    assert chunks[0]["mapping"] == [{"chunk_start": 0.0, "chunk_end": 3.0, "source_start": 0.0, "source_end": 3.0}]


def test_sortformer_initializes_configured_speaker_inventory_before_chunk_clustering():
    diarizer = SortformerDiarizer({"num_speakers": 2}, dry_run=True)

    assert diarizer.num_speakers == 2


def test_embedding_assignment_replaces_energy_heuristic_for_overlap_source_order():
    class FakeSeparator:
        def separate(self, audio, sample_rate):
            return np.full_like(audio, 0.8), np.full_like(audio, 0.2)

    class FakeAssigner:
        def reference_embeddings(self, segments, waveform, sample_rate, pairs):
            return {"SPEAKER_00": np.array([1.0]), "SPEAKER_01": np.array([0.0])}

        def assign(self, seg1, seg2, src1, src2, sample_rate, references):
            return src2, src1

    waveform = np.full(40, 0.1, dtype=np.float32)
    segments = [
        SpeakerSegment("a", "SPEAKER_00", 0.0, 3.0),
        SpeakerSegment("b", "SPEAKER_01", 1.0, 2.0),
    ]

    result = apply_overlap_separation(
        waveform, 10, segments, FakeSeparator(), overlap_threshold=0.1, speaker_assigner=FakeAssigner()
    )

    np.testing.assert_allclose(result["segment_audio"]["a"][10:20], 0.1, atol=1e-6)
    np.testing.assert_allclose(result["segment_audio"]["b"], 0.1, atol=1e-6)


def test_overlap_reconstruction_rejects_unverified_channel_order():
    class FakeSeparator:
        def separate(self, audio, sample_rate):
            return np.full_like(audio, 0.8), np.full_like(audio, 0.2)

    segments = [
        SpeakerSegment("a", "SPEAKER_00", 0.0, 3.0),
        SpeakerSegment("b", "SPEAKER_01", 1.0, 2.0),
    ]
    with pytest.raises(RuntimeError, match="requires pyannote speaker embeddings"):
        apply_overlap_separation(np.full(40, 0.1, dtype=np.float32), 10, segments, FakeSeparator(), 0.1)


def test_pyannote_assigner_rejects_short_overlap_audio_without_verified_identity():
    assigner = PyannoteOverlapAssigner.__new__(PyannoteOverlapAssigner)
    assigner._embedding = lambda audio, sample_rate: None
    first = np.array([0.2], dtype=np.float32)
    second = np.array([0.8], dtype=np.float32)
    seg1 = SpeakerSegment("a", "SPEAKER_00", 0.0, 1.0)
    seg2 = SpeakerSegment("b", "SPEAKER_01", 0.0, 1.0)

    with pytest.raises(RuntimeError, match="too short"):
        assigner.assign(seg1, seg2, first, second, 16000, {})


def test_pyannote_assigner_rejects_missing_reference_instead_of_guessing_source_order():
    assigner = PyannoteOverlapAssigner.__new__(PyannoteOverlapAssigner)
    assigner._embedding = lambda audio, sample_rate: np.array([1.0], dtype=np.float32)
    first = np.array([0.2], dtype=np.float32)
    second = np.array([0.8], dtype=np.float32)
    seg1 = SpeakerSegment("a", "SPEAKER_00", 0.0, 1.0)
    seg2 = SpeakerSegment("b", "SPEAKER_01", 0.0, 1.0)

    with pytest.raises(RuntimeError, match="no speaker reference"):
        assigner.assign(seg1, seg2, first, second, 16000, {})
