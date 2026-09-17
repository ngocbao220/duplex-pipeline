from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "cholimex" / "src"))

from cholimex.speaker_assignment import assign_candidates  # noqa: E402
from cholimex.models import Region  # noqa: E402
from cholimex.reconstruct import reconstruct_tracks  # noqa: E402


class IdentityEmbedding:
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return wav.reshape(-1)


def test_overlap_assignment_records_direct_and_swapped_cosine_scores():
    extractor = IdentityEmbedding()
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])
    references = {0: first.reshape(-1), 1: second.reshape(-1)}

    assigned, record = assign_candidates(first, second, 16000, references, extractor, 0.5)

    assert torch.equal(assigned[0], first)
    assert record["mapping"] == {"candidate_0": 0, "candidate_1": 1}
    assert record["direct_score"] == pytest.approx(1.0)
    assert record["swapped_score"] == pytest.approx(0.0)
    assert record["candidate_scores"]["candidate_0"]["speaker_0"] == pytest.approx(1.0)
    assert record["margin"] == pytest.approx(1.0)


def test_overlap_assignment_swaps_only_when_embedding_evidence_supports_it():
    extractor = IdentityEmbedding()
    first = torch.tensor([[0.0, 1.0]])
    second = torch.tensor([[1.0, 0.0]])
    references = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}

    assigned, record = assign_candidates(first, second, 16000, references, extractor, 0.5)

    assert torch.equal(assigned[0], second)
    assert record["swapped"] is True
    assert record["mapping"] == {"candidate_0": 1, "candidate_1": 0}


def test_relative_assignment_keeps_best_mapping_even_when_absolute_score_is_low():
    extractor = IdentityEmbedding()
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])

    assigned, record = assign_candidates(
        first, second, 16000,
        {0: torch.tensor([-1.0, -1.0]), 1: torch.tensor([-1.0, -2.0])}, extractor, 0.5,
    )

    assert record["score"] < 0.0
    assert record["mode"] == "relative_similarity"


def test_strict_assignment_keeps_the_legacy_threshold_failure():
    extractor = IdentityEmbedding()
    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])

    with pytest.raises(RuntimeError, match="references for both speakers"):
        assign_candidates(first, second, 16000, {0: first.reshape(-1)}, extractor, 0.5)
    with pytest.raises(RuntimeError, match="below threshold"):
        assign_candidates(first, second, 16000, {0: -first.reshape(-1), 1: -second.reshape(-1)}, extractor, 0.5,
                          mode="strict_threshold")


def test_reconstruction_debug_callback_receives_raw_candidates_and_assignment_scores():
    original = torch.tensor([[0.25, 0.5]])
    candidate_0 = torch.tensor([[1.0, 0.0]])
    candidate_1 = torch.tensor([[0.0, 1.0]])
    captured = {}

    def separator(_wav, _sample_rate):
        return candidate_0, candidate_1, 2

    def debug(record, mixture, raw_0, raw_1, assigned_0, assigned_1, sample_rate):
        captured.update(record=record, mixture=mixture, raw_0=raw_0, raw_1=raw_1,
                        assigned_0=assigned_0, assigned_1=assigned_1, sample_rate=sample_rate)

    reconstruct_tracks(
        original, 2, [Region(0.0, 1.0, "overlap")], separator,
        {0: candidate_0.reshape(-1), 1: candidate_1.reshape(-1)}, IdentityEmbedding(),
        0.5, "relative_similarity", 0.0, debug,
    )

    assert torch.equal(captured["raw_0"], candidate_0)
    assert torch.equal(captured["assigned_1"], candidate_1)
    assert captured["record"]["assignment"]["candidate_scores"]
