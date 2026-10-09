"""Purpose: Keep each speaker on one channel across a whole separated conversation.

Inputs: Two separated speaker tracks and an embedding extractor.
Outputs: The same tracks with per-block channel swaps undone.

DialogueSidon only guarantees a consistent speaker order inside a short context: within
one 120 s chunk the order can drift, and chunk seams can pick the wrong permutation. This
module splits the timeline into blocks cut at joint silences, embeds each channel per
block, clusters the embeddings into two speakers and picks the swap/keep path with a
two-state Viterbi pass so isolated low-confidence blocks cannot flip the output.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio.functional as F_audio

ANALYSIS_SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.02

Embedder = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ConsistencyReport:
    blocks: int
    swapped_blocks: int
    swapped_seconds: float
    segments: list[tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "blocks": self.blocks,
            "swapped_blocks": self.swapped_blocks,
            "swapped_seconds": round(self.swapped_seconds, 3),
            "swapped_segments": [[round(s, 3), round(e, 3)] for s, e in self.segments],
        }


def enforce_speaker_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
    sample_rate: int,
    embedder: Embedder,
    block_seconds: float = 4.0,
    snap_seconds: float = 1.0,
    min_active_seconds: float = 0.8,
    active_db: float = -40.0,
    switch_penalty: float = 0.3,
    iterations: int = 4,
    fade_seconds: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, ConsistencyReport]:
    """Return tracks whose channel order is consistent per speaker over the full timeline."""
    length = min(first.shape[-1], second.shape[-1])
    tracks = torch.stack([first.reshape(-1)[:length], second.reshape(-1)[:length]]).float().cpu()
    analysis = tracks if sample_rate == ANALYSIS_SAMPLE_RATE else F_audio.resample(
        tracks, sample_rate, ANALYSIS_SAMPLE_RATE)

    frame = int(FRAME_SECONDS * ANALYSIS_SAMPLE_RATE)
    frames = analysis.shape[-1] // frame
    if frames == 0:
        return first, second, ConsistencyReport(0, 0, 0.0, [])
    energy = analysis[:, :frames * frame].reshape(2, frames, frame).pow(2).mean(-1)
    level_db = 10.0 * torch.log10(energy + 1e-10)
    peak_db = float(level_db.max())
    active = level_db > max(active_db, peak_db - 45.0)

    bounds = _block_bounds(energy.sum(0), block_seconds, snap_seconds)
    min_active_frames = int(min_active_seconds / FRAME_SECONDS)
    embeddings: list[list[torch.Tensor | None]] = []
    for start, end in zip(bounds, bounds[1:]):
        row: list[torch.Tensor | None] = []
        for channel in (0, 1):
            mask = active[channel, start:end]
            if int(mask.sum()) < min_active_frames:
                row.append(None)
                continue
            audio = analysis[channel, start * frame:end * frame].reshape(-1, frame)[mask].reshape(-1)
            row.append(F.normalize(embedder(audio).reshape(-1).float().cpu(), dim=0))
        embeddings.append(row)

    states = _decode_permutation(embeddings, switch_penalty, iterations)
    if states is None:
        return first, second, ConsistencyReport(len(embeddings), 0, 0.0, [])

    scale = sample_rate / ANALYSIS_SAMPLE_RATE
    swap = torch.zeros(length)
    segments: list[tuple[float, float]] = []
    for index, state in enumerate(states):
        if not state:
            continue
        start = int(round(bounds[index] * frame * scale))
        end = length if index == len(states) - 1 else int(round(bounds[index + 1] * frame * scale))
        swap[start:end] = 1.0
        if segments and abs(segments[-1][1] - start / sample_rate) < 1e-6:
            segments[-1] = (segments[-1][0], end / sample_rate)
        else:
            segments.append((start / sample_rate, end / sample_rate))
    if not segments:
        return first, second, ConsistencyReport(len(states), 0, 0.0, [])

    fade = max(1, int(fade_seconds * sample_rate))
    if fade > 1:
        kernel = torch.hann_window(fade + 2, periodic=False)[1:-1]
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        swap = F.conv1d(swap.view(1, 1, -1), kernel, padding=fade // 2).reshape(-1)[:length]
    fixed_first = (1.0 - swap) * tracks[0] + swap * tracks[1]
    fixed_second = (1.0 - swap) * tracks[1] + swap * tracks[0]
    report = ConsistencyReport(
        blocks=len(states),
        swapped_blocks=int(sum(states)),
        swapped_seconds=sum(end - start for start, end in segments),
        segments=segments,
    )
    shape = first.shape[:-1] + (length,)
    return fixed_first.reshape(shape), fixed_second.reshape(shape), report


def _block_bounds(total_energy: torch.Tensor, block_seconds: float, snap_seconds: float) -> list[int]:
    """Cut roughly every ``block_seconds`` at the quietest frame within ``snap_seconds``."""
    frames = total_energy.shape[0]
    step = max(1, int(block_seconds / FRAME_SECONDS))
    snap = max(0, int(snap_seconds / FRAME_SECONDS))
    bounds = [0]
    target = step
    while target < frames - step // 2:
        low = max(bounds[-1] + step // 2, target - snap)
        high = min(frames - 1, target + snap)
        if high <= low:
            break
        cut = low + int(torch.argmin(total_energy[low:high]))
        bounds.append(cut)
        target = cut + step
    bounds.append(frames)
    return bounds


def _decode_permutation(
    embeddings: list[list[torch.Tensor | None]],
    switch_penalty: float,
    iterations: int,
) -> list[int] | None:
    """Cluster per-block embeddings into two speakers and Viterbi-decode the swap path."""
    all_embeddings = [emb for row in embeddings for emb in row if emb is not None]
    if len(all_embeddings) < 2:
        return None
    centroids = _init_centroids(embeddings, all_embeddings)
    states = [0] * len(embeddings)
    for _ in range(iterations):
        scores = [_swap_score(row, centroids) for row in embeddings]
        states = _viterbi(scores, switch_penalty)
        centroids = _update_centroids(embeddings, states, centroids)

    # Orient so the dominant channel order (by evidence) stays unchanged.
    kept = sum(1 for row, state in zip(embeddings, states) if not state and any(e is not None for e in row))
    swapped = sum(1 for row, state in zip(embeddings, states) if state and any(e is not None for e in row))
    if swapped > kept:
        states = [1 - state for state in states]
    return states


def _init_centroids(embeddings, all_embeddings) -> tuple[torch.Tensor, torch.Tensor]:
    # Prefer a block where both channels speak: the separator's own split there is the
    # most reliable statement of "these are two different people".
    best = None
    for row in embeddings:
        if row[0] is not None and row[1] is not None:
            similarity = float(row[0] @ row[1])
            if best is None or similarity < best[0]:
                best = (similarity, row[0], row[1])
    if best is not None:
        return best[1], best[2]
    stacked = torch.stack(all_embeddings)
    first = stacked[0]
    second = stacked[int(torch.argmin(stacked @ first))]
    return first, second


def _swap_score(row, centroids) -> float:
    """Positive when swapping the block's channels matches the centroids better."""
    score = 0.0
    for channel, emb in enumerate(row):
        if emb is None:
            continue
        own = float(emb @ centroids[channel])
        other = float(emb @ centroids[1 - channel])
        score += other - own
    return score


def _viterbi(scores: list[float], switch_penalty: float) -> list[int]:
    cost = [0.0, -scores[0]] if scores else [0.0, 0.0]
    back: list[tuple[int, int]] = []
    for score in scores[1:]:
        emit = (0.0, -score)
        choices = []
        new_cost = []
        for state in (0, 1):
            stay = cost[state]
            move = cost[1 - state] + switch_penalty
            if stay <= move:
                choices.append(state)
                new_cost.append(stay + emit[state])
            else:
                choices.append(1 - state)
                new_cost.append(move + emit[state])
        back.append((choices[0], choices[1]))
        cost = new_cost
    state = 0 if cost[0] <= cost[1] else 1
    path = [state]
    for choice in reversed(back):
        state = choice[state]
        path.append(state)
    return list(reversed(path))


def _update_centroids(embeddings, states, previous) -> tuple[torch.Tensor, torch.Tensor]:
    groups: list[list[torch.Tensor]] = [[], []]
    for row, state in zip(embeddings, states):
        for channel, emb in enumerate(row):
            if emb is not None:
                groups[channel ^ state].append(emb)
    return tuple(  # type: ignore[return-value]
        F.normalize(torch.stack(group).mean(0), dim=0) if group else previous[index]
        for index, group in enumerate(groups)
    )
