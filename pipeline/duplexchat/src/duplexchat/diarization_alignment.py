"""Purpose: Put each diarized speaker on a fixed channel of a separated conversation.

Inputs: Two separated speaker tracks and the dialogue's diarization speaker turns.
Outputs: The same tracks with channel swaps undone, plus a report of what was swapped.

DialogueSidon can swap its output order inside a chunk, not only at chunk seams. The
split step already diarized every dialogue, so its turns tell which speaker talks
when: for each short window with a single diarized speaker, the louder channel says
where that speaker currently is. A two-state Viterbi pass turns that evidence into a
keep/swap path (``speaker_0`` -> first channel), switch points are snapped to the
quietest nearby frame, and the swap mask is crossfaded so no click is introduced.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

FRAME_SECONDS = 0.02


@dataclass
class AlignmentReport:
    windows: int
    evidence_windows: int
    swapped_seconds: float
    segments: list[tuple[float, float]]

    def to_dict(self) -> dict:
        return {
            "windows": self.windows,
            "evidence_windows": self.evidence_windows,
            "swapped_seconds": round(self.swapped_seconds, 3),
            "swapped_segments": [[round(s, 3), round(e, 3)] for s, e in self.segments],
        }


def align_to_diarization(
    first: torch.Tensor,
    second: torch.Tensor,
    sample_rate: int,
    turns: Iterable[Mapping],
    window_seconds: float = 1.0,
    min_speaker_share: float = 0.6,
    active_db: float = -45.0,
    margin_db: float = 10.0,
    switch_penalty: float = 2.0,
    snap_seconds: float = 0.5,
    fade_seconds: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, AlignmentReport]:
    """Return tracks where the first diarized speaker stays on the first channel."""
    length = min(first.shape[-1], second.shape[-1])
    tracks = torch.stack([first.reshape(-1)[:length], second.reshape(-1)[:length]]).float().cpu()
    frame = max(1, int(FRAME_SECONDS * sample_rate))
    frames = length // frame
    speakers = _speaker_frames(turns, frames)
    if frames == 0 or speakers is None:
        return first, second, AlignmentReport(0, 0, 0.0, [])
    energy = tracks[:, :frames * frame].reshape(2, frames, frame).pow(2).mean(-1)
    level_db = 10.0 * torch.log10(energy + 1e-10)

    per_window = max(1, int(window_seconds / FRAME_SECONDS))
    scores: list[float] = []
    for start in range(0, frames, per_window):
        end = min(frames, start + per_window)
        scores.append(_window_score(level_db[:, start:end], speakers[:, start:end],
                                    min_speaker_share, active_db, margin_db))
    states = _viterbi(scores, switch_penalty)

    swap = torch.zeros(frames)
    snap = int(snap_seconds / FRAME_SECONDS)
    loudness = energy.sum(0)
    previous_cut = 0
    for index in range(len(states)):
        if index and states[index] != states[index - 1]:
            target = index * per_window
            low, high = max(previous_cut, target - snap), min(frames, target + snap)
            cut = _quietest(loudness, low, high) if high > low else target
            swap[previous_cut:cut] = float(states[index - 1])
            previous_cut = cut
    swap[previous_cut:] = float(states[-1])
    segments = _segments(swap, FRAME_SECONDS)
    report = AlignmentReport(len(states), sum(score != 0.0 for score in scores),
                             sum(end - start for start, end in segments), segments)
    if not segments:
        return first, second, report

    mask = swap.repeat_interleave(frame)
    mask = torch.cat([mask, mask[-1:].expand(length - mask.numel())])
    fade = max(1, int(fade_seconds * sample_rate))
    if fade > 1:
        kernel = torch.hann_window(fade + 2, periodic=False)[1:-1]
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        padded = F.pad(mask.view(1, 1, -1), (fade // 2, fade - 1 - fade // 2), mode="replicate")
        mask = F.conv1d(padded, kernel).reshape(-1)
    fixed_first = (1.0 - mask) * tracks[0] + mask * tracks[1]
    fixed_second = (1.0 - mask) * tracks[1] + mask * tracks[0]
    shape = first.shape[:-1] + (length,)
    return fixed_first.reshape(shape).to(first.dtype), fixed_second.reshape(shape).to(second.dtype), report


def _speaker_frames(turns: Iterable[Mapping], frames: int) -> torch.Tensor | None:
    """Return a (2, frames) activity mask for the first two diarized speakers."""
    turns = [turn for turn in turns if turn.get("speaker") is not None]
    names = sorted({str(turn["speaker"]) for turn in turns})
    if len(names) < 2:
        return None
    activity = torch.zeros(2, frames, dtype=torch.bool)
    for turn in turns:
        name = str(turn["speaker"])
        if name not in names[:2]:
            continue
        start = max(0, int(float(turn["start"]) / FRAME_SECONDS))
        end = min(frames, int(float(turn["end"]) / FRAME_SECONDS))
        activity[names.index(name), start:end] = True
    return activity


def _window_score(level_db, speakers, min_share, active_db, margin_db) -> float:
    """Positive when the window says the channels are swapped, 0 without evidence."""
    frames = level_db.shape[-1]
    only = [speakers[0] & ~speakers[1], speakers[1] & ~speakers[0]]
    shares = [float(mask.sum()) / frames for mask in only]
    speaker = 0 if shares[0] >= shares[1] else 1
    if shares[speaker] < min_share:
        return 0.0
    mask = only[speaker]
    left, right = (float(10.0 * torch.log10(torch.pow(10.0, level_db[c, mask] / 10.0).mean())) for c in (0, 1))
    if max(left, right) < active_db:
        return 0.0
    # speaker_0 belongs on the first channel; evidence saturates at ``margin_db``.
    difference = max(-1.0, min(1.0, (left - right) / margin_db))
    return -difference if speaker == 0 else difference


def _quietest(loudness: torch.Tensor, low: int, high: int) -> int:
    """Middle of the quietest stretch, so the crossfade never touches speech at its edge."""
    window = loudness[low:high]
    best = int(torch.argmin(window))
    quiet = window <= window[best] * 1.26 + 1e-12  # within 1 dB of the minimum
    start = end = best
    while start > 0 and quiet[start - 1]:
        start -= 1
    while end < len(window) - 1 and quiet[end + 1]:
        end += 1
    return low + (start + end + 1) // 2


def _viterbi(scores: list[float], switch_penalty: float) -> list[int]:
    cost = [0.0, -scores[0]]
    back: list[tuple[int, int]] = []
    for score in scores[1:]:
        emit = (0.0, -score)
        choices, new_cost = [], []
        for state in (0, 1):
            stay, move = cost[state], cost[1 - state] + switch_penalty
            choices.append(state if stay <= move else 1 - state)
            new_cost.append(min(stay, move) + emit[state])
        back.append((choices[0], choices[1]))
        cost = new_cost
    state = 0 if cost[0] <= cost[1] else 1
    path = [state]
    for choice in reversed(back):
        state = choice[state]
        path.append(state)
    return path[::-1]


def _segments(swap: torch.Tensor, frame_seconds: float) -> list[tuple[float, float]]:
    segments, start = [], None
    for index, value in enumerate(swap.tolist() + [0.0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start * frame_seconds, index * frame_seconds))
            start = None
    return segments


def suppress_duplicates(
    first: torch.Tensor,
    second: torch.Tensor,
    sample_rate: int,
    turns: Iterable[Mapping],
    window_seconds: float = 0.5,
    min_correlation: float = 0.6,
    max_level_gap_db: float = 20.0,
    attenuation_db: float = -35.0,
    fade_seconds: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mute the copy of a lone speaker that the separator also wrote to the other channel.

    Call after :func:`align_to_diarization`, so ``speaker_0`` is on the first channel. Only
    windows where diarization has a single speaker and the other channel follows the
    owner's energy envelope are attenuated, so genuine backchannels are kept.
    Returns the tracks and the number of seconds that were suppressed.
    """
    length = min(first.shape[-1], second.shape[-1])
    tracks = torch.stack([first.reshape(-1)[:length], second.reshape(-1)[:length]]).float().cpu()
    frame = max(1, int(FRAME_SECONDS * sample_rate))
    frames = length // frame
    speakers = _speaker_frames(turns, frames)
    if frames == 0 or speakers is None:
        return first, second, 0.0
    envelope = tracks[:, :frames * frame].reshape(2, frames, frame).pow(2).mean(-1).sqrt()
    level_db = 20.0 * torch.log10(envelope + 1e-10)
    only = [speakers[0] & ~speakers[1], speakers[1] & ~speakers[0]]

    mute = torch.zeros(2, frames)
    per_window = max(2, int(window_seconds / FRAME_SECONDS))
    for start in range(0, frames, per_window // 2):
        end = min(frames, start + per_window)
        for owner in (0, 1):
            if float(only[owner][start:end].float().mean()) < 0.8:
                continue
            other = 1 - owner
            gap = float(level_db[owner, start:end].mean() - level_db[other, start:end].mean())
            if gap > max_level_gap_db:
                continue  # the other channel is already quiet
            a, b = envelope[owner, start:end], envelope[other, start:end]
            if float(a.std()) == 0.0 or float(b.std()) == 0.0:
                continue
            correlation = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
            if correlation >= min_correlation:
                mute[other, start:end] = 1.0
    seconds = float(mute.amax(0).sum()) * FRAME_SECONDS
    if seconds == 0.0:
        return first, second, 0.0

    gain = 1.0 - (1.0 - 10.0 ** (attenuation_db / 20.0)) * mute
    gain = gain.repeat_interleave(frame, dim=1)
    gain = torch.cat([gain, gain[:, -1:].expand(2, length - gain.shape[1])], dim=1)
    fade = max(1, int(fade_seconds * sample_rate))
    if fade > 1:
        kernel = torch.hann_window(fade + 2, periodic=False)[1:-1]
        kernel = (kernel / kernel.sum()).view(1, 1, -1).repeat(2, 1, 1)
        padded = F.pad(gain.unsqueeze(0), (fade // 2, fade - 1 - fade // 2), mode="replicate")
        gain = F.conv1d(padded, kernel, groups=2).squeeze(0)
    tracks = tracks * gain
    shape = first.shape[:-1] + (length,)
    return tracks[0].reshape(shape).to(first.dtype), tracks[1].reshape(shape).to(second.dtype), seconds
