"""Purpose: Group diarization turns into valid DuplexChat dialogues.

Inputs: Speaker-labelled diarization segments and timing thresholds.
Outputs: Dialogue records suitable for reporting and separation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Dialogue:
    segments: list[dict]
    start: float
    end: float
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speakers(self) -> list[str]:
        return list({s["speaker"] for s in self.segments})


def split_into_dialogues(segments: list[dict], gap_seconds: float) -> list[Dialogue]:
    """Group consecutive diarization segments; split when the gap between turns >= gap_seconds."""
    if not segments:
        return []

    segments = sorted(segments, key=lambda segment: (segment["start"], segment["end"]))
    groups: list[list[dict]] = []
    current: list[dict] = [segments[0]]
    active_end = segments[0]["end"]

    for seg in segments[1:]:
        if seg["start"] - active_end >= gap_seconds:
            groups.append(current)
            current = [seg]
            active_end = seg["end"]
        else:
            current.append(seg)
            active_end = max(active_end, seg["end"])
    groups.append(current)

    return [_dialogue_from_segments(group) for group in groups]


def _dialogue_from_segments(segments: list[dict]) -> Dialogue:
    return Dialogue(
        segments=segments,
        start=min(segment["start"] for segment in segments),
        end=max(segment["end"] for segment in segments),
    )


def _two_speaker_runs(segments: list[dict]) -> list[list[dict]]:
    """
    Find all maximal contiguous sub-sequences of turns that involve exactly
    2 distinct speakers. When a 3rd speaker appears, the current run is closed
    and a new one begins with just that speaker.
    """
    if not segments:
        return []

    runs: list[list[dict]] = []
    current: list[dict] = [segments[0]]
    speakers: set[str] = {segments[0]["speaker"]}

    for seg in segments[1:]:
        if seg["speaker"] in speakers or len(speakers) < 2:
            current.append(seg)
            speakers.add(seg["speaker"])
        else:
            if len(speakers) == 2:
                runs.append(current)
            current = [seg]
            speakers = {seg["speaker"]}

    if len(speakers) == 2:
        runs.append(current)

    return runs


def is_balanced_dialogue(dialogue: Dialogue, max_single_speaker_ratio: float) -> bool:
    """Return True if no single speaker exceeds max_single_speaker_ratio of total turn time."""
    ratios = speaker_time_ratios(dialogue)
    return bool(ratios) and max(ratios.values()) <= max_single_speaker_ratio


def speaker_time_ratios(dialogue: Dialogue) -> dict[str, float]:
    """Return each speaker's share of diarized speech time in a dialogue."""
    speaker_duration: dict[str, float] = {}
    for seg in dialogue.segments:
        dur = seg["end"] - seg["start"]
        speaker_duration[seg["speaker"]] = speaker_duration.get(seg["speaker"], 0.0) + dur
    total = sum(speaker_duration.values())
    if total <= 0:
        return {}
    return {speaker: duration / total for speaker, duration in speaker_duration.items()}


def dialogue_filter_summary(segments: list[dict]) -> dict[str, int]:
    """Count the stages and rejection reasons used by conversation filtering."""
    groups = split_into_dialogues(segments, gap_seconds=5.0)
    two_speaker_runs = [run for group in groups for run in _two_speaker_runs(group.segments)]
    short = 0
    imbalanced = 0
    accepted = 0
    for run in two_speaker_runs:
        dialogue = _dialogue_from_segments(run)
        if dialogue.duration < 10.0:
            short += 1
            continue
        chunks = _split_long_dialogue(dialogue, 600.0, 10.0)
        for chunk in chunks:
            if chunk.duration < 10.0:
                short += 1
            elif not is_balanced_dialogue(chunk, 0.8):
                imbalanced += 1
            else:
                accepted += 1
    return {
        "segments": len(segments),
        "speakers": len({str(segment["speaker"]) for segment in segments}),
        "silence_groups": len(groups),
        "two_speaker_runs": len(two_speaker_runs),
        "rejected_short": short,
        "rejected_imbalanced": imbalanced,
        "accepted": accepted,
    }


def _split_long_dialogue(
    dlg: Dialogue, max_duration: float, min_duration: float,
) -> list[Dialogue]:
    """Split only at an internal dual-speaker silence; never crop active speech."""
    if dlg.duration <= max_duration:
        return [dlg]

    chunks: list[Dialogue] = []
    remaining = sorted(dlg.segments, key=lambda segment: (segment["start"], segment["end"]))
    while remaining:
        candidate = _dialogue_from_segments(remaining)
        if candidate.duration <= max_duration:
            chunks.append(candidate)
            break

        boundary = _nearest_dual_silence_boundary(
            remaining, target=candidate.start + max_duration, lower_bound=candidate.start,
        )
        if boundary is None:
            # A duration cap must not cut a word, utterance, interruption, or overlap.
            chunks.append(candidate)
            break

        silence_start, silence_end = boundary
        left = [segment for segment in remaining if segment["end"] <= silence_start]
        right = [segment for segment in remaining if segment["start"] >= silence_end]
        if not left or not right:
            chunks.append(candidate)
            break
        chunks.append(_dialogue_from_segments(left))
        remaining = right

    return chunks


def _nearest_dual_silence_boundary(
    segments: list[dict], target: float, lower_bound: float, minimum_silence: float = 0.3,
) -> tuple[float, float] | None:
    """Return an internal silence interval, first within target +/- 10 seconds."""
    active_intervals: list[list[float]] = []
    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        start, end = segment["start"], segment["end"]
        if not active_intervals or start > active_intervals[-1][1]:
            active_intervals.append([start, end])
        else:
            active_intervals[-1][1] = max(active_intervals[-1][1], end)

    silences = [
        (left[1], right[0])
        for left, right in zip(active_intervals, active_intervals[1:])
        if left[1] >= lower_bound and right[0] - left[1] >= minimum_silence
    ]
    if not silences:
        return None

    nearby = [gap for gap in silences if abs(((gap[0] + gap[1]) / 2) - target) <= 10.0]
    candidates = nearby or silences
    return min(candidates, key=lambda gap: (abs(((gap[0] + gap[1]) / 2) - target), gap[0]))


def extract_valid_dialogues(
    segments: list[dict],
    gap_seconds: float = 5.0,
    max_single_speaker_ratio: float = 0.8,
    min_duration_seconds: float = 10.0,
    max_duration_seconds: float = 600.0,
) -> list[Dialogue]:
    """
    Split by silence gaps, then within each group extract all maximal 2-speaker
    runs and keep those that pass the dominance and duration filters.
    Dialogues longer than max_duration_seconds are split into chunks.
    """
    result: list[Dialogue] = []
    for group in split_into_dialogues(segments, gap_seconds):
        for run in _two_speaker_runs(group.segments):
            dlg = _dialogue_from_segments(run)
            if dlg.duration < min_duration_seconds:
                continue
            orig_duration = dlg.duration
            chunks = _split_long_dialogue(dlg, max_duration_seconds, min_duration_seconds)
            is_split = len(chunks) > 1
            for idx, chunk in enumerate(chunks, 1):
                if chunk.duration < min_duration_seconds:
                    continue
                if is_balanced_dialogue(chunk, max_single_speaker_ratio):
                    ratios = speaker_time_ratios(chunk)
                    max_ratio = max(ratios.values()) if ratios else 0.0
                    if is_split:
                        chunk.reason = (
                            f"split_from_long_dialogue (parent_duration={orig_duration:.1f}s, "
                            f"chunk {idx}/{len(chunks)}, duration={chunk.duration:.1f}s, max_speaker_ratio={max_ratio:.2f})"
                        )
                    else:
                        chunk.reason = (
                            f"accepted_standard_2_speaker_dialogue (duration={chunk.duration:.1f}s, "
                            f"max_speaker_ratio={max_ratio:.2f})"
                        )
                    result.append(chunk)
    return result

