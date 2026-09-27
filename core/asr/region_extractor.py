"""Derive ASR windows without modifying diarization turns."""
from __future__ import annotations


def merge_spans(spans: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        joins = bool(merged) and (start - merged[-1][1] < gap if gap else start <= merged[-1][1])
        if joins:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def asr_regions(turns: list[dict], speaker: str, duration: float, gap: float = 0.5, pad: float = 0.25) -> list[tuple[float, float]]:
    if duration <= 0 or gap < 0 or pad < 0:
        raise ValueError("Invalid region duration, gap or padding")
    spans = [(max(0.0, float(t["start"])), min(duration, float(t["end"]))) for t in turns if t["speaker"] == speaker]
    merged = merge_spans(spans, gap)
    return merge_spans([(max(0.0, start - pad), min(duration, end + pad)) for start, end in merged])


def uncovered_speech(speech: list[tuple[float, float]], turns: list[tuple[float, float]]) -> tuple[float | None, list[tuple[float, float]]]:
    """Fraction of VAD-positive time covered by original speaker turns."""
    speech = merge_spans(speech)
    turns = merge_spans(turns)
    total = sum(end - start for start, end in speech)
    if total == 0:
        return None, []
    remainder = []
    for start, end in speech:
        cursor = start
        for turn_start, turn_end in turns:
            if turn_end <= cursor or turn_start >= end:
                continue
            if turn_start > cursor:
                remainder.append((cursor, min(turn_start, end)))
            cursor = max(cursor, min(end, turn_end))
            if cursor >= end:
                break
        if cursor < end:
            remainder.append((cursor, end))
    return max(0.0, 1.0 - sum(end - start for start, end in remainder) / total), remainder
