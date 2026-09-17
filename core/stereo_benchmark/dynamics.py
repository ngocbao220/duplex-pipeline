"""Reference-free turn, overlap-transition, backchannel and energy diagnostics."""

import math

import numpy as np


MERGE_GAP_SEC = 0.5
MIN_TURN_DURATION_SEC = 1.0
MAX_BACKCHANNEL_DURATION_SEC = 1.0


def cosine_distinctiveness(left: np.ndarray, right: np.ndarray) -> float:
    """ITD: one minus cosine similarity of two speaker representations; higher is more distinct."""
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        raise ValueError("Cannot calculate cosine similarity from a zero embedding")
    return float(1.0 - np.dot(left, right) / denom)


def _spurts(mask: np.ndarray, frame_sec: float, speaker: str) -> list[dict]:
    raw = []
    start = None
    for i, active in enumerate(np.append(mask, False)):
        if active and start is None:
            start = i
        elif not active and start is not None:
            raw.append([start * frame_sec, i * frame_sec])
            start = None
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] < MERGE_GAP_SEC:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [
        {"start": round(start, 4), "end": round(end, 4), "duration": round(end - start, 4), "speaker": speaker}
        for start, end in merged
    ]


def analyze_turns(left: np.ndarray, right: np.ndarray, frame_sec: float) -> dict:
    """Measure VAD-based dialogue dynamics, explicitly treating short responses as candidates only."""
    left_spurts = _spurts(left, frame_sec, "left")
    right_spurts = _spurts(right, frame_sec, "right")
    all_spurts = sorted(left_spurts + right_spurts, key=lambda event: (event["start"], event["speaker"]))
    valid_turns = [event for event in all_spurts if event["duration"] >= MIN_TURN_DURATION_SEC]
    transitions = []
    previous = None
    for event in valid_turns:
        if previous is not None and event["speaker"] != previous["speaker"]:
            transitions.append({
                "from": previous["speaker"], "to": event["speaker"], "start": event["start"],
                "overlapping": event["start"] < previous["end"],
            })
        previous = event
    candidates = []
    for event in all_spurts:
        if event["duration"] > MAX_BACKCHANNEL_DURATION_SEC:
            continue
        others = left_spurts if event["speaker"] == "right" else right_spurts
        if any(other["duration"] >= MIN_TURN_DURATION_SEC and other["start"] <= event["start"] and other["end"] >= event["end"] for other in others):
            candidates.append({**event, "type": "vad_based_backchannel_candidate"})
    duration_sec = len(left) * frame_sec
    return {
        "turns": valid_turns,
        "all_talk_spurts": all_spurts,
        "transitions": transitions,
        "transition_count": len(transitions),
        "overlapping_transition_count": sum(event["overlapping"] for event in transitions),
        "overlapping_transition_rate": (sum(event["overlapping"] for event in transitions) / len(transitions) if transitions else None),
        "mean_turn_duration_sec": _mean([event["duration"] for event in valid_turns]),
        "mean_left_turn_duration_sec": _mean([event["duration"] for event in valid_turns if event["speaker"] == "left"]),
        "mean_right_turn_duration_sec": _mean([event["duration"] for event in valid_turns if event["speaker"] == "right"]),
        "turn_exchanges_per_min": len(transitions) / (duration_sec / 60) if duration_sec else None,
        "backchannel_candidates": candidates,
        "backchannels_per_min": len(candidates) / (duration_sec / 60) if duration_sec else None,
        "mean_backchannel_duration_sec": _mean([event["duration"] for event in candidates]),
    }


def inactive_channel_energy_ratio_db(left: np.ndarray, right: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, frame_sec: float, sample_rate: int) -> dict:
    """Energy proxy in VAD-inactive channels; diagnostic only, not a ground-truth SIR."""
    frame_samples = max(1, round(frame_sec * sample_rate))
    values = {"left_to_right": [], "right_to_left": []}
    for index, (left_active, right_active) in enumerate(zip(left_mask, right_mask, strict=True)):
        start, end = index * frame_samples, min((index + 1) * frame_samples, len(left))
        if end <= start:
            continue
        left_energy = float(np.mean(left[start:end] ** 2))
        right_energy = float(np.mean(right[start:end] ** 2))
        if left_active and not right_active:
            values["left_to_right"].append(10 * math.log10((right_energy + 1e-12) / (left_energy + 1e-12)))
        elif right_active and not left_active:
            values["right_to_left"].append(10 * math.log10((left_energy + 1e-12) / (right_energy + 1e-12)))
    return {name: _distribution(value) for name, value in values.items()}


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"status": "unavailable", "reason": "no_single_speaker_activity"}
    return {"status": "ok", "mean_db": float(np.mean(values)), "median_db": float(np.median(values)), "frame_count": len(values)}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None
