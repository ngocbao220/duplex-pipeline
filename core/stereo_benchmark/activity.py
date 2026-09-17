"""Energy-VAD activity masks and full-duplex state accounting."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ActivityConfig:
    frame_sec: float = 0.02
    min_speech_sec: float = 0.08
    min_silence_sec: float = 0.06


def energy_vad(audio: np.ndarray, sample_rate: int, config: ActivityConfig) -> tuple[np.ndarray, float]:
    """Return a deterministic energy-based speech mask; no clean reference is used.

    This is a fallback VAD when an optional neural VAD is not installed. It is
    deliberately reported as ``energy_vad`` rather than a speaker-aware metric.
    """
    frame_samples = max(1, round(sample_rate * config.frame_sec))
    frame_count = int(np.ceil(len(audio) / frame_samples))
    padded = np.pad(audio, (0, frame_count * frame_samples - len(audio)))
    frames = padded.reshape(frame_count, frame_samples)
    db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-10)
    # Relative threshold adapts to quiet recordings, bounded against noise-floor activation.
    threshold_db = max(-55.0, float(np.percentile(db, 15)) + 9.0)
    mask = db >= threshold_db
    return _remove_short_runs(mask, config), threshold_db


def _remove_short_runs(mask: np.ndarray, config: ActivityConfig) -> np.ndarray:
    result = mask.copy()
    for value, minimum_sec in ((True, config.min_speech_sec), (False, config.min_silence_sec)):
        minimum = max(1, round(minimum_sec / config.frame_sec))
        start = 0
        while start < len(result):
            if result[start] != value:
                start += 1
                continue
            end = start + 1
            while end < len(result) and result[end] == value:
                end += 1
            if end - start < minimum:
                result[start:end] = not value
            start = end
    return result


def mask_segments(mask: np.ndarray, frame_sec: float, speaker: str | None = None) -> list[dict]:
    """Convert true runs in a frame mask to half-open time intervals."""
    events: list[dict] = []
    start = None
    for index, active in enumerate(np.append(mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            event = {"start": round(start * frame_sec, 4), "end": round(index * frame_sec, 4)}
            event["duration"] = round(event["end"] - event["start"], 4)
            if speaker is not None:
                event["speaker"] = speaker
            events.append(event)
            start = None
    return events


def activity_summary(left: np.ndarray, right: np.ndarray, frame_sec: float) -> dict:
    """Partition time into left-only, right-only, overlap and silence states."""
    states = {
        "left_only": left & ~right,
        "right_only": right & ~left,
        "overlap": left & right,
        "silence": ~left & ~right,
    }
    total = len(left)
    return {
        name: {
            "duration_sec": float(mask.sum() * frame_sec),
            "percentage": float(mask.sum() / total * 100) if total else 0.0,
        }
        for name, mask in states.items()
    }


def config_dict(config: ActivityConfig) -> dict:
    return asdict(config)
