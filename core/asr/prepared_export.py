"""Export ASR results into the trainer ``prepared_dir`` contract.

Layout (see docs/prepared_data.md)::

    <out_dir>/train.jsonl
    <out_dir>/samples/<sample_id>/{conversation.wav, voice_prompt_left.wav,
                                   voice_prompt_right.wav, words.json, metadata.json}

LEFT channel is the agent by default; ``swap_roles=true`` swaps the stereo
channels so the original right speaker becomes the agent.
"""
from __future__ import annotations

import json
import os
import re
import uuid
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24_000
DURATION_TOLERANCE = 0.05
AGENT_ALIASES = {"agent", "left", "a"}
USER_ALIASES = {"user", "right", "b"}

DEFAULT_EXPORT_CFG = {
    "enabled": True,
    "out_dir": None,
    "swap_roles": False,
    "language": "vi",
    "text_prompt_left": "Bạn là trợ lý thân thiện.",
    "text_prompt_right": "Bạn là khách hàng hỏi thông tin.",
    "voice_prompt": {"min_seconds": 5.0, "max_seconds": 10.0, "min_gap_to_other": 0.3},
}


def export_config(cfg: dict | None) -> dict:
    merged = {**DEFAULT_EXPORT_CFG, **(cfg or {})}
    merged["voice_prompt"] = {**DEFAULT_EXPORT_CFG["voice_prompt"], **((cfg or {}).get("voice_prompt") or {})}
    return merged


def _atomic_write_bytes(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    _atomic_write_bytes(path, lambda tmp: sf.write(tmp, clipped, sample_rate, subtype="PCM_16", format="WAV"))


def write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def normalize_speaker(value: str) -> str | None:
    key = str(value).strip().lower()
    if key in AGENT_ALIASES:
        return "agent"
    if key in USER_ALIASES:
        return "user"
    return None


def validate_words(words: list, duration: float) -> list[str]:
    """Return contract violations of a ``words.json`` array (empty list = valid)."""
    errors: list[str] = []
    if not isinstance(words, list):
        return ["words.json must be a JSON array"]
    previous = -1.0
    has_agent = False
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            errors.append(f"word[{index}] is not an object")
            continue
        if word.get("speaker") not in ("agent", "user"):
            errors.append(f"word[{index}] invalid speaker {word.get('speaker')!r}")
        has_agent = has_agent or word.get("speaker") == "agent"
        text = word.get("word")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"word[{index}] empty word")
        start, end = word.get("start"), word.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) \
                or isinstance(start, bool) or isinstance(end, bool):
            errors.append(f"word[{index}] start/end must be numbers")
            continue
        if start < 0:
            errors.append(f"word[{index}] start < 0")
        if end <= start:
            errors.append(f"word[{index}] end <= start")
        if end > duration + DURATION_TOLERANCE:
            errors.append(f"word[{index}] out_of_bounds end={end} > duration={duration:.3f}")
        if start < previous:
            errors.append(f"word[{index}] not sorted by start")
        previous = max(previous, float(start))
    if not has_agent:
        errors.append("no agent words")
    return errors


def words_for_trainer(asr_result: dict, agent_channel: int, duration: float) -> list[dict]:
    """Convert aligned per-channel ASR words into the trainer ``words.json`` schema."""
    rows = []
    for channel in asr_result.get("channels", []):
        role = "agent" if int(channel["channel"]) == agent_channel else "user"
        for word in channel.get("words", []):
            text = str(word.get("word") or "").strip()
            start, end = word.get("start"), word.get("end")
            if not text or start is None or end is None or word.get("aligned") is False:
                continue
            start = max(0.0, float(start))
            end = min(float(end), duration)
            if end <= start:
                continue
            rows.append({"speaker": role, "word": text, "start": round(start, 3), "end": round(end, 3)})
    rows.sort(key=lambda row: (row["start"], row["end"]))
    return rows


def _turn_spans(turns: list[dict], speaker: str) -> list[tuple[float, float]]:
    return [(float(t["start"]), float(t["end"])) for t in turns if str(t["speaker"]) == speaker]


def _exclusive_spans(own: list[tuple[float, float]], other: list[tuple[float, float]],
                     gap: float, duration: float) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    for start, end in own:
        current = [(max(0.0, start), min(duration, end))]
        for other_start, other_end in other:
            blocked_start, blocked_end = other_start - gap, other_end + gap
            nxt = []
            for a, b in current:
                if blocked_start > a:
                    nxt.append((a, min(b, blocked_start)))
                if blocked_end < b:
                    nxt.append((max(a, blocked_end), b))
            current = [(a, b) for a, b in nxt if b > a]
        pieces.extend(current)
    return pieces


def select_voice_prompt(audio: np.ndarray, own: list[tuple[float, float]], other: list[tuple[float, float]],
                        cfg: dict, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
    """Pick a single-speaker clip (``min``-``max`` s) that never overlaps the other speaker."""
    duration = len(audio) / sample_rate
    min_s, max_s = float(cfg["min_seconds"]), float(cfg["max_seconds"])
    spans = _exclusive_spans(own, other, float(cfg["min_gap_to_other"]), duration)
    best, best_score = None, -1.0
    for start, end in spans:
        if end - start < min_s:
            continue
        first = round(start * sample_rate)
        last = round(min(end, start + max_s) * sample_rate)
        clip = audio[first:last]
        score = float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0
        if score > best_score:
            best, best_score = clip, score
    if best is None or best_score <= 1e-4:
        return None
    return np.asarray(best, dtype=np.float32)


def sample_id_for(stereo: Path, stereo_root: Path) -> str:
    relative = stereo.relative_to(stereo_root).with_suffix("")
    slug = "__".join(relative.parts)
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", slug)


def export_sample(out_dir: Path, stereo: Path, stereo_root: Path, asr_result: dict, cfg: dict) -> dict:
    """Write one sample directory; raise ``ValueError`` with a reason if it is rejected."""
    audio, rate = sf.read(stereo, dtype="float32", always_2d=True)
    if audio.shape[1] != 2 or rate != SAMPLE_RATE or not len(audio):
        raise ValueError(f"invalid_audio: expected nonempty {SAMPLE_RATE} Hz stereo")
    swap = bool(cfg["swap_roles"])
    agent_src = 1 if swap else 0
    duration = len(audio) / SAMPLE_RATE
    words = words_for_trainer(asr_result, agent_src, duration)
    errors = validate_words(words, duration)
    if errors:
        raise ValueError("invalid_words: " + "; ".join(errors[:3]))

    channel_speakers = asr_result["provenance"]["channel_attribution"]["channel_speakers"]
    turns = asr_result.get("speaker_turns") or []
    left_src, right_src = (1, 0) if swap else (0, 1)
    spans = {ch: _turn_spans(turns, str(channel_speakers[ch])) for ch in (0, 1)}
    prompt_left = select_voice_prompt(audio[:, left_src], spans[left_src], spans[right_src], cfg["voice_prompt"])
    if prompt_left is None:
        raise ValueError("no_voice_prompt: no clean single-speaker span for agent")
    prompt_right = select_voice_prompt(audio[:, right_src], spans[right_src], spans[left_src], cfg["voice_prompt"])

    sample_id = sample_id_for(stereo, stereo_root)
    sample_dir = out_dir / "samples" / sample_id
    write_pcm16(sample_dir / "conversation.wav", audio[:, [left_src, right_src]])
    write_pcm16(sample_dir / "voice_prompt_left.wav", prompt_left)
    stale_right = sample_dir / "voice_prompt_right.wav"
    if prompt_right is not None:
        write_pcm16(stale_right, prompt_right)
    elif stale_right.exists():
        stale_right.unlink()
    real_duration = wav_duration(sample_dir / "conversation.wav")
    write_text(sample_dir / "words.json", json.dumps(words, ensure_ascii=False, indent=1))
    metadata = {
        "sample_id": sample_id,
        "duration_sec": round(real_duration, 2),
        "language": cfg["language"],
        "agent_channel": "left",
        "user_channel": "right",
        "text_prompt_left": cfg["text_prompt_left"],
        "text_prompt_right": cfg.get("text_prompt_right"),
        "voice_prompt_left": "same_conversation",
        "voice_prompt_right": "same_conversation" if prompt_right is not None else None,
        # Extra provenance (ignored by the trainer); conversation_id keeps variants on one split side.
        "conversation_id": sample_id,
        "swap_roles": swap,
        "source_speakers": {"left": channel_speakers[left_src], "right": channel_speakers[right_src]},
    }
    if not metadata["text_prompt_right"]:
        metadata.pop("text_prompt_right")
    write_text(sample_dir / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    return {"sample_id": sample_id, "sample_dir": f"samples/{sample_id}", "duration": round(real_duration, 3)}


def validate_sample_dir(root: Path, entry: dict) -> list[str]:
    """Re-check a written sample against the trainer contract."""
    errors = []
    sample_dir = entry.get("sample_dir", "")
    if not entry.get("sample_id") or Path(sample_dir).is_absolute() or ".." in Path(sample_dir).parts:
        return ["invalid manifest entry"]
    directory = root / sample_dir
    try:
        with wave.open(str(directory / "conversation.wav"), "rb") as handle:
            if handle.getnchannels() != 2 or handle.getsampwidth() != 2:
                errors.append("conversation.wav must be 2-channel PCM16")
            duration = handle.getnframes() / handle.getframerate()
    except (OSError, wave.Error, EOFError) as exc:
        return [f"conversation.wav unreadable: {exc}"]
    if abs(duration - float(entry.get("duration", duration))) > DURATION_TOLERANCE:
        errors.append("manifest duration mismatch")
    prompt = directory / "voice_prompt_left.wav"
    if not prompt.is_file() or wav_duration(prompt) <= 0:
        errors.append("missing voice_prompt_left.wav")
    try:
        errors.extend(validate_words(json.loads((directory / "words.json").read_text(encoding="utf-8")), duration))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return errors + [f"json unreadable: {exc}"]
    if not str(metadata.get("text_prompt_left") or "").strip():
        errors.append("missing text_prompt_left")
    for key, expected in (("agent_channel", "left"), ("user_channel", "right")):
        if key in metadata and metadata[key] != expected:
            errors.append(f"{key} must be {expected}")
    return errors


def write_manifest(out_dir: Path, entries: list[dict], name: str = "train.jsonl") -> Path:
    path = out_dir / name
    write_text(path, "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries))
    return path


def export_prepared(stereo_root: Path, out_dir: Path, cfg: dict | None = None, progress=None) -> dict:
    """Export every ``stereo_N.wav`` that has a ``dialogue_N.asr.json`` sibling."""
    cfg = export_config(cfg)
    stereo_files = sorted(path for path in stereo_root.rglob("stereo_*.wav")
                          if not {".runs", ".history", "debug"}.intersection(path.parts))
    report = {"stereo_root": str(stereo_root), "out_dir": str(out_dir), "total": len(stereo_files),
              "kept": 0, "rejected": 0, "reasons": {}, "hours": 0.0, "items": []}
    entries = []
    iterator = progress(stereo_files) if progress else stereo_files
    for stereo in iterator:
        asr_path = stereo.with_name(stereo.stem.replace("stereo_", "dialogue_") + ".asr.json")
        try:
            if not asr_path.is_file():
                raise ValueError("missing_asr: run step=asr first")
            entry = export_sample(out_dir, stereo, stereo_root, json.loads(asr_path.read_text(encoding="utf-8")), cfg)
            problems = validate_sample_dir(out_dir, entry)
            if problems:
                raise ValueError("contract: " + "; ".join(problems[:3]))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reason = str(exc).split(":", 1)[0] if isinstance(exc, ValueError) else type(exc).__name__
            report["rejected"] += 1
            report["reasons"][reason] = report["reasons"].get(reason, 0) + 1
            report["items"].append({"stereo": str(stereo), "status": "rejected", "reason": str(exc)})
            continue
        entries.append(entry)
        report["kept"] += 1
        report["hours"] += entry["duration"] / 3600
        report["items"].append({"stereo": str(stereo), "status": "kept", "sample_id": entry["sample_id"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    report["manifest"] = str(write_manifest(out_dir, entries))
    report["hours"] = round(report["hours"], 4)
    write_text(out_dir / "export_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    return report
