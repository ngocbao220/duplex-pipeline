"""WhisperX CTC word alignment with an explicit local checkpoint in server mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .normalizer import normalize_text


class WhisperXAligner:
    sample_rate = 16_000

    def __init__(self, cfg: dict, offline: bool, device: str):
        import whisperx

        model_name = str(cfg["local_dir"] if offline else cfg["model_id"])
        if offline and not Path(model_name).is_dir():
            raise FileNotFoundError(f"WhisperX local aligner checkpoint is missing: {model_name}")
        self.model, self.metadata = whisperx.load_align_model(
            language_code="vi", device=device, model_name=model_name,
        )
        self.device = device
        self.model_id = str(cfg["model_id"])
        self.vocabulary = set(self.metadata.get("dictionary", {}))
        if not self.vocabulary:
            raise ValueError("WhisperX aligner did not expose a character vocabulary")
        self.lowercase = sum(char.islower() for char in self.vocabulary) >= sum(char.isupper() for char in self.vocabulary)

    def align(self, raw_text: str, audio: np.ndarray) -> tuple[str, list[dict], str, str | None]:
        import whisperx

        text = normalize_text(raw_text, self.vocabulary, lowercase=self.lowercase)
        if not text:
            return text, [], "empty_text", None
        duration = len(audio) / self.sample_rate
        try:
            result = whisperx.align(
                [{"start": 0.0, "end": duration, "text": text}], self.model, self.metadata,
                np.asarray(audio, dtype=np.float32), self.device, return_char_alignments=False,
            )
        except Exception as exc:
            return text, [{"word": word, "start": None, "end": None, "score": None, "aligned": False} for word in text.split()], "failed", f"{type(exc).__name__}: {exc}"

        found = list(result.get("word_segments", []))
        words = []
        cursor = 0
        for word in text.split():
            matched = None
            for index in range(cursor, len(found)):
                if normalize_text(str(found[index].get("word", "")), lowercase=self.lowercase).strip() == word:
                    matched = found[index]
                    cursor = index + 1
                    break
            start = matched.get("start") if matched else None
            end = matched.get("end") if matched else None
            valid = start is not None and end is not None and 0 <= float(start) < float(end) <= duration + 0.01
            words.append({"word": word, "start": float(start) if valid else None,
                          "end": float(end) if valid else None,
                          "score": float(matched["score"]) if valid and matched.get("score") is not None else None,
                          "aligned": bool(valid)})
        status = "aligned" if all(word["aligned"] for word in words) else "partial"
        return text, words, status, None
