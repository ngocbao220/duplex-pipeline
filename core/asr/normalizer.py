"""Keep the ASR text faithful while preparing text for a CTC vocabulary."""
from __future__ import annotations

import unicodedata


def normalize_text(text: str, vocabulary: set[str] | None = None, lowercase: bool = True) -> str:
    value = unicodedata.normalize("NFC", text)
    value = " ".join(value.split())
    if lowercase:
        value = value.lower()
    if vocabulary is not None:
        value = "".join(char for char in value if char in vocabulary or char.isspace())
        value = " ".join(value.split())
    return value
