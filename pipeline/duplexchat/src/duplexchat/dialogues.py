"""Purpose: Summarize detected conversation and valid dialogue spans.

Inputs: Diarization segments.
Outputs: Conversation groups and two-speaker valid dialogues.
"""
from .dialogue import extract_valid_dialogues, split_into_dialogues


def summarize(segments):
    return split_into_dialogues(segments, gap_seconds=5.0), extract_valid_dialogues(segments, gap_seconds=5.0, min_duration_seconds=10.0)
