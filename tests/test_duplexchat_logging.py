from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat.runner import _compact_path, _log_run_summary  # noqa: E402


class _CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def test_run_summary_lists_devices_conversations_and_phase_details(tmp_path):
    logger = _CaptureLogger()
    _log_run_summary(
        logger,
        tmp_path / "input.wav",
        tmp_path / "phases" / "phase_01_preprocess" / "audio_16k_mono.wav",
        tmp_path,
        tmp_path / "phases",
        tmp_path / "conversations" / "manifest.json",
        ["cuda:0", "cuda:1"],
        3,
        96,
        {"preprocess": 0.25, "diarization": 21.5, "separation": 151.5},
        audio_duration_sec=346.5,
    )

    output = "\n".join(logger.messages)
    assert "Detected conversations: 3" in output
    assert "Device: cuda:0" in output
    assert "1. Preprocessing — Time: 0.25s" in output
    assert "2. Split Dialogue — Time: 21.50s" in output
    assert "2. DuplexChat Separation — Time: 151.50s" in output
    assert "Total: 173.25s" in output

    speech_md = tmp_path / "speech.md"
    assert speech_md.exists()
    speech_text = speech_md.read_text(encoding="utf-8")
    assert "| Stage | Processing Time (s) | RTF |" in speech_text
    assert "| Audio Duration | 346.50 | — |" in speech_text
    assert "1. Preprocessing" in speech_text and "0.25" in speech_text
    assert "2. Split Dialogue" in speech_text and "21.50" in speech_text
    assert "2. DuplexChat Separation" in speech_text and "151.50" in speech_text
    assert "| **Total** | **173.25** |" in speech_text


def test_summary_compacts_long_paths_without_losing_the_filename():
    path = "/kaggle/input/datasets/ngocbaotrinhtuan/inputs/easy_1.wav"

    compacted = _compact_path(path, width=24)

    assert len(compacted) == 24
    assert compacted.startswith("...")
    assert compacted.endswith("easy_1.wav")