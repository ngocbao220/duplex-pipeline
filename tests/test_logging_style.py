from __future__ import annotations

import logging

from core.orchestration.logging_style import ANSI_GREEN, ANSI_RESET, PlainFormatter, SommelierColorFormatter


def _record(level: int, message: str = "Step 2: Speaker Diarization") -> logging.LogRecord:
    return logging.LogRecord("vilier", level, __file__, 1, message, (), None)


def test_sommelier_console_format_has_colored_bracketed_info_level():
    text = SommelierColorFormatter().format(_record(logging.INFO))
    assert ANSI_GREEN in text
    assert " - vilier - [INFO] - Step 2: Speaker Diarization" in text
    assert text.endswith(ANSI_RESET)


def test_worker_formatter_has_same_message_without_ansi():
    text = PlainFormatter().format(_record(logging.WARNING, "checkpoint missing"))
    assert " - vilier - [WARNING] - checkpoint missing" in text
    assert "\x1b[" not in text
