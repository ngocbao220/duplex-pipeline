"""Sommelier-style colored console logging for pipeline orchestration."""
from __future__ import annotations

import logging
import sys
import time

ANSI_RESET = "\033[0m"
# Normal green is visibly darker than bright/bold green in Kaggle terminals.
ANSI_GREEN = "\033[0;32m"
ANSI_YELLOW = "\033[1;33m"
ANSI_RED = "\033[1;31m"


class PlainFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")


class SommelierColorFormatter(PlainFormatter):
    COLORS = {
        logging.DEBUG: ANSI_GREEN,
        logging.INFO: ANSI_GREEN,
        logging.WARNING: ANSI_YELLOW,
        logging.ERROR: ANSI_RED,
        logging.CRITICAL: ANSI_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return f"{self.COLORS.get(record.levelno, '')}{text}{ANSI_RESET}"


def get_logger(name: str) -> logging.Logger:
    """Return one non-propagating console logger with Sommelier's level colors."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not any(getattr(handler, "_sommelier_style", False) for handler in logger.handlers):
        # Sommelier emits normal phase logs on stdout; stderr is reserved for
        # third-party libraries that the worker captures separately.
        handler = logging.StreamHandler(sys.stdout)
        handler._sommelier_style = True  # type: ignore[attr-defined]
        handler.setFormatter(SommelierColorFormatter())
        logger.addHandler(handler)
    return logger


class StepTimer:
    """Emit a native-looking step heading followed by elapsed time and RTF."""
    def __init__(self, logger: logging.Logger, step: str, *, duration_sec: float | None = None, details: str = "") -> None:
        self.logger = logger
        self.step = step
        self.duration_sec = duration_sec
        self.details = details
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        suffix = f" ({self.details})" if self.details else ""
        self.logger.info("%s%s", self.step, suffix)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.started
        self.elapsed = elapsed
        if exc is None:
            rtf = elapsed / self.duration_sec if self.duration_sec else 0.0
            self.logger.info("%s - Processing time: %.2fs, RT factor: %.4f", self.step, elapsed, rtf)
        else:
            self.logger.error("%s failed: %s: %s", self.step, exc_type.__name__, exc)
        return False
