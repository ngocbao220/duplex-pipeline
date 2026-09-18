"""Sommelier-style colored console logging for pipeline orchestration.

Log format theo logging.md:
  [INFO]    màu xanh cyan
  [WARNING] màu vàng
  [ERROR]   màu đỏ
  Section headers: =============== Title ===============
"""
from __future__ import annotations

import logging
import sys
import time

ANSI_RESET   = "\033[0m"
ANSI_BOLD    = "\033[1m"
ANSI_CYAN    = "\033[0;36m"   # INFO
ANSI_YELLOW  = "\033[1;33m"   # WARNING
ANSI_RED     = "\033[1;31m"   # ERROR
ANSI_GREEN   = "\033[0;32m"   # SUCCESS / DEBUG


def section(title: str) -> str:
    """Trả về dòng header kiểu: =============== Title ==============="""
    return f"\n{ANSI_BOLD}{'=' * 15} {title} {'=' * 15}{ANSI_RESET}"


class _ColorFormatter(logging.Formatter):
    """Format log theo chuẩn logging.md: [LEVEL] message (có màu)."""

    LEVEL_TAG = {
        logging.DEBUG:    f"{ANSI_GREEN}[INFO]{ANSI_RESET}",
        logging.INFO:     f"{ANSI_CYAN}[INFO]{ANSI_RESET}",
        logging.WARNING:  f"{ANSI_YELLOW}[WARNING]{ANSI_RESET}",
        logging.ERROR:    f"{ANSI_RED}[ERROR]{ANSI_RESET}",
        logging.CRITICAL: f"{ANSI_RED}[ERROR]{ANSI_RESET}",
    }

    def format(self, record: logging.LogRecord) -> str:
        tag = self.LEVEL_TAG.get(record.levelno, "[INFO]")
        msg = record.getMessage()
        return f"{tag}  {msg}"


# Giữ PlainFormatter cho file handler / backward compat
class PlainFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")


# Legacy alias (các file cũ import SommelierColorFormatter)
class SommelierColorFormatter(PlainFormatter):
    COLORS = {
        logging.DEBUG:    ANSI_GREEN,
        logging.INFO:     ANSI_CYAN,
        logging.WARNING:  ANSI_YELLOW,
        logging.ERROR:    ANSI_RED,
        logging.CRITICAL: ANSI_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return f"{self.COLORS.get(record.levelno, '')}{text}{ANSI_RESET}"


def get_logger(name: str) -> logging.Logger:
    """Return một console logger dùng format [INFO]/[WARNING]/[ERROR] theo logging.md."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not any(getattr(h, "_pipeline_style", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler._pipeline_style = True  # type: ignore[attr-defined]
        handler.setFormatter(_ColorFormatter())
        logger.addHandler(handler)
    return logger


class StepTimer:
    """Emit section header khi bắt đầu bước, emit Time + RTF khi kết thúc."""

    def __init__(
        self,
        logger: logging.Logger,
        step: str,
        *,
        duration_sec: float | None = None,
        details: str = "",
    ) -> None:
        self.logger = logger
        self.step = step
        self.duration_sec = duration_sec
        self.details = details
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        suffix = f" ({self.details})" if self.details else ""
        self.logger.info(f"{self.step}{suffix}")
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.started
        self.elapsed = elapsed
        if exc is None:
            rtf = elapsed / self.duration_sec if self.duration_sec else 0.0
            self.logger.info(
                "Time: %.2fs | RTF: %.4f", elapsed, rtf
            )
        else:
            self.logger.error("%s failed: %s: %s", self.step, exc_type.__name__, exc)
        return False
