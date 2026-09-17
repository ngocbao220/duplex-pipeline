from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def is_console_noise(line: str) -> bool:
    """Keep known third-party chatter in the worker log, not the progress console."""
    text = strip_ansi(line).strip()
    return text.startswith((
        "OneLogger:",
        "No exporters were provided.",
        "Warning: You are sending unauthenticated requests to the HF Hub.",
        "[NeMo W ",
        "Diarizing:",
    ))


def stream_process(command: list[str], cwd: Path, log: Path, env: dict | None = None) -> int:
    """Stream merged stdout/stderr and reap the complete process group on interruption."""
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ if env is None else env)
    # Workers are non-interactive subprocesses.  A notebook's inline backend
    # is not installed in their isolated virtual environments.
    child_env['MPLBACKEND'] = 'Agg'
    child_env['PYTHONUNBUFFERED'] = '1'
    child_env.setdefault('MPLCONFIGDIR', str(log.parent / 'matplotlib'))
    child_env.setdefault('NUMBA_CACHE_DIR', str(log.parent / 'numba'))
    with log.open('a', encoding='utf-8') as handle:
        with subprocess.Popen(command, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace',
                              start_new_session=True, bufsize=1) as child:
            try:
                for line in child.stdout:
                    handle.write(strip_ansi(line))
                    handle.flush()
                    if not is_console_noise(line):
                        print(line, end='', flush=True)
                return child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    child.wait()
                raise
