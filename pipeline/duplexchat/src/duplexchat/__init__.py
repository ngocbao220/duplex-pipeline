"""Purpose: Expose the DuplexChat pipeline public runner.

Inputs: Imports by the isolated CLI and orchestration worker.
Outputs: ``run_single_audio`` as the package entrypoint.
"""

__all__ = ["run_single_audio"]


def __getattr__(name: str):
    if name == "run_single_audio":
        from .runner import run_single_audio
        return run_single_audio
    raise AttributeError(name)
