"""Purpose: Expose the Cholimex pipeline public runner.

Inputs: Imports by the isolated CLI and orchestration worker.
Outputs: ``run_cholimex_file`` as the package entrypoint.
"""

__all__ = ["run_cholimex_file"]


def __getattr__(name: str):
    if name == "run_cholimex_file":
        from .runner import run_cholimex_file
        return run_cholimex_file
    raise AttributeError(name)
