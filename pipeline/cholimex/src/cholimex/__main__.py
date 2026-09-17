"""Purpose: Provide the standalone Cholimex command-line entrypoint.

Inputs: ``single`` command-line arguments.
Outputs: Cholimex run artifacts, benchmark reports and process exit status.
"""
from core.orchestration.cli import run_pipeline_command


raise SystemExit(run_pipeline_command("cholimex"))
