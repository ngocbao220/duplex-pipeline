"""Purpose: Provide the standalone DuplexChat command-line entrypoint.

Inputs: ``single`` command-line arguments.
Outputs: DuplexChat run artifacts, benchmark reports and process exit status.
"""
from core.orchestration.cli import run_pipeline_command


raise SystemExit(run_pipeline_command("duplexchat"))
