"""Purpose: Expose Sommelier through the shared pipeline CLI.

Inputs: The common `single` command arguments.
Outputs: The shared canonical output contract.
"""
from core.orchestration.cli import run_pipeline_command

raise SystemExit(run_pipeline_command("sommelier"))
