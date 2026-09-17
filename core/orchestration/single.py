"""Single-file bridge to the isolated pipeline workers."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

from .runner import ROOT, code_identity, launch_pipeline, pipeline_config
from .contract import write_json


def run_single(name: str, source: Path, output: Path, debug: bool, separate_chunk: float,
               device_ids: list[int] | None,
               filter_music: bool = False, music_model: str = "htdemucs",
               diarization_backend: str | None = None,
               diarization_model: str | None = None,
               separation_model: str | None = None) -> int:
    """Run one adapter without reference audio."""
    from core.config import load_config

    cfg = load_config(ROOT / "configs/config.json")
    args = SimpleNamespace(debug=debug, separate_chunk=separate_chunk, device_ids=device_ids,
                           filter_music=filter_music, music_model=music_model,
                           diarization_backend=diarization_backend, diarization_model=diarization_model,
                           separation_model=separation_model,
                           vilier_config=ROOT / "configs/vilier.json",
                           duplexchat_config=ROOT / "configs/duplexchat.json", sample_rate=16000)
    run_dir = output.parent / ".runs" / uuid.uuid4().hex
    result_path = run_dir / "results.json"
    request = {"pipeline": name, "config": pipeline_config(name, args, cfg),
               "code": code_identity(name), "force": False, "pred_root": str(output.parent),
               "results": str(result_path), "samples": [{"key": output.name, "mixture": str(source)}]}
    exit_code = launch_pipeline(name, request, run_dir)
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    row = rows[0] if rows else {"status": "failed", "error": "worker did not write result"}
    if row.get("status") != "complete":
        print(f"Failed: {row.get('error', f'worker exit={exit_code}')}\n"
              f"-> run.json: {output / 'run.json'}\n"
              f"-> worker log: {run_dir / name / 'worker.log'}", flush=True)
        return exit_code or 1
    print(f"Done\n-> output: {output}\n-> run.json: {output / 'run.json'}", flush=True)
    return exit_code
