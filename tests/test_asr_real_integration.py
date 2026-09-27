"""Opt-in server smoke with one real stereo sample and local checkpoints."""
import json
import os
from pathlib import Path

import pytest
import torch

from core.asr.pipeline import ASRPipeline
from core.orchestration.contract import write_json


@pytest.mark.skipif(
    not all(os.environ.get(name) for name in (
        "ASR_INTEGRATION_STEREO", "ASR_INTEGRATION_DIALOGUE_DIR", "ASR_INTEGRATION_CONFIG",
        "SILERO_VAD_MODEL_PATH", "SPEECHBRAIN_MODEL_PATH",
    )),
    reason="Set one stereo sample, its Phase 1 dialogue directory, local ASR config, Silero and SpeechBrain paths",
)
def test_real_stereo_sample_produces_aligned_json(tmp_path):
    stereo = Path(os.environ["ASR_INTEGRATION_STEREO"])
    dialogue_dir = Path(os.environ["ASR_INTEGRATION_DIALOGUE_DIR"])
    cfg = json.loads(Path(os.environ["ASR_INTEGRATION_CONFIG"]).read_text(encoding="utf-8"))
    pipeline_name = os.environ.get("ASR_INTEGRATION_PIPELINE", "duplexchat")
    engine = ASRPipeline(cfg, offline=True, device="cuda:0" if torch.cuda.is_available() else "cpu",
                         pipeline_name=pipeline_name)
    result = engine.process(stereo, dialogue_dir, pipeline_name)
    output = tmp_path / stereo.name.replace("stereo_", "dialogue_").replace(".wav", ".asr.json")
    write_json(output, result)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert len(loaded["channels"]) == 2
    assert len({row["speaker"] for row in loaded["channels"]}) == 2
    assert loaded["provenance"]["asr"]["backend"] == "sherpa-onnx"
    assert loaded["provenance"]["aligner"]["backend"] == "whisperx"
    assert loaded["merged_timeline"]
    assert all(0 <= row["start"] < row["end"] <= loaded["duration"] for row in loaded["merged_timeline"])
    assert loaded["merged_timeline"] == sorted(
        loaded["merged_timeline"], key=lambda row: (row["start"], row["end"], row["channel"])
    )
