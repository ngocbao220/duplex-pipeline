"""Regression checks for GPU visibility passed into pipeline workers."""
from __future__ import annotations

from pathlib import Path
import unittest

from omegaconf import OmegaConf


class RuntimeGpuVisibilityTests(unittest.TestCase):
    def test_automatic_gpu_target_does_not_mask_visible_devices(self):
        source = (Path(__file__).resolve().parents[1] / "run_pipeline.py").read_text(encoding="utf-8")

        self.assertIn('if gpu_id.lower() in ("auto", "all"):', source)
        self.assertIn('env.pop("CUDA_VISIBLE_DEVICES", None)', source)

    def test_server_mode_forwards_every_sommelier_model_path(self):
        from run_pipeline import _setup_runtime_environment

        cfg = OmegaConf.create({
            "gpu": "auto",
            "env": {
                "offline": True,
                "paths": {
                    "sortformer": "/models/sortformer",
                    "silero_vad": "/models/silero-vad",
                    "speechbrain": "/models/spker",
                    "sepreformer": "/models/epoch.0180.pth",
                },
            },
        })

        env = _setup_runtime_environment(cfg)

        self.assertEqual(env["MODE"], "sever")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["SILERO_VAD_MODEL_PATH"], "/models/silero-vad")
        self.assertEqual(env["SORTFORMER_MODEL_PATH"], "/models/sortformer")
        self.assertEqual(env["SPEECHBRAIN_MODEL_PATH"], "/models/spker")
        self.assertEqual(env["SEPREFORMER"], "/models/epoch.0180.pth")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_batch_launcher_does_not_mask_the_requested_physical_gpu(self):
        from run_pipeline import _batch_runtime_environment

        env = _batch_runtime_environment({"CUDA_VISIBLE_DEVICES": "5"})

        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
