"""Regression checks for GPU visibility passed into pipeline workers."""
from __future__ import annotations

from pathlib import Path
import unittest


class RuntimeGpuVisibilityTests(unittest.TestCase):
    def test_automatic_gpu_target_does_not_mask_visible_devices(self):
        source = (Path(__file__).resolve().parents[1] / "run_pipeline.py").read_text(encoding="utf-8")

        self.assertIn('if gpu_id.lower() in ("auto", "all"):', source)
        self.assertIn('env.pop("CUDA_VISIBLE_DEVICES", None)', source)
