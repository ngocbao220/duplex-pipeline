"""Regression checks for the Kaggle notebook bootstrap."""
from __future__ import annotations

import json
from pathlib import Path
import unittest


class KaggleNotebookBootstrapTests(unittest.TestCase):
    def test_notebook_exposes_sommelier_source_to_the_kernel_without_editable_install(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "duplex-pipelines.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("pipeline/sommelier/src", source)
        self.assertIn("import sommelier", source)
        self.assertNotIn("pip install -e .", source)
        self.assertIn('UserSecretsClient().get_secret("HF_TOKEN")', source)
