"""Create a DialogueSidon checksum manifest beside a trusted local model bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

from core.local_model_validation import write_dialoguesidon_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create checksums for a trusted local DialogueSidon source bundle.")
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_dialoguesidon_manifest(args.model_dir.resolve())
    print(f"Wrote trusted-source manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
