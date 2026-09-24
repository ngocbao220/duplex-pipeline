#!/usr/bin/env python3
"""Build a t-SNE map from one UTF-8 TXT transcript per dialogue."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.transcript_topic_map import create_topic_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing one .txt per dialogue")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local vietnamese-bi-encoder snapshot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or a CUDA device such as cuda:0")
    args = parser.parse_args()

    try:
        png_path, csv_path = create_topic_map(
            args.input_dir, args.model_dir, args.output_dir, device=args.device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Topic map failed: {error}", file=sys.stderr)
        return 1

    print(f"t-SNE map: {png_path}")
    print(f"Coordinates: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
