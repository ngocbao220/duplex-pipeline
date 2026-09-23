#!/usr/bin/env python3
"""Keep a live aggregate of completed stereo benchmark reports from an existing run."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import sleep

from core.stereo_benchmark.report import RunningSummary, write_json_atomic


def update_summary(output_dir: Path, summary: RunningSummary, seen: set[Path]) -> int:
    added = 0
    for path in sorted((output_dir / "files").glob("*/report.json")):
        if path in seen:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # A worker may still be writing this report; retry next scan.
        if not isinstance(report, dict):
            continue
        summary.add(report)
        seen.add(path)
        added += 1

    write_json_atomic(output_dir / "live_corpus_report.json", {
        "source": "completed per-file report.json files",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "summary": summary.snapshot(),
    })
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Benchmark output folder containing files/*/report.json")
    parser.add_argument("--interval-seconds", type=float, default=60, help="Time between scans (default: 60)")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit")
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error(f"Benchmark output folder does not exist: {args.output_dir}")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    summary = RunningSummary()
    seen: set[Path] = set()
    try:
        while True:
            added = update_summary(args.output_dir, summary, seen)
            if added:
                print(f"Aggregated {summary.sample_count} completed reports (+{added})", flush=True)
            if args.once:
                return
            sleep(args.interval_seconds)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
