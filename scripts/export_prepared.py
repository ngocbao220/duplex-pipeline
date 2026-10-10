"""Export ASR results into the trainer prepared_dir (train.jsonl + samples/)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.asr.prepared_export import export_config, export_prepared

GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def _log(level: str, message: str) -> None:
    color = {"INFO": GREEN, "WARNING": YELLOW, "ERROR": RED}[level]
    print(f"{color}[{level}]{RESET} {message}", file=sys.stderr if level == "ERROR" else sys.stdout, flush=True)


def _progress(items):
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="Export", unit="sample")


def run_export(stereo_root: Path, out_dir: Path, cfg: dict) -> int:
    print("=============== Prepared Export ===============", flush=True)
    _log("INFO", f"Stereo root: {stereo_root}")
    _log("INFO", f"Output: {out_dir}")
    _log("INFO", f"Agent channel: {'right (swap_roles)' if cfg.get('swap_roles') else 'left'}")
    started = time.perf_counter()
    report = export_prepared(stereo_root, out_dir, cfg, progress=_progress)
    elapsed = time.perf_counter() - started
    rtf = elapsed / (report["hours"] * 3600) if report["hours"] else 0.0
    _log("INFO", f"Samples: {report['total']}  Kept: {report['kept']}  Rejected: {report['rejected']}")
    for reason, count in sorted(report["reasons"].items()):
        _log("WARNING", f"++++ {count} {reason}")
    _log("INFO", f"Total duration: {report['hours']:.2f} h")
    _log("INFO", f"Time/RTF: {elapsed:.1f}s / {rtf:.4f}")
    _log("INFO", f"Done, result save to {report['manifest']}")
    if report["total"] and not report["kept"]:
        _log("ERROR", "All samples rejected; see export_report.json")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-json", default="{}", help="asr.export config as JSON")
    args = parser.parse_args()
    return run_export(args.stereo_root, args.out_dir, export_config(json.loads(args.config_json)))


if __name__ == "__main__":
    raise SystemExit(main())
