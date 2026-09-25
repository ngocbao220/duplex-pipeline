#!/usr/bin/env python3
"""Run a reference-free benchmark on one timeline-aligned stereo audio file."""

import argparse
import os
import sys
from pathlib import Path

# On the Linux server, constrain the entire benchmark process before importing
# Torch/NumPy/ONNX so native helper threads inherit the same single-core mask.
if __name__ == "__main__" and sys.platform == "linux":
    allowed_cpus = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(allowed_cpus)})

from core.runtime_cpu import enforce_single_cpu_thread  # Set BLAS/OpenMP limits before model imports.
from core.stereo_benchmark.preflight import benchmark_models_ready, check_benchmark_models, print_model_check_summary
from core.stereo_benchmark.report import render_tables
from core.stereo_benchmark.runner import print_summary, run_benchmark, run_corpus_benchmark
from core.stereo_benchmark.runtime import apply_environment_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_mode = parser.add_mutually_exclusive_group(required=False)
    input_mode.add_argument("--audio", type=Path, help="One two-channel speaker-separated audio file")
    input_mode.add_argument("--corpus", type=Path, help="Directory recursively containing stereo audio files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--env-config", type=Path, default=ROOT / "configs" / "env" / "sever.yaml", help="Hydra environment YAML providing strict local benchmark model paths")
    parser.add_argument("--dnsmos-model-dir", type=Path, default=None, help="Directory containing Microsoft's sig_bak_ovr.onnx and model_v8.onnx")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Concurrent benchmark files (default: 2; CPU remains pinned to one core on Linux)")
    parser.add_argument("--check-models", action="store_true", help="Check availability of all benchmark models and dependencies before running")
    parser.add_argument("--debug", action="store_true", help="Write separated channels and detailed event files")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Stereo benchmark requires --workers >= 1")
    configured_paths = apply_environment_config(args.env_config)
    dnsmos_model_dir = args.dnsmos_model_dir or Path(configured_paths["DNSMOS_MODEL_PATH"])

    if not args.audio and not args.corpus and not args.check_models:
        parser.error("one of the arguments --audio, --corpus, or --check-models is required")

    if args.check_models:
        results = check_benchmark_models(dnsmos_model_dir, device=args.device)
        print_model_check_summary(results)
        if not benchmark_models_ready(results):
            print("Benchmark aborted: strict local model preflight failed.", file=sys.stderr)
            sys.exit(1)
        if not args.audio and not args.corpus:
            sys.exit(0)

    if args.audio:
        report, report_path = run_benchmark(args.audio, args.output_dir, args.device, args.debug, dnsmos_model_dir)
        print_summary(report, report_path)
    elif args.corpus:
        report, report_path = run_corpus_benchmark(args.corpus, args.output_dir, args.device, args.debug, dnsmos_model_dir, workers=args.workers)
        print("Pipeline: Stereo Full-Duplex Corpus Benchmark\n")
        print(render_tables(report["summary"]))

        # Collect and print warnings from the first successfully processed sample
        from core.stereo_benchmark.report import render_warnings
        try:
            import json
            first_report_path = next(Path(args.output_dir).resolve().joinpath(s["report"]) for s in report["samples"] if s["status"] == "ok")
            if warnings := render_warnings([json.loads(first_report_path.read_text())]):
                print(warnings)
        except StopIteration:
            pass

        print(
            f"\nCandidates: {report['candidate_count']} | Selected: {report['selected_count']}"
            f"/{report['sample_limit']} | Successful: {report['summary']['sample_count']}"
            f"\nJSON report: {report_path}"
        )


if __name__ == "__main__":
    main()
