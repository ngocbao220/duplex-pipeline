#!/usr/bin/env python3
"""Run a reference-free benchmark on one timeline-aligned stereo audio file."""

import argparse
import sys
from pathlib import Path

from core.stereo_benchmark.preflight import benchmark_models_ready, check_benchmark_models, print_model_check_summary
from core.stereo_benchmark.report import render_tables
from core.stereo_benchmark.runner import print_summary, run_benchmark, run_corpus_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_mode = parser.add_mutually_exclusive_group(required=False)
    input_mode.add_argument("--audio", type=Path, help="One two-channel speaker-separated audio file")
    input_mode.add_argument("--corpus", type=Path, help="Directory recursively containing stereo audio files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dnsmos-model-dir", type=Path, default=Path("models/dnsmos"), help="Directory containing Microsoft's sig_bak_ovr.onnx and model_v8.onnx")
    parser.add_argument("--check-models", action="store_true", help="Check availability of all benchmark models and dependencies before running")
    parser.add_argument("--debug", action="store_true", help="Write separated channels and detailed event files")
    args = parser.parse_args()

    if not args.audio and not args.corpus and not args.check_models:
        parser.error("one of the arguments --audio, --corpus, or --check-models is required")

    if args.check_models:
        results = check_benchmark_models(args.dnsmos_model_dir, device=args.device)
        print_model_check_summary(results)
        if not benchmark_models_ready(results):
            print("Benchmark aborted: strict local model preflight failed.", file=sys.stderr)
            sys.exit(1)
        if not args.audio and not args.corpus:
            sys.exit(0)

    if args.audio:
        report, report_path = run_benchmark(args.audio, args.output_dir, args.device, args.debug, args.dnsmos_model_dir)
        print_summary(report, report_path)
    elif args.corpus:
        report, report_path = run_corpus_benchmark(args.corpus, args.output_dir, args.device, args.debug, args.dnsmos_model_dir)
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

        print(f"\nCandidates: {report['candidate_count']} | Successful: {report['summary']['sample_count']}\nJSON report: {report_path}")


if __name__ == "__main__":
    main()
