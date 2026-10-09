#!/usr/bin/env python3
"""Purpose: Benchmark the speed of every pipeline step on one random sample of raw audio.

Inputs: A raw audio directory, sample size, seed and Hydra overrides for run_pipeline.py.
Outputs: ``<bench_dir>/sample.json``, ``timing.json`` (per phase), ``steps.jsonl`` (per sub-step)
and, through summarize_speed.py, ``report.md`` / ``report.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PHASES = {
    "split": ["step=split_dialogue"],
    "duplexchat": ["step=separate_stereo", "pipeline=duplexchat"],
    "sommelier": ["step=separate_stereo", "pipeline=sommelier"],
    "cholimex": ["step=cholimex", "pipeline=cholimex"],
}


def sample_raw_files(raw_dir: Path, pattern: str, n: int, seed: int) -> list[Path]:
    """Pick ``n`` files at random (reproducible for one seed) from the top level of ``raw_dir``."""
    files = sorted(path for path in raw_dir.glob(pattern) if path.is_file())
    if not files:
        raise SystemExit(f"No files matching '{pattern}' in {raw_dir}")
    return sorted(random.Random(seed).sample(files, min(n, len(files))))


def audio_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        return None


def build_sample(bench_dir: Path, raw_dir: Path, pattern: str, n: int, seed: int) -> dict:
    sample_path = bench_dir / "sample.json"
    if sample_path.is_file():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        print(f"Reusing sample: {sample['count']} files, {sample['audio_hours']:.2f} h")
        return sample
    linked = bench_dir / "raw"
    linked.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sample_raw_files(raw_dir, pattern, n, seed):
        target = linked / source.name
        if not target.exists():
            target.symlink_to(source.resolve())
        rows.append({"file": source.name, "source": str(source.resolve()), "audio_seconds": audio_seconds(source)})
    total = sum(row["audio_seconds"] or 0.0 for row in rows)
    sample = {
        "raw_dir": str(raw_dir.resolve()), "pattern": pattern, "seed": seed, "requested": n,
        "count": len(rows), "audio_seconds": total, "audio_hours": total / 3600, "files": rows,
    }
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Sampled {len(rows)} files, {total / 3600:.2f} h of raw audio -> {linked}")
    return sample


def phase_command(phase: str, bench_dir: Path, gpu: str, workers: int, overrides: list[str], dry_run: bool) -> list[str]:
    cmd = [
        sys.executable, str(ROOT_DIR / "run_pipeline.py"), *PHASES[phase],
        f"data.raw_dir={bench_dir / 'raw'}",
        f"data.dialogue_dir={bench_dir / 'dialogue'}",
        f"data.duplex_out_dir={bench_dir / 'duplexchat'}",
        f"data.sommelier_out_dir={bench_dir / 'sommelier'}",
        f"data.cholimex_out_dir={bench_dir / 'cholimex'}",
        f"gpu={gpu}", f"optimization.workers={workers}", *overrides,
    ]
    if dry_run:
        cmd.append("dry_run=true")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw audio directory to sample from")
    parser.add_argument("--bench-dir", type=Path, default=None, help="Default: outputs/speed_bench/<run_id>")
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", default="split,duplexchat,sommelier,cholimex",
                        help=f"Comma-separated phases, in order. Choices: {','.join(PHASES)}")
    parser.add_argument("--run-id", default=None, help="Reuse to resume a benchmark")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides, e.g. env=sever data.source=youtube")
    args = parser.parse_args()

    phases = [phase.strip() for phase in args.steps.split(",") if phase.strip()]
    unknown = [phase for phase in phases if phase not in PHASES]
    if unknown:
        parser.error(f"Unknown steps: {unknown}")
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("bench-%Y%m%dT%H%M%SZ")
    bench_dir = (args.bench_dir or ROOT_DIR / "outputs" / "speed_bench" / run_id).resolve()
    bench_dir.mkdir(parents=True, exist_ok=True)
    build_sample(bench_dir, args.raw_dir, args.pattern, args.n, args.seed)

    env = dict(os.environ)
    env.update({
        "PIPELINE_RUN_ID": run_id,
        "PIPELINE_TIMING_REPORT": str(bench_dir / "timing.json"),
        "PIPELINE_STEP_TIMINGS": str(bench_dir / "steps.jsonl"),
    })
    for phase in phases:
        cmd = phase_command(phase, bench_dir, args.gpu, args.workers, args.overrides, args.dry_run)
        print(f"\n=== [bench] {phase}: {' '.join(cmd)}", flush=True)
        exit_code = subprocess.call(cmd, cwd=ROOT_DIR, env={**env, "PIPELINE_BENCH_PHASE": phase})
        if exit_code != 0:
            print(f"[bench] {phase} failed with exit code {exit_code}; stopping.")
            return exit_code

    if not args.dry_run:
        from summarize_speed import summarize

        report = summarize(bench_dir)
        print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
