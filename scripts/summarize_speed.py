#!/usr/bin/env python3
"""Purpose: Turn a speed-benchmark directory into a step/pipeline comparison report.

Inputs: ``<bench_dir>/sample.json``, ``timing.json`` and optional ``steps.jsonl`` from benchmark_speed.py.
Outputs: ``<bench_dir>/report.json`` and ``<bench_dir>/report.md``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PHASE_LABELS = {
    "split_dialogue": "Split dialogue",
    "separate_dialogue": "DuplexChat (DialogueSidon)",
    "sommelier": "Sommelier",
    "cholimex": "Cholimex refine",
}
PIPELINES = {
    "DuplexChat": ["split_dialogue", "separate_dialogue"],
    "Sommelier": ["split_dialogue", "sommelier"],
    "Cholimex": ["split_dialogue", "separate_dialogue", "cholimex"],
}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def _per_hour(seconds: float, audio_seconds: float | None) -> float | None:
    return seconds / (audio_seconds / 3600) if audio_seconds else None


def phase_row(phase: dict, raw_seconds: float | None) -> dict:
    fresh = [item for item in phase["items"] if item.get("status") == "complete" and not item.get("resumed")]
    rtfs = [item["real_time_factor"] for item in fresh if item.get("real_time_factor") is not None]
    audio = phase.get("audio_seconds")
    wall = phase["elapsed_seconds"]
    rtf = wall / audio if audio else None
    return {
        "step": phase["step"],
        "label": PHASE_LABELS.get(phase["step"], phase["step"]),
        "files": phase["item_count"],
        "failed": phase["failed_item_count"],
        "resumed": sum(bool(item.get("resumed")) for item in phase["items"]),
        "audio_hours": audio / 3600 if audio else None,
        "wall_seconds": wall,
        "seconds_per_audio_hour": _per_hour(wall, audio),
        "seconds_per_raw_hour": _per_hour(wall, raw_seconds),
        "rtf": rtf,
        "x_realtime": 1 / rtf if rtf else None,
        "file_rtf_p50": _percentile(rtfs, 50),
        "file_rtf_p95": _percentile(rtfs, 95),
        "gpu": phase.get("gpu"),
        "resource_plan": phase.get("resource_plan"),
    }


def substep_rows(steps_path: Path) -> list[dict]:
    if not steps_path.is_file():
        return []
    totals: dict[tuple[str, str], dict] = {}
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("phase") or "", row["step"])
        total = totals.setdefault(key, {"phase": key[0], "step": key[1], "calls": 0, "elapsed_seconds": 0.0, "audio_seconds": 0.0})
        total["calls"] += 1
        total["elapsed_seconds"] += row["elapsed_seconds"]
        total["audio_seconds"] += row.get("audio_seconds") or 0.0
    for total in totals.values():
        total["seconds_per_audio_hour"] = _per_hour(total["elapsed_seconds"], total["audio_seconds"])
        total["rtf"] = total["elapsed_seconds"] / total["audio_seconds"] if total["audio_seconds"] else None
    return sorted(totals.values(), key=lambda item: (item["phase"], item["step"]))


def summarize(bench_dir: Path) -> Path:
    bench_dir = Path(bench_dir)
    sample = json.loads((bench_dir / "sample.json").read_text(encoding="utf-8"))
    timing = json.loads((bench_dir / "timing.json").read_text(encoding="utf-8"))
    raw_seconds = sample.get("audio_seconds")
    latest = {phase["step"]: phase for phase in timing["phases"]}  # a rerun replaces the earlier phase
    phases = [phase_row(phase, raw_seconds) for phase in latest.values()]
    by_step = {row["step"]: row for row in phases}
    pipelines = []
    for name, steps in PIPELINES.items():
        if all(step in by_step for step in steps):
            pipelines.append({
                "pipeline": name, "steps": steps,
                "seconds_per_raw_hour": sum(by_step[step]["seconds_per_raw_hour"] or 0.0 for step in steps),
            })
    report = {
        "run_id": timing.get("workflow_id"), "sample": {key: value for key, value in sample.items() if key != "files"},
        "phases": phases, "substeps": substep_rows(bench_dir / "steps.jsonl"), "pipelines": pipelines,
    }
    (bench_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = bench_dir / "report.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def _fmt(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:,.{digits}f}"


def render_markdown(report: dict) -> str:
    sample = report["sample"]
    lines = [
        f"# Speed benchmark {report['run_id']}", "",
        f"Sample: {sample['count']} raw files (seed {sample['seed']}), {_fmt(sample['audio_hours'], 2)} h audio.", "",
        "## Steps", "",
        "| Step | Files (fail) | Audio in (h) | Wall (s) | s / 1 h audio | s / 1 h raw | RTF | × realtime | file RTF p50 / p95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["phases"]:
        lines.append(
            f"| {row['label']} | {row['files']} ({row['failed']}) | {_fmt(row['audio_hours'], 2)} | {_fmt(row['wall_seconds'], 0)} "
            f"| {_fmt(row['seconds_per_audio_hour'])} | {_fmt(row['seconds_per_raw_hour'])} | {_fmt(row['rtf'], 4)} "
            f"| {_fmt(row['x_realtime'])} | {_fmt(row['file_rtf_p50'], 4)} / {_fmt(row['file_rtf_p95'], 4)} |"
        )
    if report["pipelines"]:
        lines += ["", "## Pipelines end-to-end", "", "| Pipeline | Steps | s / 1 h raw |", "| --- | --- | --- |"]
        for row in report["pipelines"]:
            lines.append(f"| {row['pipeline']} | {' + '.join(row['steps'])} | {_fmt(row['seconds_per_raw_hour'])} |")
    if report["substeps"]:
        lines += [
            "", "## Sub-steps (summed per file, ignores parallelism)", "",
            "| Phase | Sub-step | Calls | Time (s) | s / 1 h audio | RTF |", "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in report["substeps"]:
            lines.append(
                f"| {row['phase']} | {row['step']} | {row['calls']} | {_fmt(row['elapsed_seconds'], 0)} "
                f"| {_fmt(row['seconds_per_audio_hour'])} | {_fmt(row['rtf'], 4)} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_dir", type=Path)
    print(summarize(parser.parse_args().bench_dir))
