"""Orchestrate the one-file, reference-free stereo benchmark and artifact output."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .activity import ActivityConfig, activity_summary, config_dict, energy_vad, mask_segments
from .audio import load_stereo
from .dynamics import (
    MAX_BACKCHANNEL_DURATION_SEC, MERGE_GAP_SEC, MIN_TURN_DURATION_SEC,
    analyze_turns, inactive_channel_energy_ratio_db,
)
from .models import acoustic_metrics, prepare_speech, speaker_metrics
from .report import flatten_report, render_tables, summarize_reports


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_benchmark(audio_path: Path, output_dir: Path, device: str = "auto", debug: bool = False, dnsmos_model_dir: Path | None = Path("models/dnsmos")) -> tuple[dict, Path]:
    """Run reference-free diagnostics for a single, two-channel WAV/audio file."""
    started = perf_counter()
    audio = load_stereo(audio_path)
    loaded_at = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = resolve_device(device)
    activity_config = ActivityConfig()
    left_mask, left_threshold = energy_vad(audio.left, audio.sample_rate, activity_config)
    right_mask, right_threshold = energy_vad(audio.right, audio.sample_rate, activity_config)
    # Frame count can differ only on malformed input; trim defensively for every paired statistic.
    frame_count = min(len(left_mask), len(right_mask))
    left_mask, right_mask = left_mask[:frame_count], right_mask[:frame_count]
    frame_sec = activity_config.frame_sec
    activity = activity_summary(left_mask, right_mask, frame_sec)
    dynamics = analyze_turns(left_mask, right_mask, frame_sec)
    activity_at = perf_counter()
    left_speech = prepare_speech(audio.left, audio.sample_rate, left_mask, frame_sec)
    right_speech = prepare_speech(audio.right, audio.sample_rate, right_mask, frame_sec)
    prepared_at = perf_counter()
    acoustic = acoustic_metrics(left_speech, right_speech, selected_device, dnsmos_model_dir)
    acoustic_at = perf_counter()
    identity = speaker_metrics(left_speech, right_speech, selected_device)
    identity_at = perf_counter()
    leakage = inactive_channel_energy_ratio_db(audio.left, audio.right, left_mask, right_mask, frame_sec, audio.sample_rate)
    diagnostics_at = perf_counter()
    report = {
        "input": {
            "path": str(audio.path.resolve()), "sha256": _sha256(audio.path), "sample_rate": audio.sample_rate,
            "duration_sec": audio.duration_sec, "samples": audio.frames, "channels": 2,
            "left": "estimated speaker track 1", "right": "estimated speaker track 2",
        },
        "acoustic_quality": acoustic,
        "speaker_identity": identity,
        "speech_activity": {"vad": "energy_vad", "left_threshold_dbfs": left_threshold, "right_threshold_dbfs": right_threshold, **activity},
        "turn_taking": {key: value for key, value in dynamics.items() if key not in {"turns", "all_talk_spurts", "transitions", "backchannel_candidates"}},
        "backchannel": {"method": "VAD-based candidate; no ASR/linguistic criterion", "count": len(dynamics["backchannel_candidates"]), "per_minute": dynamics["backchannels_per_min"], "mean_duration_sec": dynamics["mean_backchannel_duration_sec"]},
        "leakage_proxy": {"name": "inactive_channel_energy_ratio_db", **leakage},
        "runtime": {
            "timestamp_utc": datetime.now(UTC).isoformat(), "device": selected_device,
            "stage_seconds": {
                "load_audio": loaded_at - started,
                "activity_and_turns": activity_at - loaded_at,
                "prepare_speech": prepared_at - activity_at,
                "acoustic_metrics": acoustic_at - prepared_at,
                "speaker_identity": identity_at - acoustic_at,
                "diagnostics": diagnostics_at - identity_at,
            },
            "total_seconds": diagnostics_at - started,
        },
        "versions": {"benchmark": "0.1.0", "python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__},
        "config": {"activity": config_dict(activity_config), "turn_analysis": {"merge_gap_sec": MERGE_GAP_SEC, "min_turn_duration_sec": MIN_TURN_DURATION_SEC, "max_backchannel_duration_sec": MAX_BACKCHANNEL_DURATION_SEC}},
    }
    timeline = {"left_vad": mask_segments(left_mask, frame_sec, "left"), "right_vad": mask_segments(right_mask, frame_sec, "right"), "turns": dynamics["turns"], "overlaps": mask_segments(left_mask & right_mask, frame_sec), "backchannel_candidates": dynamics["backchannel_candidates"], "transitions": dynamics["transitions"]}
    report_path = output_dir / "report.json"
    timeline_path = output_dir / "timeline.json"
    _write_json(report_path, report)
    _write_json(timeline_path, timeline)
    (output_dir / "summary.md").write_text(render_tables(summarize_reports([report])) + "\n", encoding="utf-8")
    if debug:
        sf.write(output_dir / "left.wav", audio.left, audio.sample_rate)
        sf.write(output_dir / "right.wav", audio.right, audio.sample_rate)
        _write_json(output_dir / "vad_left.json", timeline["left_vad"])
        _write_json(output_dir / "vad_right.json", timeline["right_vad"])
        _write_json(output_dir / "turns.json", timeline["turns"])
        _write_json(output_dir / "overlap.json", timeline["overlaps"])
        _write_json(output_dir / "backchannels.json", timeline["backchannel_candidates"])
    return report, report_path


def discover_corpus_audio(corpus_dir: Path) -> list[Path]:
    """Recursively find audio candidates; only stereo files are eligible."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory does not exist: {root}")
    
    candidates = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
            try:
                import soundfile as sf
                if sf.info(path).channels == 2:
                    candidates.append(path)
            except Exception:
                pass
    return sorted(candidates)


def run_corpus_benchmark(corpus_dir: Path, output_dir: Path, device: str = "auto", debug: bool = False, dnsmos_model_dir: Path | None = Path("models/dnsmos")) -> tuple[dict, Path]:
    """Benchmark all audio candidates, retaining per-file failures instead of aborting a corpus."""
    started = perf_counter()
    corpus_dir, output_dir = Path(corpus_dir), Path(output_dir)
    candidates = discover_corpus_audio(corpus_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports, samples, rows = [], [], []
    import tqdm
    for index, audio_path in enumerate(tqdm.tqdm(candidates, desc="Benchmarking files")):
        source = str(audio_path.relative_to(corpus_dir))
        file_dir = output_dir / "files" / f"{index:06d}"
        try:
            report, report_path = run_benchmark(audio_path, file_dir, device, debug, dnsmos_model_dir)
            reports.append(report)
            samples.append({"source": source, "status": "ok", "report": str(report_path.relative_to(output_dir))})
            rows.append(flatten_report(report, source))
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            samples.append({"source": source, "status": "failed", "error": message})
            rows.append({"source": source, "status": "failed", "error": message})
    summary = summarize_reports(reports)
    elapsed = perf_counter() - started
    corpus_report = {
        "input": str(corpus_dir.resolve()), "candidate_count": len(candidates), "summary": summary, "samples": samples,
        "runtime": {"total_seconds": elapsed, "files_per_second": len(candidates) / elapsed if elapsed else None},
    }
    report_path = output_dir / "corpus_report.json"
    _write_json(report_path, corpus_report)
    (output_dir / "summary.md").write_text(render_tables(summary) + "\n", encoding="utf-8")
    _write_csv(output_dir / "per_file.csv", rows)
    return corpus_report, report_path


def print_summary(report: dict, report_path: Path) -> None:
    from .report import render_warnings
    print("Pipeline: Stereo Full-Duplex Benchmark\n")
    print(f"Input: {report['input']['path']}\nDuration: {report['input']['duration_sec']:.2f} s | Sample rate: {report['input']['sample_rate']} Hz | Channels: 2")
    print("\n" + render_tables(summarize_reports([report])))
    if warnings := render_warnings([report]):
        print(warnings)
    runtime = report["runtime"]
    stages = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in runtime["stage_seconds"].items())
    print(f"\nBenchmark complete in {runtime['total_seconds']:.2f}s ({stages})\nJSON report: {report_path}")


def _status(value: dict) -> str:
    return "ok" if value.get("status") == "ok" else f"N/A ({value.get('reason', 'unavailable')})"


def _format(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _format_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = sorted({key for row in rows for key in row}) or ["source", "status", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
