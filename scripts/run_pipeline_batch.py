"""Batch processing runner script for DuplexChat and Sommelier pipelines.
Supports custom input/output directory flags for convenient execution on Kaggle or local environments.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from core.runtime_cpu import enforce_single_cpu_thread
    enforce_single_cpu_thread()
except ImportError:
    pass

from core.model_utils import resolve_local_model_path
from core.resource_tuning import GpuMemorySampler, choose_workers_per_gpu, cpu_worker_budget, probe_gpus
from pipeline.duplexchat.src.duplexchat.model_options import DIARIZATION_MODELS, resolve_model_alias

TIMING_REPORT_PATH = Path("outputs/pipeline_timing.json")


def _available_gpu_count() -> int:
    """Read the host-visible GPU count before assigning explicit physical IDs."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _split_dialogue_complete(output: Path) -> bool:
    """A split is reusable only after its manifest and every declared WAV exist."""
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        dialogues = manifest["dialogues"]
        if int(manifest["dialogue_count"]) != len(dialogues):
            return False
        return all((output / row["filename"]).is_file() and (output / row["filename"]).stat().st_size > 0 for row in dialogues)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _duplexchat_complete(dialogue_dir: Path, output: Path) -> bool:
    """Require a stereo artifact for every dialogue emitted by the split phase."""
    dialogues = sorted(dialogue_dir.glob("dialogue_*.wav"))
    if not dialogues:
        return _split_dialogue_complete(dialogue_dir)
    for dialogue in dialogues:
        match = re.fullmatch(r"dialogue_(\d+)\.wav", dialogue.name)
        if match is None:
            return False
        stereo = output / f"stereo_{match.group(1)}.wav"
        if not stereo.is_file() or stereo.stat().st_size == 0:
            return False
    return True


def _cholimex_complete(output: Path, conversation_idx: int) -> bool:
    stereo = output / f"cholimex_stereo_{conversation_idx}.wav"
    return stereo.is_file() and stereo.stat().st_size > 0


def _archive_incomplete_output(output: Path) -> None:
    """Preserve failed partial artifacts before a phase is restarted."""
    if not output.exists():
        return
    archive = output.parent / ".history" / uuid.uuid4().hex / output.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    output.rename(archive)
    print(f"[RESUME] Archived incomplete output: {output} -> {archive}")


def _resumed_timing(source: Path, output: Path) -> dict:
    now = dt.datetime.now(dt.UTC).isoformat()
    return {
        "input": str(source.resolve()), "output": str(output.resolve()),
        "started_at_utc": now, "ended_at_utc": now, "elapsed_seconds": 0.0,
        "audio_seconds": _audio_duration_seconds(source), "real_time_factor": 0.0,
        "audio_seconds_per_wall_second": None, "exit_code": 0, "status": "complete", "resumed": True,
    }


def _run_gpu_items(items, gpu_list: list[str], handler, *, resource_mode: str,
                   configured_workers_per_gpu: int, reserve_ratio: float,
                   max_workers_per_gpu: int):
    """Run file-level GPU work, calibrating actual peak VRAM before expansion."""
    def run_with_tickets(batch_items, tickets):
        import queue
        ticket_queue: queue.Queue = queue.Queue()
        for ticket in tickets:
            ticket_queue.put(ticket)

        def invoke(item):
            gpu = ticket_queue.get()
            try:
                return handler(item, gpu)
            finally:
                ticket_queue.put(gpu)

        with ThreadPoolExecutor(max_workers=len(tickets)) as pool:
            return list(pool.map(invoke, batch_items))

    manual_tickets = [gpu for gpu in gpu_list for _ in range(configured_workers_per_gpu)]
    if resource_mode != "auto" or not items:
        return run_with_tickets(items, manual_tickets), {
            "mode": "manual", "workers_per_gpu": configured_workers_per_gpu,
        }

    available = {snapshot.index for snapshot in probe_gpus()}
    calibration_gpus = [gpu for gpu in gpu_list if gpu in available][:len(items)]
    if not calibration_gpus:
        return run_with_tickets(items, manual_tickets), {
            "mode": "auto", "fallback": "GPU telemetry unavailable", "workers_per_gpu": configured_workers_per_gpu,
        }

    calibration_items = items[:len(calibration_gpus)]
    sampler = GpuMemorySampler()
    sampler.start()
    with ThreadPoolExecutor(max_workers=len(calibration_items)) as pool:
        calibration_results = list(pool.map(
            lambda pair: handler(pair[1], calibration_gpus[pair[0]]),
            enumerate(calibration_items),
        ))
    snapshots = sampler.stop()
    cpu_cap = cpu_worker_budget(len(calibration_gpus))
    per_gpu = {
        gpu: choose_workers_per_gpu(
            snapshots[gpu], cpu_workers_per_gpu=cpu_cap, reserve_ratio=reserve_ratio,
            max_workers_per_gpu=max_workers_per_gpu,
        )
        for gpu in calibration_gpus if gpu in snapshots
    }
    tickets = [gpu for gpu in gpu_list for _ in range(per_gpu.get(gpu, 1))]
    peak_mib = {gpu: snapshots[gpu].peak_memory_used_mib for gpu in per_gpu}
    plan = {
        "mode": "auto", "workers_per_gpu": per_gpu, "cpu_workers_per_gpu": cpu_cap,
        "reserve_ratio": reserve_ratio, "peak_memory_mib": peak_mib,
    }
    print(f"Auto-tune: {plan}")
    return calibration_results + run_with_tickets(items[len(calibration_items):], tickets), plan


def _summarize_split_dialogue_manifests(output_dir: Path, input_audio_count: int) -> dict:
    """Build an inspectable LID and rejection audit from per-audio manifests."""
    audio = []
    counts = {
        "input_audio": input_audio_count,
        "manifest_audio": 0,
        "candidate_dialogues": 0,
        "exported_dialogues": 0,
        "rejected_by_lid": 0,
        "rejected_short": 0,
        "rejected_imbalanced": 0,
        "monologue_audio": 0,
        "zero_dialogue_audio": 0,
    }
    lid_configs = []

    for manifest_path in sorted(output_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = manifest["filter_summary"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue

        lid = summary.get("lid", {})
        lid_configs.append(lid)
        dialogue_count = int(manifest.get("dialogue_count", 0))
        counts["manifest_audio"] += 1
        counts["candidate_dialogues"] += int(summary.get("candidate_dialogue_count", 0))
        counts["exported_dialogues"] += int(summary.get("exported_dialogue_count", dialogue_count))
        counts["rejected_by_lid"] += int(summary.get("rejected_by_lid", 0))
        counts["rejected_short"] += int(summary.get("rejected_short", 0))
        counts["rejected_imbalanced"] += int(summary.get("rejected_imbalanced", 0))
        counts["monologue_audio"] += int(summary.get("diarized_speaker_count", 0) < 2)
        counts["zero_dialogue_audio"] += int(dialogue_count == 0)
        audio.append({
            "source_audio": manifest.get("source_audio"),
            "dialogue_count": dialogue_count,
            "skip_reason": manifest.get("skip_reason"),
            "filter_summary": summary,
        })

    enabled_values = {bool(config.get("enabled")) for config in lid_configs}
    lid = {"enabled": None, "model": None, "min_vi_probability": None}
    if len(enabled_values) == 1:
        lid["enabled"] = enabled_values.pop()
    if lid_configs:
        first = lid_configs[0]
        lid["model"] = first.get("model")
        lid["min_vi_probability"] = first.get("min_vi_probability")

    return {"step": "split_dialogue", "lid": lid, "counts": counts, "audio": audio}


def _write_split_dialogue_report(output_dir: Path, report: dict) -> Path:
    """Atomically persist the split-dialogue audit beside its manifests."""
    path = output_dir / "split_dialogue_report.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _print_split_dialogue_summary(report: dict, report_path: Path) -> None:
    """Print the batch-level LID decision summary and every zero-output audio."""
    lid = report["lid"]
    counts = report["counts"]
    status = {True: "ENABLED", False: "DISABLED", None: "UNKNOWN"}[lid["enabled"]]
    print("=== Split dialogue filter summary ===")
    print(
        f"LID: {status} | model={lid['model'] or '-'} | "
        f"minimum_vi_probability={lid['min_vi_probability'] if lid['min_vi_probability'] is not None else '-'}"
    )
    print(
        "Audio: {input_audio} input, {manifest_audio} completed manifests, {zero_dialogue_audio} produced 0 clips".format(**counts)
    )
    print(
        "Clips: {candidate_dialogues} candidates, {exported_dialogues} exported, "
        "{rejected_by_lid} rejected by LID, {rejected_short} short, {rejected_imbalanced} imbalanced".format(**counts)
    )
    for item in report["audio"]:
        if item["dialogue_count"] == 0:
            print(f"[REJECTED] {Path(str(item['source_audio'])).name}: {item['skip_reason'] or 'no accepted dialogue'}")
    print(f"Split dialogue audit: {report_path}")


def _check_model_status(name: str, raw_id: str | None, env_var: str | None = None, default_sub: str | None = None, required_files: list[str] | None = None) -> tuple[str, str, bool]:
    """Check if model exists locally or is configured, and return (display_str, status_str, is_valid)."""
    if not raw_id and not env_var:
        return ("(none)", "NOT CONFIGURED", False)
    
    resolved, is_local = resolve_local_model_path(raw_id, env_var=env_var, default_subpath=default_sub)
    if not resolved:
        return ("(none)", "NOT CONFIGURED", False)
    
    p = Path(resolved)
    if p.exists():
        if required_files:
            missing = [f for f in required_files if not (p / f).exists()]
            if missing:
                return (str(p), f"INCOMPLETE (missing {', '.join(missing)})", False)
        return (str(p), "VALID (local)", True)
    
    from core.model_utils import is_offline_mode
    if is_offline_mode():
        return (str(resolved), "NOT FOUND (Offline mode requires local files)", False)
    return (str(resolved), "CONFIGURED (Remote HuggingFace Hub)", True)


def _log_preflight_info(args, input_dir: Path, output_dir: Path) -> None:
    """Print available models with validity status and an ordered flow tree (1, 2, 3...)."""
    from core.model_utils import is_offline_mode
    mode_str = "SEVER (Offline / Local weights required)" if is_offline_mode() else "DEV (Online / HuggingFace Hub download allowed)"
    print("\n" + "=" * 70)
    print(f" PIPELINE PREFLIGHT CHECK: Step [{args.step}] (GPU {args.gpu}) | Mode: [{mode_str}]")
    print("=" * 70)

    # 1. Models check
    print("\n[1] Models Configuration & Availability:")
    models_to_check = []
    
    if args.step == "split_dialogue":
        diar_model = resolve_model_alias(args.diarization_model, DIARIZATION_MODELS) or args.diarization_model or "pyannote/speaker-diarization-community-1"
        backend = (args.diarization_backend or "auto").lower()
        if "sortformer" in backend or "sortformer" in diar_model.lower():
            models_to_check.append(("Diarization (Sortformer)", diar_model, "SORTFORMER_MODEL_PATH", "diar_streaming_sortformer_4spk-v2.1", None))
        else:
            models_to_check.append(("Diarization (Pyannote)", diar_model, "PYANNOTE_MODEL_PATH", "speaker-diarization-community-1", None))
            
        if args.filter_music:
            models_to_check.append(("Music Filter (Demucs)", "htdemucs", "DEMUCS_MODEL_PATH", "htdemucs", None))
        else:
            print("  ├── Music Filter (Demucs)       : DISABLED (--no-filter-music)")

        if args.lid:
            models_to_check.append(("Language ID (Whisper LID)", "openai/whisper-small", "WHISPER_MODEL_PATH", "whisper-small", ["config.json"]))
        else:
            print("  ├── Language ID (Whisper LID)   : DISABLED")

    elif args.step == "separate_dialogue":
        sep_model = str(args.separation_model.resolve()) if args.separation_model else "sarulab-speech/DialogueSidon"
        req = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
        models_to_check.append(("Dialogue Separation (Sidon)", sep_model, "DIALOGUESIDON_MODEL_PATH", "DialogueSidon", req))

    elif args.step == "sommelier":
        sepreformer_ckpt = os.environ.get("SEPREFORMER")
        models_to_check.append(("SepReformer Checkpoint", sepreformer_ckpt, "SEPREFORMER", None, None))
        sort_model = os.environ.get("SORTFORMER_MODEL_PATH") or "nvidia/diar_streaming_sortformer_4spk-v2.1"
        models_to_check.append(("Sortformer Diarization", sort_model, "SORTFORMER_MODEL_PATH", "diar_streaming_sortformer_4spk-v2.1", None))
        models_to_check.append(("Speaker Embedding", "speechbrain/spkrec-ecapa-voxceleb", "SPEECHBRAIN_MODEL_PATH", "spkrec-ecapa-voxceleb", None))
        hf_token_set = bool(os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"))
        token_status = "CONFIGURED" if hf_token_set else "OPTIONAL (Local models used in offline mode)"
        print(f"  ├── Hugging Face Token          : {token_status}")

    for idx, (label, raw_id, env_var, default_sub, req_files) in enumerate(models_to_check):
        prefix = "  └── " if idx == len(models_to_check) - 1 else "  ├── "
        disp, status, is_valid = _check_model_status(label, raw_id, env_var, default_sub, req_files)
        icon = "✓" if is_valid else "✗"
        print(f"{prefix}{label:<28}: [{icon}] {status} -> {disp}")

    # 2. Execution Flow Tree
    print("\n[2] Execution Flow Tree:")
    if args.step == "split_dialogue":
        print("  1. Standardize Audio Input (Resample to 16kHz mono, normalize volume)")
        print("  2. Speaker Diarization (Identify speaker segments across time)")
        print("  3. Dialogue Filtering & Grouping (Filter valid 2-speaker conversational dialogue chunks)")
        music_status = "Apply Demucs background music removal" if args.filter_music else "Skipped (--no-filter-music)"
        print(f"  4. Music Filtering ({music_status})")
        lid_status = f"Whisper LID filter for language '{args.lid}' (keep probability >= {args.min_lid_prob})" if args.lid else "Skipped (no --lid)"
        print(f"  5. Language Identification ({lid_status})")
        print(f"  6. Export Dialogue WAVs & Manifest (Save dialogue_*.wav into subfolder for each audio)")

    elif args.step == "separate_dialogue":
        print("  1. Scan Input Subfolders (Collect dialogue_*.wav from each dialogue folder)")
        print(f"  2. Chunk Audio Windows (Slice into {args.separation_chunk}s overlapping chunks)")
        print("  3. DialogueSidon Diffusion Model (Separate overlapping voices into 2 isolated speaker tracks)")
        print("  4. Stitch & Reconstruct Stereo (Resample to 24kHz, balance channels, export audio.stereo.wav)")

    elif args.step == "sommelier":
        print("  1. Preprocess mixture (Standardize to 16kHz mono WAV)")
        print("  2. Sortformer Diarization & VAD (Detect voice activity and speaker turns)")
        print("  3. SepReformer Separation (Disentangle overlapping speaker voices)")
        print("  4. Constrain Speakers (Enforce 2-speaker linkage across entire podcast)")
        print("  5. Reconstruct Tracks (Recombine separated MP3 segments into full 24kHz stereo WAV)")
    elif args.step == "cholimex":
        print("  1. Pair each DuplexChat stereo_N.wav with dialogue_N.wav mixture")
        print("  2. VAD mask the two estimated tracks")
        print("  3. Preserve separated overlap and restore non-overlap mixture speech")

    print(f"\n  • Input Dir  : {input_dir}")
    print(f"  • Output Dir : {output_dir}")
    print("=" * 70 + "\n")


def _audio_duration_seconds(path: Path) -> float | None:
    """Return source audio duration when SoundFile can inspect it, else None."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / info.samplerate if info.samplerate and info.frames > 0 else None
    except Exception:
        return None


def _timed_subprocess(cmd: list[str], source: Path, output: Path, env: dict[str, str]) -> dict:
    started_at = dt.datetime.now(dt.UTC)
    started = perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)
    elapsed = perf_counter() - started
    audio_seconds = _audio_duration_seconds(source)
    return {
        "input": str(source.resolve()),
        "output": str(output.resolve()),
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
        "audio_seconds_per_wall_second": audio_seconds / elapsed if audio_seconds and elapsed else None,
        "exit_code": result.returncode,
        "status": "complete" if result.returncode == 0 else "failed",
    }


def _phase_report(step: str, args, started_at: dt.datetime, elapsed: float, items: list[dict], resource_plan: dict | None = None) -> dict:
    audio_seconds = sum(item["audio_seconds"] for item in items if item["audio_seconds"] is not None)
    completed = sum(item["status"] == "complete" for item in items)
    return {
        "step": step,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "gpu": args.gpu,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "item_count": len(items),
        "completed_item_count": completed,
        "failed_item_count": len(items) - completed,
        "audio_seconds": audio_seconds if any(item["audio_seconds"] is not None for item in items) else None,
        "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
        "audio_seconds_per_wall_second": audio_seconds / elapsed if audio_seconds and elapsed else None,
        "resource_plan": resource_plan,
        "items": items,
    }


def _write_timing_report(workflow_id: str, phase: dict) -> Path:
    path = TIMING_REPORT_PATH.resolve()
    existing: dict = {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if existing.get("workflow_id") != workflow_id:
        existing = {"schema_version": 1, "workflow_id": workflow_id, "started_at_utc": phase["started_at_utc"], "phases": []}
    existing["phases"].append(phase)
    phases = existing["phases"]
    total_elapsed = sum(item["elapsed_seconds"] for item in phases)
    total_audio = sum(item["audio_seconds"] or 0.0 for item in phases)
    all_items = [{"step": phase_item["step"], **item} for phase_item in phases for item in phase_item["items"]]
    existing["ended_at_utc"] = phase["ended_at_utc"]
    existing["summary"] = {
        "phase_count": len(phases),
        "item_count": len(all_items),
        "completed_item_count": sum(item["status"] == "complete" for item in all_items),
        "failed_item_count": sum(item["status"] == "failed" for item in all_items),
        "elapsed_seconds": total_elapsed,
        "audio_seconds": total_audio or None,
        "real_time_factor": total_elapsed / total_audio if total_audio else None,
        "audio_seconds_per_wall_second": total_audio / total_elapsed if total_audio and total_elapsed else None,
    }
    existing["slowest_phases"] = sorted(
        ({"step": item["step"], "elapsed_seconds": item["elapsed_seconds"]} for item in phases),
        key=lambda item: item["elapsed_seconds"], reverse=True,
    )
    existing["slowest_items"] = sorted(all_items, key=lambda item: item["elapsed_seconds"], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _print_timing_summary(phase: dict, report_path: Path) -> None:
    speed = phase["audio_seconds_per_wall_second"]
    speed_text = f" | speed={speed:.3f}x" if speed is not None else ""
    print(f"Timing: {phase['step']}={phase['elapsed_seconds']:.2f}s{speed_text} -> {report_path}")


def run_batch():
    parser = argparse.ArgumentParser(description="Batch process audio files across pipelines with custom input/output directories.")
    parser.add_argument("--step", "-s", choices=["split_dialogue", "separate_dialogue", "sommelier", "cholimex"], required=True,
                        help="Pipeline step to execute.")
    parser.add_argument("--input-dir", "-i", type=Path, required=True,
                        help="Input directory containing audio WAV files or dialogue subfolders (e.g., /kaggle/input/... or data/raw/youtube).")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory to save processed results (e.g., /kaggle/working/data/processed/...).")
    parser.add_argument("--mixture-dir", type=Path,
                        help="Dialogue-filter output root used to pair dialogue_N.wav with DuplexChat stereo_N.wav for Cholimex.")
    parser.add_argument("--gpu", "-g", type=str, default="0",
                        help="Target GPU device ID (e.g., 0 or 1). Default: 0")
    parser.add_argument("--pattern", type=str, default="*.wav",
                        help="File glob pattern for audio inputs. Default: *.wav")
    parser.add_argument("--filter-music", action="store_true", default=True,
                        help="Enable Demucs music filtering in split_dialogue step.")
    parser.add_argument("--no-filter-music", dest="filter_music", action="store_false",
                        help="Disable Demucs music filtering.")
    parser.add_argument("--lid", type=str, default=None,
                        help="Enable Language Identification filter (e.g., 'vi' for Vietnamese).")
    parser.add_argument("--min-lid-prob", type=float, default=0.5,
                        help="Keep a LID candidate only when its Vietnamese probability meets this threshold.")
    parser.add_argument("--diarization-backend", type=str, default=None,
                        help="Diarization backend (e.g. pyannote, sortformer, diarizen, auto).")
    parser.add_argument("--diarization-model", type=str, default=None,
                        help="Diarization model ID or alias (e.g. nvidia/diar_streaming_sortformer_4spk-v2.1).")
    parser.add_argument("--diarize-chunk", type=str, default=None,
                        help="Max chunk duration in seconds for diarization (e.g. 120.0, 600.0, or 'full').")
    parser.add_argument("--separation-chunk", "--separate-chunk", dest="separation_chunk", type=float, default=120.0,
                        help="Chunk duration in seconds for dialogue separation (default: 120.0).")
    parser.add_argument("--separation-model", type=Path, default=None,
                        help="Verified local DialogueSidon model directory passed to every separation process.")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Number of parallel worker processes to saturate GPU (default: 2 for A100 40GB).")
    parser.add_argument("--resource-mode", choices=("auto", "manual"), default="auto",
                        help="Auto-calibrate workers from observed VRAM, or use --workers unchanged.")
    parser.add_argument("--gpu-memory-reserve-ratio", type=float, default=0.10,
                        help="Fraction of total VRAM reserved during auto-tuning (default: 0.10).")
    parser.add_argument("--max-workers-per-gpu", type=int, default=8,
                        help="Hard safety cap for auto-tuned file workers per GPU.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print executable commands without running them.")
    args = parser.parse_args()
    workflow_id = os.environ.get("PIPELINE_RUN_ID") or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    phase_started_at = dt.datetime.now(dt.UTC)
    phase_started = perf_counter()
    timing_items: list[dict] = []

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    mixture_dir = args.mixture_dir.resolve() if args.mixture_dir else None

    if args.step == "cholimex" and mixture_dir is None:
        parser.error("--mixture-dir is required when --step cholimex")
    if mixture_dir is not None and args.step == "cholimex" and not mixture_dir.is_dir():
        print(f"[INFO] Mixture directory does not exist: {mixture_dir}. Skipping Cholimex.")
        return

    if not input_dir.exists():
        print(f"[INFO] Input directory does not exist: {input_dir}. Skipping step {args.step}.")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[WARNING] Could not create output directory {output_dir}: {e}. Skipping step {args.step}.")
        return

    _log_preflight_info(args, input_dir, output_dir)

    env = dict(os.environ)
    gpu_str = str(args.gpu).strip()

    # Resolve GPU devices intelligently
    if gpu_str.lower() in ("auto", "all"):
        try:
            n_gpus = _available_gpu_count()
            if n_gpus > 0:
                gpu_list = [str(i) for i in range(n_gpus)]
            else:
                gpu_list = ["0"]
        except Exception:
            gpu_list = ["0"]
    else:
        raw_gpus = [g.strip() for g in gpu_str.split(",") if g.strip()]
        try:
            n_gpus = _available_gpu_count()
            if n_gpus > 0:
                invalid = [g for g in raw_gpus if g.isdigit() and int(g) >= n_gpus]
                if invalid:
                    parser.error(f"Requested GPU(s) {', '.join(invalid)} are outside host range 0..{n_gpus - 1}")
                gpu_list = list(dict.fromkeys(raw_gpus))
            else:
                gpu_list = raw_gpus
        except Exception:
            gpu_list = raw_gpus

    gpu_label = ",".join(gpu_list)
    num_gpus = len(gpu_list)
    workers_per_gpu = max(1, args.workers)
    total_workers = workers_per_gpu * num_gpus if num_gpus > 1 else workers_per_gpu
    if not 0.0 <= args.gpu_memory_reserve_ratio < 1.0:
        parser.error("--gpu-memory-reserve-ratio must be in [0, 1)")
    if args.max_workers_per_gpu < 1:
        parser.error("--max-workers-per-gpu must be positive")

    # Limit CPU thread pinning to avoid 100% CPU spikes across massive cores
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    resource_plan = None
    import queue
    def _create_gpu_queue() -> queue.Queue:
        q: queue.Queue = queue.Queue()
        for gpu in gpu_list:
            for _ in range(workers_per_gpu):
                q.put(gpu)
        return q

    if args.step == "split_dialogue":
        wav_files = sorted(list(input_dir.glob(args.pattern)))
        if not wav_files:
            print(f"No files matching '{args.pattern}' found in {input_dir}")
            return

        print(f"=== Running split_dialogue on {len(wav_files)} files from {input_dir} (GPU {gpu_label}, workers={total_workers}) ===")
        if args.dry_run:
            for idx, wav in enumerate(wav_files, start=1):
                sub_out = output_dir / wav.stem
                target_gpu = gpu_list[(idx - 1) % len(gpu_list)]
                dev_args = ["--device-ids", "0"] if target_gpu != "cpu" else ["--device-ids", "cpu"]
                gpu_tag = f"[GPU {target_gpu}] " if target_gpu != "cpu" else "[CPU] "
                filter_flag = "--filter-music" if args.filter_music else "--no-filter-music"
                cmd = [
                    sys.executable, "-m", "duplexchat", "split_valid_dialogue",
                    "--input", str(wav),
                    "--output-dir", str(sub_out),
                    *dev_args,
                    filter_flag
                ]
                if args.lid:
                    cmd.extend(["--lid", args.lid, "--min-lid-prob", str(args.min_lid_prob)])
                if args.diarization_backend:
                    cmd.extend(["--diarization-backend", args.diarization_backend])
                if args.diarization_model:
                    cmd.extend(["--diarization-model", args.diarization_model])
                if args.diarize_chunk:
                    cmd.extend(["--diarize-chunk", str(args.diarize_chunk)])
                print(f"[{idx}/{len(wav_files)}] {gpu_tag}{wav.name} -> {sub_out.name}")
                print("  Command:", " ".join(cmd))
        else:
            def _process_one_wav(item, target_gpu):
                idx, wav = item
                sub_out = output_dir / wav.stem
                if _split_dialogue_complete(sub_out):
                    print(f"[RESUMED] {wav.name}: complete split output at {sub_out}")
                    return _resumed_timing(wav, sub_out)
                _archive_incomplete_output(sub_out)
                worker_env = dict(env)
                if target_gpu != "cpu":
                    worker_env["CUDA_VISIBLE_DEVICES"] = target_gpu
                    dev_args = ["--device-ids", "0"]
                    gpu_tag = f"[GPU {target_gpu}] "
                else:
                    dev_args = ["--device-ids", "cpu"]
                    gpu_tag = "[CPU] "

                filter_flag = "--filter-music" if args.filter_music else "--no-filter-music"
                cmd = [sys.executable, "-m", "duplexchat", "split_valid_dialogue", "--input", str(wav), "--output-dir", str(sub_out), *dev_args, filter_flag]
                if args.lid:
                    cmd.extend(["--lid", args.lid, "--min-lid-prob", str(args.min_lid_prob)])
                if args.diarization_backend:
                    cmd.extend(["--diarization-backend", args.diarization_backend])
                if args.diarization_model:
                    cmd.extend(["--diarization-model", args.diarization_model])
                if args.diarize_chunk:
                    cmd.extend(["--diarize-chunk", str(args.diarize_chunk)])
                print(f"[{idx}/{len(wav_files)}] {gpu_tag}{wav.name} -> {sub_out.name}")
                timing = _timed_subprocess(cmd, wav, sub_out, worker_env)
                if timing["exit_code"] != 0:
                    print(f"Error processing {wav.name} (exit code {timing['exit_code']})")
                elif not list(sub_out.glob("dialogue_*.wav")):
                    manifest_path = sub_out / "manifest.json"
                    reason = "no dialogue clips passed filter"
                    if manifest_path.exists():
                        try:
                            reason = json.loads(manifest_path.read_text(encoding="utf-8")).get("skip_reason") or reason
                        except (OSError, json.JSONDecodeError):
                            pass
                    print(f"[WARNING] '{wav.name}' produced 0 dialogue clips (Reason: {reason})")
                return timing

            timing_items, resource_plan = _run_gpu_items(
                list(enumerate(wav_files, start=1)), gpu_list, _process_one_wav,
                resource_mode=args.resource_mode, configured_workers_per_gpu=workers_per_gpu,
                reserve_ratio=args.gpu_memory_reserve_ratio, max_workers_per_gpu=args.max_workers_per_gpu,
            )

    elif args.step == "separate_dialogue":
        # Can take a directory containing dialogue subfolders or direct dialogue files
        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        if not subdirs:
            if list(input_dir.glob("dialogue_*.wav")):
                subdirs = [input_dir]
            else:
                print(f"No dialogue subdirectories or files found in {input_dir}")
                return

        print(f"=== Running separate_dialogue on {len(subdirs)} dialogue folders from {input_dir} (GPU {gpu_label}, workers={total_workers}) ===")
        if args.dry_run:
            for idx, sdir in enumerate(subdirs, start=1):
                sub_out = output_dir / sdir.name
                target_gpu = gpu_list[(idx - 1) % len(gpu_list)]
                dev_args = ["--device-ids", "0"] if target_gpu != "cpu" else ["--device-ids", "cpu"]
                gpu_tag = f"[GPU {target_gpu}] " if target_gpu != "cpu" else "[CPU] "
                cmd = [
                    sys.executable, "-m", "duplexchat", "separate_dialogue",
                    "--input", str(sdir),
                    "--output-dir", str(sub_out),
                    *dev_args,
                    "--separation-chunk", str(args.separation_chunk)
                ]
                if args.separation_model:
                    cmd.extend(["--separation-model", str(args.separation_model.resolve())])
                print(f"[{idx}/{len(subdirs)}] {gpu_tag}{sdir.name} -> {sub_out.name}")
                print("  Command:", " ".join(cmd))
        else:
            def _process_one_subdir(item, target_gpu):
                idx, sdir = item
                sub_out = output_dir / sdir.name
                source_files = sorted(sdir.glob("dialogue_*.wav"))
                if not source_files:
                    print(f"[WARNING] Skipping '{sdir.name}': contains 0 dialogue files (filtered out in Phase 1).")
                    return {
                        "input": str(sdir.resolve()),
                        "output": str(sub_out.resolve()),
                        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "ended_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "elapsed_seconds": 0.0,
                        "audio_seconds": 0.0,
                        "real_time_factor": None,
                        "audio_seconds_per_wall_second": None,
                        "exit_code": 0,
                        "status": "skipped",
                    }

                if _duplexchat_complete(sdir, sub_out):
                    print(f"[RESUMED] {sdir.name}: complete DuplexChat output at {sub_out}")
                    return _resumed_timing(sdir, sub_out)
                _archive_incomplete_output(sub_out)

                worker_env = dict(env)
                if target_gpu != "cpu":
                    worker_env["CUDA_VISIBLE_DEVICES"] = target_gpu
                    dev_args = ["--device-ids", "0"]
                    gpu_tag = f"[GPU {target_gpu}] "
                else:
                    dev_args = ["--device-ids", "cpu"]
                    gpu_tag = "[CPU] "

                cmd = [sys.executable, "-m", "duplexchat", "separate_dialogue", "--input", str(sdir), "--output-dir", str(sub_out), *dev_args, "--separation-chunk", str(args.separation_chunk)]
                if args.separation_model:
                    cmd.extend(["--separation-model", str(args.separation_model.resolve())])
                print(f"[{idx}/{len(subdirs)}] {gpu_tag}{sdir.name} -> {sub_out.name}")
                source = source_files[0] if len(source_files) == 1 else sdir
                timing = _timed_subprocess(cmd, source, sub_out, worker_env)
                if len(source_files) > 1:
                    durations = [_audio_duration_seconds(path) for path in source_files]
                    audio_seconds = sum(value for value in durations if value is not None)
                    timing["audio_seconds"] = audio_seconds or None
                    timing["real_time_factor"] = timing["elapsed_seconds"] / audio_seconds if audio_seconds else None
                    timing["audio_seconds_per_wall_second"] = audio_seconds / timing["elapsed_seconds"] if audio_seconds and timing["elapsed_seconds"] else None
                if timing["exit_code"] != 0:
                    print(f"Error processing {sdir.name} (exit code {timing['exit_code']})")
                return timing

            timing_items, resource_plan = _run_gpu_items(
                list(enumerate(subdirs, start=1)), gpu_list, _process_one_subdir,
                resource_mode=args.resource_mode, configured_workers_per_gpu=workers_per_gpu,
                reserve_ratio=args.gpu_memory_reserve_ratio, max_workers_per_gpu=args.max_workers_per_gpu,
            )

    elif args.step == "sommelier":
        def _dialogue_key(p: Path) -> int:
            m = re.search(r"dialogue_(\d+)", p.stem)
            return int(m.group(1)) if m else 0

        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        if subdirs:
            print(f"=== Running Sommelier on {len(subdirs)} dialogue folders from {input_dir} (GPU {gpu_label}, workers={total_workers}) ===")
            if args.dry_run:
                for idx, sdir in enumerate(subdirs, start=1):
                    sub_out = output_dir / sdir.name
                    target_gpu = gpu_list[(idx - 1) % len(gpu_list)]
                    gpu_tag = f"[GPU {target_gpu}] " if target_gpu != "cpu" else "[CPU] "
                    dialogue_files = sorted(list(sdir.glob("dialogue_*.wav")), key=_dialogue_key)
                    if not dialogue_files:
                        dialogue_files = sorted(list(sdir.glob(args.pattern)))
                    if not dialogue_files:
                        print(f"[{idx}/{len(subdirs)}] {gpu_tag}No audio clips found in {sdir.name}")
                        continue
                    print(f"[{idx}/{len(subdirs)}] {gpu_tag}{sdir.name} ({len(dialogue_files)} dialogues) -> {sub_out.name}")
                    for d_idx, dwav in enumerate(dialogue_files, start=1):
                        cmd = [
                            sys.executable, "-m", "sommelier", "single",
                            "--input", str(dwav),
                            "--output-dir", str(sub_out)
                        ]
                        print(f"  [{d_idx}/{len(dialogue_files)}] {dwav.name} -> {sub_out.name}")
                        print("    Command:", " ".join(cmd))
            else:
                gpu_queue = _create_gpu_queue()

                def _process_sommelier_subdir(item):
                    idx, sdir = item
                    sub_out = output_dir / sdir.name
                    sub_out.mkdir(parents=True, exist_ok=True)
                    target_gpu = gpu_queue.get()
                    try:
                        worker_env = dict(env)
                        if target_gpu != "cpu":
                            worker_env["CUDA_VISIBLE_DEVICES"] = target_gpu
                            gpu_tag = f"[GPU {target_gpu}] "
                        else:
                            gpu_tag = "[CPU] "

                        dialogue_files = sorted(list(sdir.glob("dialogue_*.wav")), key=_dialogue_key)
                        if not dialogue_files:
                            dialogue_files = sorted(list(sdir.glob(args.pattern)))
                        if not dialogue_files:
                            print(f"[{idx}/{len(subdirs)}] {gpu_tag}No audio clips found in {sdir.name}")
                            return []

                        print(f"[{idx}/{len(subdirs)}] {gpu_tag}{sdir.name} ({len(dialogue_files)} dialogues) -> {sub_out.name}")
                        sub_timings = []
                        for d_idx, dwav in enumerate(dialogue_files, start=1):
                            cmd = [
                                sys.executable, "-m", "sommelier", "single",
                                "--input", str(dwav),
                                "--output-dir", str(sub_out)
                            ]
                            print(f"  [{d_idx}/{len(dialogue_files)}] {dwav.name} -> {sub_out.name}")
                            timing = _timed_subprocess(cmd, dwav, sub_out, worker_env)
                            sub_timings.append(timing)
                            if timing["exit_code"] != 0:
                                print(f"Error processing {dwav.name} (exit code {timing['exit_code']})")
                        return sub_timings
                    finally:
                        gpu_queue.put(target_gpu)

                with ThreadPoolExecutor(max_workers=max(1, total_workers)) as pool:
                    results = list(pool.map(_process_sommelier_subdir, enumerate(subdirs, start=1)))
                    for r in results:
                        timing_items.extend(r)
        else:
            dialogue_files = sorted(list(input_dir.glob("dialogue_*.wav")), key=_dialogue_key)
            if not dialogue_files:
                dialogue_files = sorted(list(input_dir.glob(args.pattern)))
            if not dialogue_files:
                print(f"No files matching '{args.pattern}' found in {input_dir}")
                return

            print(f"=== Running Sommelier on {len(dialogue_files)} files from {input_dir} (GPU {args.gpu}, workers={args.workers}) ===")
            if args.dry_run:
                for idx, wav in enumerate(dialogue_files, start=1):
                    sub_out = output_dir / wav.stem
                    cmd = [
                        sys.executable, "-m", "sommelier", "single",
                        "--input", str(wav),
                        "--output-dir", str(sub_out)
                    ]
                    print(f"[{idx}/{len(dialogue_files)}] {wav.name} -> {sub_out.name}")
                    print("  Command:", " ".join(cmd))
            else:
                def _process_sommelier_single(item):
                    idx, wav = item
                    sub_out = output_dir / wav.stem
                    sub_out.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable, "-m", "sommelier", "single",
                        "--input", str(wav),
                        "--output-dir", str(sub_out)
                    ]
                    print(f"[{idx}/{len(dialogue_files)}] {wav.name} -> {sub_out.name}")
                    timing = _timed_subprocess(cmd, wav, sub_out, env)
                    if timing["exit_code"] != 0:
                        print(f"Error processing {wav.name} (exit code {timing['exit_code']})")
                    return timing

                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                    timing_items = list(pool.map(_process_sommelier_single, enumerate(dialogue_files, start=1)))

    elif args.step == "cholimex":
        assert mixture_dir is not None
        pairs = []
        missing_mixtures = []
        for stereo in sorted(input_dir.rglob("stereo_*.wav")):
            match = re.fullmatch(r"stereo_(\d+)\.wav", stereo.name)
            if match is None:
                continue
            relative_parent = stereo.parent.relative_to(input_dir)
            mixture = mixture_dir / relative_parent / f"dialogue_{match.group(1)}.wav"
            if not mixture.is_file():
                missing_mixtures.append((stereo, mixture))
                continue
            output = output_dir / relative_parent / f"cholimex_{match.group(1)}"
            pairs.append((stereo, mixture, output, int(match.group(1))))

        for stereo, mixture in missing_mixtures:
            print(f"[WARNING] Skipping {stereo.relative_to(input_dir)}: matching mixture is missing at {mixture}")
        if not pairs:
            print(f"No DuplexChat stereo files with matching dialogue mixtures found in {input_dir}")
            return

        print(f"=== Running Cholimex refinement on {len(pairs)} DuplexChat clips from {input_dir} (GPU {gpu_label}, workers={total_workers}) ===")
        if args.dry_run:
            for idx, (stereo, mixture, output, _conversation_idx) in enumerate(pairs, start=1):
                print(f"[{idx}/{len(pairs)}] {stereo.name} + {mixture.name} -> {output}")
                print("  Command:", " ".join([
                    sys.executable, "-m", "cholimex", "collection", "--input", str(stereo),
                    "--mixture", str(mixture), "--output-dir", str(output),
                ]))
        else:
            gpu_queue = _create_gpu_queue()

            def _process_cholimex(item):
                idx, (stereo, mixture, output, conversation_idx) = item
                target_gpu = gpu_queue.get()
                try:
                    if _cholimex_complete(output, conversation_idx):
                        print(f"[RESUMED] {stereo.name}: complete Cholimex output at {output}")
                        return {
                            "input": str(stereo.resolve()), "output": str(output.resolve()),
                            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                            "ended_at_utc": dt.datetime.now(dt.UTC).isoformat(), "elapsed_seconds": 0.0,
                            "audio_seconds": _audio_duration_seconds(stereo), "real_time_factor": 0.0,
                            "audio_seconds_per_wall_second": None, "exit_code": 0, "status": "complete", "resumed": True,
                        }
                    _archive_incomplete_output(output)
                    worker_env = dict(env)
                    if target_gpu != "cpu":
                        worker_env["CUDA_VISIBLE_DEVICES"] = target_gpu
                        gpu_tag = f"[GPU {target_gpu}] "
                    else:
                        gpu_tag = "[CPU] "
                    cmd = [
                        sys.executable, "-m", "cholimex", "collection", "--input", str(stereo),
                        "--mixture", str(mixture), "--output-dir", str(output),
                    ]
                    print(f"[{idx}/{len(pairs)}] {gpu_tag}{stereo.name} + {mixture.name} -> {output}")
                    timing = _timed_subprocess(cmd, stereo, output, worker_env)
                    if timing["exit_code"] != 0:
                        print(f"Error refining {stereo.name} (exit code {timing['exit_code']})")
                    return timing
                finally:
                    gpu_queue.put(target_gpu)

            with ThreadPoolExecutor(max_workers=max(1, total_workers)) as pool:
                timing_items = list(pool.map(_process_cholimex, enumerate(pairs, start=1)))


    if not args.dry_run:
        if args.step == "split_dialogue":
            split_report = _summarize_split_dialogue_manifests(output_dir, input_audio_count=len(wav_files))
            split_report_path = _write_split_dialogue_report(output_dir, split_report)
            _print_split_dialogue_summary(split_report, split_report_path)
        phase = _phase_report(args.step, args, phase_started_at, perf_counter() - phase_started, timing_items, resource_plan)
        report_path = _write_timing_report(workflow_id, phase)
        _print_timing_summary(phase, report_path)
    print("\nBatch execution complete.")


if __name__ == "__main__":
    run_batch()
