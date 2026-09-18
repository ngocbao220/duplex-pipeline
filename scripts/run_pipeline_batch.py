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
from time import perf_counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.model_utils import resolve_local_model_path
from pipeline.duplexchat.src.duplexchat.model_options import DIARIZATION_MODELS, resolve_model_alias

TIMING_REPORT_PATH = Path("outputs/pipeline_timing.json")


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
        models_to_check.append(("SepReformer Checkpoint", os.environ.get("VILIER_SEPREFORMER_CHECKPOINT"), "VILIER_SEPREFORMER_CHECKPOINT", None, None))
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
        lid_status = f"Whisper LID filter for language '{args.lid}' (keep probability >= 0.5)" if args.lid else "Skipped (no --lid)"
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


def _phase_report(step: str, args, started_at: dt.datetime, elapsed: float, items: list[dict]) -> dict:
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
    parser.add_argument("--step", "-s", choices=["split_dialogue", "separate_dialogue", "sommelier"], required=True,
                        help="Pipeline step to execute.")
    parser.add_argument("--input-dir", "-i", type=Path, required=True,
                        help="Input directory containing audio WAV files or dialogue subfolders (e.g., /kaggle/input/... or data/raw/youtube).")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory to save processed results (e.g., /kaggle/working/data/processed/...).")
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
    parser.add_argument("--diarization-backend", type=str, default=None,
                        help="Diarization backend (e.g. pyannote, sortformer, diarizen, auto).")
    parser.add_argument("--diarization-model", type=str, default=None,
                        help="Diarization model ID or alias (e.g. nvidia/diar_streaming_sortformer_4spk-v2.1).")
    parser.add_argument("--separation-chunk", "--separate-chunk", dest="separation_chunk", type=float, default=120.0,
                        help="Chunk duration in seconds for dialogue separation (default: 120.0).")
    parser.add_argument("--separation-model", type=Path, default=None,
                        help="Verified local DialogueSidon model directory passed to every separation process.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print executable commands without running them.")
    args = parser.parse_args()
    workflow_id = os.environ.get("PIPELINE_RUN_ID") or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    phase_started_at = dt.datetime.now(dt.UTC)
    phase_started = perf_counter()
    timing_items: list[dict] = []

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log_preflight_info(args, input_dir, output_dir)

    env = dict(os.environ)
    gpu_str = str(args.gpu).strip()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    # Limit CPU thread pinning to avoid 100% CPU spikes across massive cores
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    # When CUDA_VISIBLE_DEVICES is set, CUDA devices are re-indexed starting from 0 inside the process
    num_gpus = len([g for g in gpu_str.split(",") if g.strip()])
    inner_device_ids = [str(i) for i in range(num_gpus)]

    if args.step == "split_dialogue":
        wav_files = sorted(list(input_dir.glob(args.pattern)))
        if not wav_files:
            print(f"No files matching '{args.pattern}' found in {input_dir}")
            return

        print(f"=== Running split_dialogue on {len(wav_files)} files from {input_dir} (GPU {args.gpu}) ===")
        for idx, wav in enumerate(wav_files, start=1):
            sub_out = output_dir / wav.stem
            filter_flag = "--filter-music" if args.filter_music else "--no-filter-music"
            cmd = [
                sys.executable, "-m", "duplexchat", "split_valid_dialogue",
                "--input", str(wav),
                "--output-dir", str(sub_out),
                "--device-ids", *inner_device_ids,
                filter_flag
            ]
            if args.lid:
                cmd.extend(["--lid", args.lid])
            if args.diarization_backend:
                cmd.extend(["--diarization-backend", args.diarization_backend])
            if args.diarization_model:
                cmd.extend(["--diarization-model", args.diarization_model])
            print(f"[{idx}/{len(wav_files)}] {wav.name} -> {sub_out.name}")
            if args.dry_run:
                print("  Command:", " ".join(cmd))
            else:
                timing = _timed_subprocess(cmd, wav, sub_out, env)
                timing_items.append(timing)
                if timing["exit_code"] != 0:
                    print(f"Error processing {wav.name} (exit code {timing['exit_code']})")

    elif args.step == "separate_dialogue":
        # Can take a directory containing dialogue subfolders or direct dialogue files
        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        if not subdirs:
            # Check if input_dir itself has dialogue_*.wav
            if list(input_dir.glob("dialogue_*.wav")):
                subdirs = [input_dir]
            else:
                print(f"No dialogue subdirectories or files found in {input_dir}")
                return

        print(f"=== Running separate_dialogue on {len(subdirs)} dialogue folders from {input_dir} (GPU {args.gpu}) ===")
        for idx, sdir in enumerate(subdirs, start=1):
            sub_out = output_dir / sdir.name
            cmd = [
                sys.executable, "-m", "duplexchat", "separate_dialogue",
                "--input", str(sdir),
                "--output-dir", str(sub_out),
                "--device-ids", *inner_device_ids,
                "--separation-chunk", str(args.separation_chunk)
            ]
            if args.separation_model:
                cmd.extend(["--separation-model", str(args.separation_model.resolve())])
            print(f"[{idx}/{len(subdirs)}] {sdir.name} -> {sub_out.name}")
            if args.dry_run:
                print("  Command:", " ".join(cmd))
            else:
                source_files = sorted(sdir.glob("dialogue_*.wav"))
                source = source_files[0] if len(source_files) == 1 else sdir
                timing = _timed_subprocess(cmd, source, sub_out, env)
                if len(source_files) > 1:
                    durations = [_audio_duration_seconds(path) for path in source_files]
                    audio_seconds = sum(value for value in durations if value is not None)
                    timing["audio_seconds"] = audio_seconds or None
                    timing["real_time_factor"] = timing["elapsed_seconds"] / audio_seconds if audio_seconds else None
                    timing["audio_seconds_per_wall_second"] = audio_seconds / timing["elapsed_seconds"] if audio_seconds and timing["elapsed_seconds"] else None
                timing_items.append(timing)
                if timing["exit_code"] != 0:
                    print(f"Error processing {sdir.name} (exit code {timing['exit_code']})")

    elif args.step == "sommelier":
        def _dialogue_key(p: Path) -> int:
            m = re.search(r"dialogue_(\d+)", p.stem)
            return int(m.group(1)) if m else 0

        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        if subdirs:
            print(f"=== Running Sommelier on {len(subdirs)} dialogue folders from {input_dir} (GPU {args.gpu}) ===")
            for idx, sdir in enumerate(subdirs, start=1):
                sub_out = output_dir / sdir.name
                sub_out.mkdir(parents=True, exist_ok=True)
                dialogue_files = sorted(list(sdir.glob("dialogue_*.wav")), key=_dialogue_key)
                if not dialogue_files:
                    dialogue_files = sorted(list(sdir.glob(args.pattern)))
                if not dialogue_files:
                    print(f"[{idx}/{len(subdirs)}] No audio clips found in {sdir.name}")
                    continue

                print(f"[{idx}/{len(subdirs)}] {sdir.name} ({len(dialogue_files)} dialogues) -> {sub_out.name}")
                for d_idx, dwav in enumerate(dialogue_files, start=1):
                    cmd = [
                        sys.executable, "-m", "sommelier", "single",
                        "--input", str(dwav),
                        "--output-dir", str(sub_out)
                    ]
                    print(f"  [{d_idx}/{len(dialogue_files)}] {dwav.name} -> {sub_out.name}")
                    if args.dry_run:
                        print("    Command:", " ".join(cmd))
                    else:
                        timing = _timed_subprocess(cmd, dwav, sub_out, env)
                        timing_items.append(timing)
                        if timing["exit_code"] != 0:
                            print(f"Error processing {dwav.name} (exit code {timing['exit_code']})")
        else:
            dialogue_files = sorted(list(input_dir.glob("dialogue_*.wav")), key=_dialogue_key)
            if not dialogue_files:
                dialogue_files = sorted(list(input_dir.glob(args.pattern)))
            if not dialogue_files:
                print(f"No files matching '{args.pattern}' found in {input_dir}")
                return

            print(f"=== Running Sommelier on {len(dialogue_files)} files from {input_dir} (GPU {args.gpu}) ===")
            for idx, wav in enumerate(dialogue_files, start=1):
                sub_out = output_dir / wav.stem
                sub_out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "-m", "sommelier", "single",
                    "--input", str(wav),
                    "--output-dir", str(sub_out)
                ]
                print(f"[{idx}/{len(dialogue_files)}] {wav.name} -> {sub_out.name}")
                if args.dry_run:
                    print("  Command:", " ".join(cmd))
                else:
                    timing = _timed_subprocess(cmd, wav, sub_out, env)
                    timing_items.append(timing)
                    if timing["exit_code"] != 0:
                        print(f"Error processing {wav.name} (exit code {timing['exit_code']})")


    if not args.dry_run:
        phase = _phase_report(args.step, args, phase_started_at, perf_counter() - phase_started, timing_items)
        report_path = _write_timing_report(workflow_id, phase)
        _print_timing_summary(phase, report_path)
    print("\nBatch execution complete.")


if __name__ == "__main__":
    run_batch()
