"""CLI primitives shared by the three independently installed pipelines."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def build_pipeline_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {name} speaker-separation pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    if name == "cholimex":
        collection = commands.add_parser("collection", help="Refine one DuplexChat stereo conversation.")
        collection.add_argument("--input", type=Path, required=True, help="Numbered DuplexChat stereo WAV")
        collection.add_argument("--mixture", type=Path, required=True, help="Original mono mixture WAV")
        collection.add_argument("--output-dir", type=Path, required=True)

    single = commands.add_parser("single", help="Run one supplied mixture audio file.")
    single.add_argument("--input", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, required=True)
    single.add_argument("--debug", action="store_true")
    if name == "duplexchat":
        single.add_argument("--separation-chunk", "--separate-chunk", dest="separate_chunk", type=float, default=120.0)
        single.add_argument("--separation-model", type=Path, default=None, help="Verified local DialogueSidon model directory")
        single.add_argument("--device-ids", type=int, nargs="+")
        single.add_argument("--filter-music", action="store_true", help="Remove background music with Demucs before dialogue separation")
        single.add_argument("--music-model", type=str, default="htdemucs", help="Demucs music separation model (default: htdemucs)")
        single.add_argument("--diarization-backend", type=str, default=None, help="Diarization backend (pyannote, sortformer, diarizen, auto)")
        single.add_argument("--diarization-model", type=str, default=None, help="Diarization model name or HF repo ID")

        split = commands.add_parser("split_valid_dialogue", help="Preprocess, diarize, extract valid dialogues and filter music.")
        split.add_argument("--input", type=Path, required=True)
        split.add_argument("--output-dir", type=Path, required=True)
        split.add_argument("--debug", action="store_true")
        split.add_argument("--diarize-chunk", type=float, default=None)
        split.add_argument("--device-ids", type=int, nargs="+")
        split.add_argument("--filter-music", action="store_true", default=True)
        split.add_argument("--no-filter-music", dest="filter_music", action="store_false")
        split.add_argument("--music-model", type=str, default="htdemucs")
        split.add_argument("--lid", type=str, default=None, help="Language Identification filter code (e.g. 'vi' for Vietnamese)")
        split.add_argument("--lid-model", type=str, default="openai/whisper-small", help="Whisper LID model path or identifier")
        split.add_argument("--min-vi-prob", "--min-lid-prob", dest="min_vi_prob", type=float, default=0.5, help="Minimum language probability threshold")
        split.add_argument("--diarization-backend", type=str, default=None)
        split.add_argument("--diarization-model", type=str, default=None)

        sep = commands.add_parser("separate_dialogue", help="Run DialogueSidon separation on extracted dialogue files.")
        sep.add_argument("--input", type=Path, required=True)
        sep.add_argument("--output-dir", type=Path, required=True)
        sep.add_argument("--device-ids", type=int, nargs="+")
        sep.add_argument("--separation-chunk", "--separate-chunk", dest="separate_chunk", type=float, default=120.0)
        sep.add_argument("--separation-model", type=Path, default=None, help="Verified local DialogueSidon model directory")
        sep.add_argument("--num-steps", type=int, default=30)
        sep.add_argument("--debug", action="store_true")
        validate = commands.add_parser("validate_model", help="Validate a local DialogueSidon bundle without loading it on a GPU.")
        validate.add_argument("--separation-model", type=Path, required=True, help="Local DialogueSidon model directory")
    return parser


def run_pipeline_command(name: str, argv: list[str] | None = None) -> int:
    args = build_pipeline_parser(name).parse_args(argv)
    if args.command == "collection":
        if name != "cholimex":
            raise AssertionError(f"Unsupported collection command for {name}")
        if not args.input.is_file():
            raise SystemExit(f"DuplexChat stereo input does not exist: {args.input}")
        if not args.mixture.is_file():
            raise SystemExit(f"Original mixture input does not exist: {args.mixture}")
        if args.output_dir.exists():
            raise SystemExit(f"Cholimex collection output already exists: {args.output_dir}")
        from core.config import load_config
        from cholimex.collection import refine_stereo_file

        output_dir = args.output_dir.resolve()
        stereo = refine_stereo_file(
            args.input.resolve(), output_dir, load_config(ROOT / "configs/config.json"),
            mixture_path=args.mixture.resolve(),
        )
        print(f"Done\n-> VAD: {output_dir / 'vad_left.txt'}, {output_dir / 'vad_right.txt'}\n-> stereo: {stereo}", flush=True)
        return 0
    if args.command == "split_valid_dialogue":
        if name != "duplexchat":
            raise AssertionError(f"Unsupported split_valid_dialogue command for {name}")
        if not args.input.is_file():
            raise SystemExit(f"Input audio file does not exist: {args.input}")
        from duplexchat.runner import split_valid_dialogues
        lid_arg = getattr(args, "lid", None)
        filter_vietnamese = (lid_arg is not None and str(lid_arg).strip().lower() in ("vi", "vietnamese"))
        result = split_valid_dialogues(
            str(args.input.resolve()),
            str(args.output_dir.resolve()),
            diarize_chunk=getattr(args, "diarize_chunk", None),
            diarization_backend=getattr(args, "diarization_backend", "auto") or "auto",
            diarization_model=getattr(args, "diarization_model", "pyannote/speaker-diarization-community-1") or "pyannote/speaker-diarization-community-1",
            device_ids=getattr(args, "device_ids", None),
            filter_music=getattr(args, "filter_music", True),
            music_model=getattr(args, "music_model", "htdemucs"),
            filter_vietnamese=filter_vietnamese,
            lid_model=getattr(args, "lid_model", "openai/whisper-small"),
            min_vi_prob=getattr(args, "min_vi_prob", 0.5),
            debug=getattr(args, "debug", False),
        )
        print(f"Done split_valid_dialogue\n-> Output: {args.output_dir}\n-> Dialogue count: {result['dialogue_count']}", flush=True)
        return 0
    if args.command == "separate_dialogue":
        if name != "duplexchat":
            raise AssertionError(f"Unsupported separate_dialogue command for {name}")
        from duplexchat.runner import separate_dialogue_files
        result = separate_dialogue_files(
            str(args.input.resolve()),
            str(args.output_dir.resolve()),
            device_ids=getattr(args, "device_ids", None),
            num_steps=getattr(args, "num_steps", 30),
            separate_chunk=getattr(args, "separate_chunk", 120.0),
            separation_model=str(args.separation_model.resolve()) if args.separation_model else None,
            debug=getattr(args, "debug", False),
        )
        print(f"Done separate_dialogue\n-> Output: {args.output_dir}\n-> Stereo files: {len(result['stereo_files'])}", flush=True)
        return 0
    if args.command == "validate_model":
        if name != "duplexchat":
            raise AssertionError(f"Unsupported validate_model command for {name}")
        from core.local_model_validation import validate_dialoguesidon_model

        paths = validate_dialoguesidon_model(args.separation_model.resolve())
        print(f"DialogueSidon local model valid\n-> Model: {args.separation_model.resolve()}\n-> Artifacts: {len(paths)}", flush=True)
        return 0
    if args.command == "single":
        if not args.input.is_file():
            raise SystemExit(f"Input audio file does not exist: {args.input}")
        from .single import run_single
        return run_single(name, args.input.resolve(), args.output_dir.resolve(), args.debug,
                           getattr(args, "separate_chunk", 120.0), getattr(args, "device_ids", None),
                           getattr(args, "filter_music", False), getattr(args, "music_model", "htdemucs"),
                           getattr(args, "diarization_backend", None), getattr(args, "diarization_model", None),
                           str(args.separation_model.resolve()) if getattr(args, "separation_model", None) else None)

    raise AssertionError(f"Unsupported command: {args.command}")
