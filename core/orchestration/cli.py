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
        collection.add_argument("--config", type=Path, default=None, help="Path to Cholimex config JSON or YAML")

    single = commands.add_parser("single", help="Run one supplied mixture audio file.")
    single.add_argument("--input", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, required=True)
    single.add_argument("--debug", action="store_true")
    if name == "sommelier":
        single.add_argument("--config", type=Path, default=None, help="Sommelier config YAML or JSON")
        single.add_argument("--overlap-threshold", type=float, default=None)
        single.add_argument("--speaker-link-threshold", type=float, default=None)
        single.add_argument("--max-chunk-duration", type=float, default=None)
        single.add_argument("--demucs", action="store_true", default=None)
        single.add_argument("--expected-speakers", type=int, default=None)

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
        split.add_argument("--diarize-chunk", type=str, default=None, help="Max diarization chunk in seconds, or 'full' to process entire audio without chunking")
        split.add_argument("--device-ids", type=int, nargs="+")
        split.add_argument("--filter-music", action="store_true", default=True)
        split.add_argument("--no-filter-music", dest="filter_music", action="store_false")
        split.add_argument("--music-model", type=str, default="htdemucs")
        split.add_argument("--lid", type=str, default=None, help="Language Identification filter code (e.g. 'vi' for Vietnamese)")
        split.add_argument("--lid-model", type=str, default="openai/whisper-small", help="Whisper LID model path or identifier")
        split.add_argument("--min-vi-prob", "--min-lid-prob", dest="min_vi_prob", type=float, default=0.5, help="Minimum language probability threshold")
        split.add_argument("--diarization-backend", type=str, default=None)
        split.add_argument("--diarization-model", type=str, default=None)
        split.add_argument("--dialogue-gap-seconds", type=float, default=5.0)
        split.add_argument("--min-dialogue-duration-seconds", type=float, default=10.0)
        split.add_argument("--max-dialogue-duration-seconds", type=float, default=600.0)
        split.add_argument("--max-single-speaker-ratio", type=float, default=0.8)
        split.add_argument("--preferred-split-pause-seconds", type=float, default=3.0)
        split.add_argument("--min-split-pause-seconds", type=float, default=1.5)
        split.add_argument("--speaker-link-threshold", type=float, default=0.75, help="Cosine similarity threshold for linking speaker embeddings across chunks")
        split.add_argument("--expected-speakers", type=int, default=2, help="Expected number of speakers in the conversation")

        sep = commands.add_parser("separate_dialogue", help="Run DialogueSidon separation on extracted dialogue files.")
        sep.add_argument("--input", type=Path, required=True)
        sep.add_argument("--output-dir", type=Path, required=True)
        sep.add_argument("--device-ids", type=int, nargs="+")
        sep.add_argument("--separation-chunk", "--separate-chunk", dest="separate_chunk", type=float, default=120.0)
        sep.add_argument("--separation-model", type=Path, default=None, help="Verified local DialogueSidon model directory")
        sep.add_argument("--num-steps", type=int, default=30)
        sep.add_argument("--progress-manifest", type=Path, default=None,
                         help="Collection-level manifest updated as each dialogue is separated")
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

        config_path = getattr(args, "config", None)
        if config_path is None:
            if (ROOT / "configs/config.json").exists():
                config_path = ROOT / "configs/config.json"
            elif (ROOT / "configs/pipeline/cholimex.yaml").exists():
                config_path = ROOT / "configs/pipeline/cholimex.yaml"

        cholimex_cfg = load_config(config_path) if config_path else load_config()

        output_dir = args.output_dir.resolve()
        stereo = refine_stereo_file(
            args.input.resolve(),
            output_dir,
            cholimex_cfg,
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
            dialogue_gap_seconds=args.dialogue_gap_seconds,
            min_dialogue_duration_seconds=args.min_dialogue_duration_seconds,
            max_dialogue_duration_seconds=args.max_dialogue_duration_seconds,
            max_single_speaker_ratio=args.max_single_speaker_ratio,
            preferred_split_pause_seconds=args.preferred_split_pause_seconds,
            min_split_pause_seconds=args.min_split_pause_seconds,
            speaker_link_threshold=getattr(args, "speaker_link_threshold", 0.75),
            expected_speakers=getattr(args, "expected_speakers", 2),
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
            progress_manifest=getattr(args, "progress_manifest", None),
        )
        print(f"Done separate_dialogue\n-> Output: {args.output_dir}\n-> Stereo files: {len(result['stereo_files'])}", flush=True)
        return 1 if result.get("failures") else 0
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
        extra = {}
        for k in ("config", "overlap_threshold", "speaker_link_threshold", "max_chunk_duration", "demucs", "expected_speakers"):
            if hasattr(args, k):
                extra[k] = getattr(args, k)
        return run_single(name, args.input.resolve(), args.output_dir.resolve(), args.debug,
                           getattr(args, "separate_chunk", 120.0), getattr(args, "device_ids", None),
                           getattr(args, "filter_music", False), getattr(args, "music_model", "htdemucs"),
                           getattr(args, "diarization_backend", None), getattr(args, "diarization_model", None),
                           str(args.separation_model.resolve()) if getattr(args, "separation_model", None) else None,
                           **extra)

    raise AssertionError(f"Unsupported command: {args.command}")
