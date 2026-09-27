"""Batch ASR for numbered final stereo files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.asr.pipeline import ASRPipeline, input_fingerprints
from core.orchestration.contract import write_json


def _offline_preflight(cfg: dict, pipeline_name: str) -> None:
    zip_cfg = cfg["zipformer"]
    root = Path(zip_cfg["local_dir"])
    needed = [root / zip_cfg[name] for name in ("encoder", "decoder", "joiner", "bpe_model")]
    aligner = Path(cfg["aligner"]["local_dir"])
    speechbrain = Path(os.environ.get("SPEECHBRAIN_MODEL_PATH", ""))
    silero = Path(os.environ.get("SILERO_VAD_MODEL_PATH", ""))
    missing = [str(path) for path in needed if not path.is_file()]
    if not aligner.is_dir() or not (aligner / "config.json").is_file() or not any(
        (aligner / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        missing.append(str(aligner) + " (config.json and model weights)")
    if pipeline_name == "duplexchat" and (not speechbrain.is_dir() or not all(
        (speechbrain / name).is_file() for name in ("hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt", "mean_var_norm_emb.ckpt")
    )):
        missing.append(str(speechbrain) + " (SpeechBrain ECAPA bundle)")
    if not (silero.is_file() or (silero.is_dir() and any((silero / name).is_file() for name in (
        "silero_vad.jit", "silero_vad.pt", "files/silero_vad.jit", "hubconf.py"
    )))):
        missing.append(str(silero) + " (Silero VAD)")
    if missing:
        raise FileNotFoundError("ASR offline preflight failed: " + ", ".join(missing))


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("duplexchat", "sommelier"), required=True)
    parser.add_argument("--stereo-root", type=Path, required=True)
    parser.add_argument("--dialogue-root", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config_json)
    if args.offline:
        os.environ["MODE"] = "sever"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    stereo_files = sorted(path for path in args.stereo_root.rglob("stereo_*.wav")
                          if not {".runs", ".history", "debug"}.intersection(path.parts))
    if not stereo_files:
        print(f"ASR: no numbered stereo files under {args.stereo_root}", file=sys.stderr)
        return 1
    report_path = args.stereo_root / "asr_report.json"
    report = {"pipeline": args.pipeline, "total": len(stereo_files), "complete": 0, "failed": 0, "items": []}
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    pending = []
    for stereo in stereo_files:
        relative = stereo.relative_to(args.stereo_root)
        output = stereo.with_name(stereo.stem.replace("stereo_", "dialogue_") + ".asr.json")
        if output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                provenance = existing["provenance"]
                fingerprints = input_fingerprints(stereo, args.dialogue_root / relative.parent, args.pipeline)
                if provenance["config_sha256"] == signature and all(provenance.get(name) == value for name, value in fingerprints.items()):
                    report["complete"] += 1
                    report["items"].append({"stereo": str(relative), "status": "resumed", "output": str(output)})
                    print(f"ASR [RESUMED] {relative}", flush=True)
                    write_json(report_path, report)
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
        pending.append((stereo, relative, output))
    if not pending:
        print(f"ASR: {report['complete']}/{report['total']} complete -> {report_path}")
        return 0
    try:
        if args.offline:
            _offline_preflight(cfg, args.pipeline)
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        engine = ASRPipeline(cfg, args.offline, device, args.pipeline)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report.update(status="failed", failed=len(pending), error=error)
        report["items"].extend({"stereo": str(relative), "status": "failed", "error": error}
                               for _stereo, relative, _output in pending)
        write_json(report_path, report)
        print(f"ASR initialization failed: {error}", file=sys.stderr, flush=True)
        return 1
    for number, (stereo, relative, output) in enumerate(pending, start=1):
        dialogue_dir = args.dialogue_root / relative.parent
        print(f"ASR [{number}/{len(pending)} pending] {relative}", flush=True)
        try:
            if output.is_file():
                archived = output.parent / ".history" / uuid.uuid4().hex / output.name
                archived.parent.mkdir(parents=True, exist_ok=True)
                output.replace(archived)
            result = engine.process(stereo, dialogue_dir, args.pipeline)
            write_json(output, result)
            report["complete"] += 1
            report["items"].append({"stereo": str(relative), "status": "complete", "output": str(output)})
        except Exception as exc:
            report["failed"] += 1
            report["items"].append({"stereo": str(relative), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"ASR failed {relative}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        write_json(report_path, report)
    print(f"ASR: {report['complete']}/{report['total']} complete, {report['failed']} failed -> {report_path}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(run())
