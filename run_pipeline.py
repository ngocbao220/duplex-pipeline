#!/usr/bin/env python3
"""Unified Hydra Entry Point for Duplex-Pipelines (DuplexChat, Sommelier, Stereo Benchmark).

Supports hierarchical YAML configuration, flexible command-line parameter overrides,
and colored logging strictly aligned with logging.md.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.orchestration.logging_style import get_logger, section, StepTimer

logger = get_logger("run_pipeline")


def _setup_runtime_environment(cfg: DictConfig) -> dict[str, str]:
    """Configure environment variables for offline mode, GPU allocation, and thread limits."""
    env = dict(os.environ)

    # Mode: Offline (Server cluster) vs Online (Dev)
    is_offline = bool(cfg.env.get("offline", False))
    if is_offline:
        logger.info("[MODE: SEVER] Offline mode — using local disk models only.")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    else:
        logger.info("[MODE: DEV] Dev mode — remote model downloads from HuggingFace permitted.")
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
        env.pop("HF_DATASETS_OFFLINE", None)

    # GPU target
    gpu_id = str(cfg.gpu).strip()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Pipeline Run ID for timing aggregation
    if "PIPELINE_RUN_ID" not in env:
        env["PIPELINE_RUN_ID"] = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    # Enforce strictly 1 CPU thread per process
    try:
        from core.runtime_cpu import enforce_single_cpu_thread
        enforce_single_cpu_thread()
    except ImportError:
        pass

    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    # PYTHONPATH
    duplex_src = str(ROOT_DIR / "pipeline" / "duplexchat" / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    paths = [str(ROOT_DIR), duplex_src]
    if existing_pythonpath:
        paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    # Specific model paths
    if cfg.env.paths.get("sortformer"):
        env["SORTFORMER_MODEL_PATH"] = str(cfg.env.paths.sortformer)
    if cfg.env.paths.get("dialoguesidon"):
        env["DIALOGUESIDON_MODEL_PATH"] = str(cfg.env.paths.dialoguesidon)
    if cfg.env.paths.get("speechbrain"):
        env["SPEECHBRAIN_MODEL_PATH"] = str(cfg.env.paths.speechbrain)
    if cfg.env.paths.get("whisper"):
        env["WHISPER_MODEL_PATH"] = str(cfg.env.paths.whisper)
    if cfg.env.paths.get("sepreformer"):
        env["SEPREFORMER"] = str(cfg.env.paths.sepreformer)

    return env


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool = False) -> int:
    """Execute command or print in dry-run mode."""
    cmd_str = " ".join(cmd)
    if dry_run:
        logger.info("[DRY-RUN] Would execute:\n  %s", cmd_str)
        return 0
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)
    return res.returncode


def step_convert(cfg: DictConfig, env: dict[str, str]) -> int:
    """Step 0: Convert crawled audio files to 16kHz mono WAV."""
    crawl_dir = Path(cfg.data.crawl_dir)
    raw_dir = Path(cfg.data.raw_dir)
    has_crawl = crawl_dir.exists() and any(crawl_dir.iterdir()) if crawl_dir.exists() else False
    if not has_crawl and not cfg.dry_run:
        logger.info("No crawled files found in %s. Skipping convert step.", crawl_dir)
        return 0

    print(section("Convert Audio to WAV"))
    convert_workers = cfg.get("optimization", {}).get("convert_workers", 8)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "convert_crawl_to_raw.py"),
        "--crawl-dir", str(crawl_dir),
        "--raw-dir", str(raw_dir.parent),
        "--workers", str(convert_workers),
    ]
    return _run_cmd(cmd, env, dry_run=cfg.dry_run)


def step_split_dialogue(cfg: DictConfig, env: dict[str, str]) -> int:
    """Phase 1: Preprocess, Diarize, Dialogue Filter, Music Removal & LID."""
    print(section("Dialogue Filtering"))
    raw_dir = _resolve_raw_data_dir(cfg)
    dialogue_dir = Path(cfg.data.dialogue_dir)

    if not raw_dir.exists() and not cfg.dry_run:
        logger.info("Raw directory %s does not exist. Skipping dialogue split step.", raw_dir)
        return 0

    workers = cfg.get("optimization", {}).get("workers", 2)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_pipeline_batch.py"),
        "--step", "split_dialogue",
        "--input-dir", str(raw_dir),
        "--output-dir", str(dialogue_dir),
        "--gpu", str(cfg.gpu),
        "--pattern", str(cfg.data.get("pattern", "*.wav")),
        "--workers", str(workers),
    ]

    split_cfg = cfg.get("dialogue_split") or cfg.get("pipeline", {})

    # Diarization
    diar_cfg = split_cfg.get("diarization") or cfg.get("diarization", {})
    if diar_cfg:
        if diar_cfg.get("backend"):
            cmd.extend(["--diarization-backend", str(diar_cfg.backend)])
        if diar_cfg.get("model"):
            cmd.extend(["--diarization-model", str(diar_cfg.model)])
        max_chunk = diar_cfg.get("max_chunk_seconds") or diar_cfg.get("chunk_duration")
        if max_chunk is not None:
            cmd.extend(["--diarize-chunk", str(max_chunk)])


    # Music filtering
    music_filter_enabled = split_cfg.get("music_filter", {}).get("enabled", True)
    if music_filter_enabled:
        cmd.append("--filter-music")
    else:
        cmd.append("--no-filter-music")

    # Language Identification (LID)
    lid_cfg = split_cfg.get("lid", {})
    if lid_cfg.get("enabled", False) and lid_cfg.get("code"):
        cmd.extend(["--lid", str(lid_cfg.code)])

    if cfg.dry_run:
        cmd.append("--dry-run")

    return _run_cmd(cmd, env, dry_run=cfg.dry_run)


def step_separate_dialogue(cfg: DictConfig, env: dict[str, str]) -> int:
    """Phase 2 (DuplexChat): Separate dialogue channels with DialogueSidon."""
    print(section("DuplexChat"))
    dialogue_dir = Path(cfg.data.dialogue_dir)
    duplex_out_dir = Path(cfg.data.duplex_out_dir)

    if not dialogue_dir.exists() and not cfg.dry_run:
        logger.info("Dialogue directory %s does not exist. Skipping dialogue separation step.", dialogue_dir)
        return 0

    sep_chunk = cfg.pipeline.get("separation", {}).get("chunk", 60.0)
    workers = cfg.get("optimization", {}).get("workers", 2)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_pipeline_batch.py"),
        "--step", "separate_dialogue",
        "--input-dir", str(dialogue_dir),
        "--output-dir", str(duplex_out_dir),
        "--gpu", str(cfg.gpu),
        "--separation-chunk", str(sep_chunk),
        "--workers", str(workers),
    ]

    sep_model = cfg.pipeline.get("separation", {}).get("model")
    if sep_model:
        model_path = Path(sep_model)
        if model_path.exists():
            cmd.extend(["--separation-model", str(model_path.resolve())])

    if cfg.dry_run:
        cmd.append("--dry-run")

    return _run_cmd(cmd, env, dry_run=cfg.dry_run)


def step_sommelier(cfg: DictConfig, env: dict[str, str]) -> int:
    """Phase 2 (Sommelier): Overlap detection, SepReformer separation & track reconstruction."""
    print(section("Sommelier"))
    input_dir = _resolve_raw_data_dir(cfg)
    sommelier_out_dir = Path(cfg.data.sommelier_out_dir)

    if not input_dir.exists() and not cfg.dry_run:
        logger.info("Raw input directory %s does not exist. Skipping Sommelier step.", input_dir)
        return 0

    workers = cfg.get("optimization", {}).get("workers", 2)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_pipeline_batch.py"),
        "--step", "sommelier",
        "--input-dir", str(input_dir),
        "--output-dir", str(sommelier_out_dir),
        "--gpu", str(cfg.gpu),
        "--workers", str(workers),
    ]

    if cfg.dry_run:
        cmd.append("--dry-run")

    return _run_cmd(cmd, env, dry_run=cfg.dry_run)


def step_benchmark(cfg: DictConfig, env: dict[str, str]) -> int:
    """Phase 3: Stereo Benchmark & Data Retention Reporting."""
    print(section("Stereo Benchmark"))
    pipeline_name = str(cfg.pipeline.name)
    corpus_dir = Path(cfg.data.duplex_out_dir if pipeline_name == "duplexchat" else cfg.data.sommelier_out_dir)
    bench_out_dir = Path(cfg.benchmark.output_dir)
    dnsmos_dir = Path(cfg.benchmark.dnsmos_dir)

    if not corpus_dir.exists() and not cfg.dry_run:
        logger.info("Corpus directory %s does not exist. Skipping benchmark step.", corpus_dir)
        return 0

    workers = cfg.get("optimization", {}).get("workers", 2)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "benchmark_stereo.py"),
        "--corpus", str(corpus_dir),
        "--dnsmos-model-dir", str(dnsmos_dir),
        "--output-dir", str(bench_out_dir),
        "--workers", str(workers),
    ]
    if cfg.benchmark.get("check_models", True):
        cmd.append("--check-models")
    if cfg.benchmark.get("device"):
        cmd.extend(["--device", str(cfg.benchmark.device)])
    if cfg.benchmark.get("debug", False):
        cmd.append("--debug")

    ret = _run_cmd(cmd, env, dry_run=cfg.dry_run)
    if ret != 0:
        return ret

    # Retention Report
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.env.paths.base_data) / "processed"
    retention_md = Path(cfg.benchmark.retention_report_md)

    if (not raw_dir.exists() or not processed_dir.exists()) and not cfg.dry_run:
        logger.info("Raw or processed directory does not exist. Skipping retention report.")
        return 0

    retention_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "report_data_retention.py"),
        "--raw-dir", str(raw_dir),
        "--processed-dir", str(processed_dir),
        "--output-md", str(retention_md),
    ]
    return _run_cmd(retention_cmd, env, dry_run=cfg.dry_run)


def _resolve_raw_data_dir(cfg: DictConfig) -> Path:
    """Find the raw dataset directory matching data.source with Kaggle and local fallbacks."""
    raw_path = Path(cfg.data.raw_dir)
    if raw_path.exists():
        return raw_path

    source = str(cfg.data.source)
    alt_sources = [source, source.replace("_", "-"), source.replace("-", "_")]
    if "_" in source:
        alt_sources.append(source.split("_")[0])

    base_data = Path(str(cfg.env.paths.get("base_data", "data")))
    candidates: list[Path] = []
    for s in alt_sources:
        candidates.append(base_data / s)
    for s in alt_sources:
        candidates.append(base_data / "raw" / s)
    if raw_path.parent.exists():
        for s in alt_sources:
            candidates.append(raw_path.parent / s)

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            logger.info("Auto-resolved raw dataset directory: %s -> %s", raw_path, cand)
            cfg.data.raw_dir = str(cand)
            return cand

    # Local fallback when unmounted on dev machine
    if (str(base_data).startswith("/kaggle") or str(base_data).startswith("/storage-voice")) and not base_data.exists():
        local_cand = ROOT_DIR / "data" / "raw" / source
        if not local_cand.exists():
            local_cand = ROOT_DIR / "data" / source
        cfg.data.raw_dir = str(local_cand)
        return local_cand

    return raw_path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Auto-fallback: if configured base_data or base_models points to unmounted cluster/kaggle dirs,
    # safely redirect to local workspace data & models.
    if cfg.get("env") and cfg.env.get("paths"):
        base_data = str(cfg.env.paths.get("base_data", ""))
        if (base_data.startswith("/storage-voice") or base_data.startswith("/kaggle")) and not Path(base_data).exists():
            local_base_data = ROOT_DIR / "data"
            logger.warning(
                "Data directory %s not mounted. Falling back to local workspace data: %s",
                base_data,
                local_base_data,
            )
            cfg.env.paths.base_data = str(local_base_data)

        base_models = str(cfg.env.paths.get("base_models", ""))
        if (base_models.startswith("/storage-voice") or base_models.startswith("/kaggle")) and not Path(base_models).exists():
            local_base_models = ROOT_DIR / "models"
            logger.warning(
                "Models directory %s not mounted. Falling back to local workspace models: %s",
                base_models,
                local_base_models,
            )
            cfg.env.paths.base_models = str(local_base_models)

    # Resolve dataset paths for current execution
    _resolve_raw_data_dir(cfg)

    step = str(cfg.step).lower()
    pipeline_name = str(cfg.pipeline.name).lower()
    env = _setup_runtime_environment(cfg)

    logger.info("Hydra Config Initialized | Pipeline: [%s] | Step: [%s] | GPU: [%s]", pipeline_name, step, cfg.gpu)

    if step == "convert":
        sys.exit(step_convert(cfg, env))

    elif step == "split_dialogue":
        sys.exit(step_split_dialogue(cfg, env))

    elif step == "separate_dialogue":
        sys.exit(step_separate_dialogue(cfg, env))

    elif step == "sommelier":
        sys.exit(step_sommelier(cfg, env))

    elif step == "benchmark":
        sys.exit(step_benchmark(cfg, env))

    elif step == "all":
        # Step 0: Convert
        if step_convert(cfg, env) != 0:
            logger.error("Convert step failed.")
            sys.exit(1)

        if pipeline_name == "duplexchat":
            # Phase 1: Dialogue Filtering
            if step_split_dialogue(cfg, env) != 0:
                logger.error("Dialogue filtering step failed.")
                sys.exit(1)
            # Phase 2: Separate Dialogue
            if step_separate_dialogue(cfg, env) != 0:
                logger.error("DuplexChat separation step failed.")
                sys.exit(1)
            # Phase 3: Benchmark
            if step_benchmark(cfg, env) != 0:
                logger.error("Benchmark step failed.")
                sys.exit(1)

        elif pipeline_name == "sommelier":
            # Phase 2: Sommelier full pipeline
            if step_sommelier(cfg, env) != 0:
                logger.error("Sommelier step failed.")
                sys.exit(1)
            # Phase 3: Benchmark
            if step_benchmark(cfg, env) != 0:
                logger.error("Benchmark step failed.")
                sys.exit(1)

        logger.info("All pipeline phases completed successfully.")

    else:
        logger.error("Unknown step '%s'. Allowed: all | convert | split_dialogue | separate_dialogue | sommelier | benchmark", step)
        sys.exit(1)


if __name__ == "__main__":
    main()
