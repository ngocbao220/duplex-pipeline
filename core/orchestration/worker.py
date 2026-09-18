"""One isolated process per pipeline, receiving mixtures only (never reference audio)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))
from core.orchestration.contract import run_sample, write_json
from core.orchestration.logging_style import get_logger


def cholimex(source, output, config):
    from core.config import Config, PATH_FIELDS
    from cholimex.runner import run_cholimex_file
    from cholimex.devices import resolve_device
    config = dict(config)
    debug = bool(config.pop('debug', False))
    cfg = Config(**{key: Path(value) if key in PATH_FIELDS and value is not None else value
                    for key, value in config.items()})
    result = run_cholimex_file(source, output, cfg)
    if not debug:
        import shutil
        shutil.rmtree(output / 'debug', ignore_errors=True)
    return output / 'audio.stereo.wav', {
        'device': resolve_device(cfg.runtime_device, cfg.allow_cpu_fallback), 'pipeline_result': result}


def duplexchat(source, output, config):
    from duplexchat.runner import run_single_audio
    from core.orchestration.contract import write_json
    config = dict(config)
    debug = bool(config.pop('debug', False))
    write_json(output / 'config.json', config)
    result = run_single_audio(str(source), output_dir=str(output), debug=debug, **config)
    return None, {
        'stereo_files': [str(path) for path in result['stereo_files']],
        'devices': result['devices'],
        'phase_times': result.get('phase_times', {}),
        'audio_duration_sec': result.get('audio_duration_sec', 0.0),
    }


def sommelier(source, output, config):
    from sommelier.runner import run
    return run(source, output, config)


ADAPTERS = {'cholimex': cholimex, 'duplexchat': duplexchat, 'sommelier': sommelier}


def run_batch(request: dict, adapter=None) -> list[dict]:
    name = request['pipeline']
    adapter = adapter or ADAPTERS[name]
    results = []
    logger = get_logger(name)
    logger.info("Step 1: Running pipeline (%d samples)", len(request['samples']))
    for index, sample in enumerate(request['samples'], 1):
        logger.info("Processing sample %d/%d: %s", index, len(request['samples']), sample['key'])
        result = run_sample(name, sample, Path(request['pred_root']) / sample['key'],
                            request['config'], request['code'], adapter, force=request['force'])
        results.append(result)
        write_json(Path(request['results']), results)
        if result["status"] == "complete":
            logger.info("Complete sample %s%s", sample['key'], " (resumed)" if result.get('resumed') else "")
        else:
            logger.error("Failed sample %s; details: %s", sample['key'], Path(request['pred_root']) / sample['key'] / 'run.json')
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request', type=Path)
    group.add_argument('--check-imports', choices=list(ADAPTERS))
    args = parser.parse_args()
    if args.check_imports:
        __import__(args.check_imports)
        get_logger(args.check_imports).info("Imports OK")
        return 0
    request = json.loads(args.request.read_text())
    get_logger(request['pipeline']).info("Pipeline=%s interpreter=%s", request['pipeline'], sys.executable)
    rows = run_batch(request)
    return int(any(row['status'] != 'complete' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
