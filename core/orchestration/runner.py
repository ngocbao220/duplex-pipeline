from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .contract import write_json
from .process import stream_process

ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ('cholimex', 'duplexchat', 'vilier')


def code_identity(name):
    digest = hashlib.sha256()
    locations = [ROOT / 'core', ROOT / 'pipeline' / name]
    if name != 'vilier':
        locations.append(ROOT / 'core')
    for location in locations:
        for directory, subdirs, files in os.walk(location):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith('.') and d not in {'__pycache__', 'outputs', 'logs', 'inputs'})
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix in {'.py', '.toml', '.lock', '.yaml'}:
                    digest.update(str(path.relative_to(ROOT)).encode())
                    digest.update(path.read_bytes())
    return digest.hexdigest()


def pipeline_config(name, args, cfg):
    if name == 'cholimex':
        config = json.loads(json.dumps(asdict(cfg), default=str))
        config['debug'] = bool(args.debug)
        return config
    path = args.vilier_config if name == 'vilier' else ROOT / 'configs/sommelier.json' if name == 'sommelier' else args.duplexchat_config
    config = json.loads(path.read_text())
    if name == 'vilier':
        config.setdefault('asr', {})['enabled'] = False
        config.setdefault('state_labeling', {})['enabled'] = False
        config.setdefault('runtime', {})['dry_run'] = False
        config.setdefault('entrypoint', {})['sample_rate'] = args.sample_rate
    config['debug'] = bool(args.debug)
    if name == 'duplexchat':
        config['separate_chunk'] = float(getattr(args, 'separate_chunk', config.get('separate_chunk', 120.0)))
        config['device_ids'] = getattr(args, 'device_ids', None)
        config['filter_music'] = bool(getattr(args, 'filter_music', False))
        config['music_model'] = str(getattr(args, 'music_model', 'htdemucs'))
        if getattr(args, 'separation_model', None):
            config['separation_model'] = str(args.separation_model)
        if getattr(args, 'diarization_backend', None):
            config['diarization_backend'] = str(args.diarization_backend)
        if getattr(args, 'diarization_model', None):
            config['diarization_model'] = str(args.diarization_model)
    return config


def launch_pipeline(name, request, run_dir):
    request_path = run_dir / name / 'request.json'
    write_json(request_path, request)
    project = ROOT / 'pipeline' / name
    env = dict(os.environ)
    # Never let an activated parent venv or uv project override collapse the isolated environments.
    env.pop('VIRTUAL_ENV', None)
    env['UV_PROJECT_ENVIRONMENT'] = str(project / '.venv')
    env['UV_CACHE_DIR'] = str(ROOT / '.uv-cache')
    env.pop('PYTHONPATH', None)
    command = ['uv', 'run', '--project', str(project), '--no-dev',
               'python', str(ROOT / 'core/orchestration/worker.py'), '--request', str(request_path)]
    try:
        return stream_process(command, project, run_dir / name / 'worker.log', env)
    except OSError as exc:
        print(f'[{name}] Cannot start worker: {exc}', flush=True)
        write_json(run_dir / name / 'launch_error.json', {'error': str(exc)})
        return 1



