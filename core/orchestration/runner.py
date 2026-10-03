from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .contract import write_json
from .process import stream_process

ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ('cholimex', 'duplexchat', 'sommelier')


def code_identity(name):
    digest = hashlib.sha256()
    locations = [ROOT / 'core', ROOT / 'pipeline' / name]
    for location in locations:
        for directory, subdirs, files in os.walk(location):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith('.') and d not in {'__pycache__', 'outputs', 'logs', 'inputs'})
            for filename in sorted(files):
                path = Path(directory) / filename
                if path.suffix in {'.py', '.toml', '.lock', '.yaml'}:
                    digest.update(str(path.relative_to(ROOT)).encode())
                    digest.update(path.read_bytes())
    return digest.hexdigest()


def pipeline_config(name, args, cfg):
    if name == 'cholimex':
        config = json.loads(json.dumps(asdict(cfg), default=str))
        config['debug'] = bool(args.debug)
        return config
    if name == 'sommelier':
        import yaml

        config_path = getattr(args, 'config', None)
        if config_path and Path(config_path).is_file():
            config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        elif (ROOT / 'configs/pipeline/sommelier.yaml').exists():
            config = yaml.safe_load((ROOT / 'configs/pipeline/sommelier.yaml').read_text(encoding='utf-8'))
        else:
            config = {}
        for key in ('overlap_threshold', 'speaker_link_threshold', 'max_chunk_duration', 'demucs', 'expected_speakers'):
            val = getattr(args, key, None)
            if val is not None:
                config[key] = val
    else:
        config = json.loads(args.duplexchat_config.read_text())
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


import sys


def launch_pipeline(name, request, run_dir):
    request_path = run_dir / name / 'request.json'
    write_json(request_path, request)
    project = ROOT / 'pipeline' / name

    env = dict(os.environ)
    extra_paths = [str(ROOT), str(ROOT / 'pipeline' / name / 'src')]
    existing_pythonpath = env.get('PYTHONPATH', '')
    if existing_pythonpath:
        env['PYTHONPATH'] = os.pathsep.join(extra_paths) + os.pathsep + existing_pythonpath
    else:
        env['PYTHONPATH'] = os.pathsep.join(extra_paths)
    command = [sys.executable, str(ROOT / 'core/orchestration/worker.py'), '--request', str(request_path)]

    try:
        return stream_process(command, project, run_dir / name / 'worker.log', env)
    except OSError as exc:
        print(f'[{name}] Cannot start worker: {exc}', flush=True)
        write_json(run_dir / name / 'launch_error.json', {'error': str(exc)})
        return 1
