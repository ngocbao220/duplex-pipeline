from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
import uuid
from pathlib import Path

from .logging_style import get_logger


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(pipeline: str, source: Path, config: dict, code: str) -> str:
    payload = [pipeline, sha256(source), config, code]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


STEREO_FILENAME = "audio.stereo.wav"


def collection_sha256(manifest: Path) -> str:
    payload = bytearray(manifest.read_bytes())
    for row in json.loads(manifest.read_text())["conversations"]:
        payload.extend(sha256(Path(row["stereo"])).encode())
    return hashlib.sha256(payload).hexdigest()


def stereo_files_sha256(stereo_files: list[str | Path]) -> str:
    payload = bytearray()
    for index, stereo in enumerate(stereo_files, start=1):
        path = Path(stereo)
        payload.extend(f"{index}:{path.name}:".encode())
        payload.extend(sha256(path).encode())
    return hashlib.sha256(payload).hexdigest()


def validate_conversation_collection(manifest: Path) -> int:
    data = json.loads(manifest.read_text())
    rows = data["conversations"]
    if data.get("conversation_count") != len(rows):
        raise ValueError("conversation_count does not match manifest")
    for row in rows:
        stereo = Path(row["stereo"])
        info = __import__("soundfile").info(stereo)
        if info.channels != 2 or info.samplerate != 24_000 or info.frames <= 0:
            raise ValueError(f"Invalid 24 kHz stereo conversation: {stereo}")
        metadata = json.loads((stereo.parent / "metadata.json").read_text())
        if metadata.get("conversation_idx") != row.get("conversation_idx"):
            raise ValueError(f"Conversation metadata does not match manifest: {stereo}")
    return len(rows)


def validate_stereo(source: Path, stereo: Path) -> float:
    import numpy as np
    import soundfile as sf

    reference = sf.info(source)
    duration = reference.frames / reference.samplerate
    if duration <= 0:
        raise ValueError('Empty input timeline')
    info = sf.info(stereo)
    if info.channels != 2 or info.frames <= 0:
        raise ValueError(f'Expected nonempty stereo output: {stereo}')
    if abs(info.frames / info.samplerate - duration) > max(1 / info.samplerate, 1 / reference.samplerate):
        raise ValueError(f'Stereo timeline differs from input: {stereo}')
    for block in sf.blocks(stereo, blocksize=65536):
        if not np.isfinite(block).all():
            raise ValueError(f'Nonfinite samples: {stereo}')
    return duration


def validate_duplexchat_stereo_files(stereo_files: list[str | Path]) -> int:
    import numpy as np
    import soundfile as sf

    for index, stereo in enumerate(stereo_files, start=1):
        path = Path(stereo)
        if path.name != f"stereo_{index}.wav":
            raise ValueError(f"Unexpected DuplexChat stereo filename: {path}")
        info = sf.info(path)
        if info.channels != 2 or info.samplerate != 24_000 or info.frames <= 0:
            raise ValueError(f"Invalid 24 kHz stereo conversation: {path}")
        for block in sf.blocks(path, blocksize=65536):
            if not np.isfinite(block).all():
                raise ValueError(f"Nonfinite samples: {path}")
    return len(stereo_files)


def validate_output(pipeline: str, source: Path, stereo: Path | None, metadata: dict) -> float:
    if pipeline == "duplexchat":
        validate_duplexchat_stereo_files(metadata["stereo_files"])
        # A DuplexChat result contains cropped conversations, not one track
        # aligned to the input.  The source may also be an FFmpeg-supported
        # format (for example M4A) that SoundFile cannot decode.
        import soundfile as sf
        return sum(
            (info := sf.info(Path(stereo))).frames / info.samplerate
            for stereo in metadata["stereo_files"]
        )
    if stereo is None:
        raise ValueError("Full-input output requires a stereo WAV")
    return validate_stereo(source, stereo)


def vilier_tracks(manifest: Path) -> list[Path]:
    speakers = json.loads(manifest.read_text())['speakers']
    if len(speakers) != 2:
        raise ValueError(f'Vilier must produce exactly two speakers; got {len(speakers)}')
    return [manifest.parent / speaker['track_wav'] for speaker in speakers]


def reusable(pipeline: str, output: Path, identity: str, source: Path) -> dict | None:
    try:
        result = json.loads((output / 'run.json').read_text())
        if result['status'] != 'complete' or result['fingerprint'] != identity:
            return None
        if pipeline == "duplexchat":
            stereo_files = result["metadata"]["stereo_files"]
            validate_duplexchat_stereo_files(stereo_files)
            if result["stereo_sha256"] != stereo_files_sha256(stereo_files):
                return None
        elif pipeline == "sommelier":
            stereo = output / result["stereo_path"]
            validate_stereo(source, stereo)
            if result['audio_sha256'] != sha256(stereo):
                return None
        else:
            stereo = output / STEREO_FILENAME
            validate_stereo(source, stereo)
            if result['audio_sha256'] != sha256(stereo):
                return None
        return result
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def run_sample(pipeline, sample, output, config, code, adapter, force=False) -> dict:
    source = Path(sample['mixture'])
    output = Path(output)
    started = time.perf_counter()
    result = {'key': sample['key'], 'pipeline': pipeline, 'config': config, 'code': code,
              'input': str(source), 'status': 'running', 'resumed': False}
    try:
        identity = fingerprint(pipeline, source, config, code)
        previous = reusable(pipeline, output, identity, source) if not force else None
        if previous:
            return {**previous, 'resumed': True}
        if output.exists():
            archive = output.parent / '.history' / uuid.uuid4().hex / output.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
        output.mkdir(parents=True, exist_ok=True)
        result['fingerprint'] = identity
        write_json(output / 'run.json', result)
        stereo, metadata = adapter(source, output, config)
        duration = validate_output(pipeline, source, stereo, metadata)
        elapsed = time.perf_counter() - started
        result.update(status='complete', duration_sec=duration, inference_seconds=elapsed,
                      rtf=elapsed / duration if duration else None, metadata=metadata)
        if pipeline == "duplexchat":
            result["stereo_sha256"] = stereo_files_sha256(metadata["stereo_files"])
        elif pipeline == "sommelier":
            # Sommelier emits a numbered stereo_N.wav for dialogue inputs.
            # Do not create audio.stereo.wav: it duplicates the same audio and
            # discards the dialogue index needed by downstream pairing.
            result["stereo_path"] = stereo.name
            result["audio_sha256"] = sha256(stereo)
        else:
            canonical = output / STEREO_FILENAME
            if stereo.resolve() != canonical.resolve():
                shutil.copy2(stereo, canonical)
            result["audio_sha256"] = sha256(canonical)
    except Exception as exc:
        trace = traceback.format_exc()
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      traceback=trace, inference_seconds=time.perf_counter() - started)
        get_logger(pipeline).error("Sample %s failed: %s: %s", sample['key'], type(exc).__name__, exc)
    write_json(output / 'run.json', result)
    return result
