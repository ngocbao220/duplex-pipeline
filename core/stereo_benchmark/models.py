"""Lazy optional model metrics for already prepared 16 kHz speech."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
SQUIM_WINDOW_SAMPLES = SAMPLE_RATE * 10
SQUIM_BATCH_SIZE = 8
SPEAKER_WINDOW_SAMPLES = SAMPLE_RATE * 3
SPEAKER_BATCH_SIZE = 16


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def prepare_speech(audio: np.ndarray, sample_rate: int, mask: np.ndarray, frame_sec: float) -> np.ndarray:
    """Extract VAD-active samples and resample once for all model metrics."""
    frame_samples = max(1, round(frame_sec * sample_rate))
    speech = np.concatenate(
        [audio[index * frame_samples:min((index + 1) * frame_samples, len(audio))]
         for index, active in enumerate(mask) if active]
    ) if mask.any() else np.array([], dtype=np.float32)
    if sample_rate == SAMPLE_RATE or not speech.size:
        return np.asarray(speech, dtype=np.float32)
    import torch
    import torchaudio.functional as ta_functional

    return ta_functional.resample(torch.from_numpy(speech).unsqueeze(0), sample_rate, SAMPLE_RATE).squeeze(0).numpy()


def acoustic_metrics(left: np.ndarray, right: np.ndarray, device: str, dnsmos_model_dir: Path | None = None) -> dict:
    """Compute reference-free acoustic metrics from prepared 16 kHz speech."""
    squim_left, squim_right = _squim_many((left, right), device)
    return {
        "dnsmos": _dnsmos_metrics(left, right, dnsmos_model_dir),
        "squim": _combine_channels(squim_left, squim_right),
        "nisqa": _nisqa_metrics(left, right, device),
    }


def _nisqa_metrics(left: np.ndarray, right: np.ndarray, device: str) -> dict:
    return _combine_channels(_nisqa_one(left, device), _nisqa_one(right, device))


def _load_nisqa_model_class():
    try:
        from nisqa.NISQA_model import nisqaModel
        return nisqaModel
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from nisqa.nisqa_lib import nisqaModel
        return nisqaModel
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from nisqa import nisqaModel
        return nisqaModel
    except (ImportError, ModuleNotFoundError):
        pass
    raise ModuleNotFoundError("Could not import nisqaModel from nisqa.NISQA_model or nisqa.nisqa_lib")


@lru_cache(maxsize=1)
def _ensure_nisqa_installed() -> bool:
    try:
        _load_nisqa_model_class()
        return True
    except ModuleNotFoundError:
        return False


def _nisqa_one(audio: np.ndarray, device: str) -> dict:
    if not audio.size:
        return unavailable("Audio is empty for NISQA")
    if not _ensure_nisqa_installed():
        return unavailable("NISQA unavailable: local benchmark environment is missing the 'nisqa' package; install its local wheel during environment provisioning")
    try:
        nisqaModel = _load_nisqa_model_class()
        import tempfile
        import soundfile as sf

        nisqa_target = _nisqa_checkpoint_path()

        # NISQA opens this path in a second audio backend.  A named open file
        # can fail there on mounted or restrictive server filesystems.
        with tempfile.TemporaryDirectory(prefix="nisqa-smoke-") as directory:
            audio_path = Path(directory) / "input.wav"
            sf.write(audio_path, audio, SAMPLE_RATE)
            sf.info(audio_path)  # Report a WAV write/read problem before NISQA.
            model = nisqaModel(_nisqa_prediction_args(nisqa_target, audio_path))
            res = model.predict()
            score = float(res["mos_pred"].iloc[0]) if hasattr(res, "iloc") else float(res["mos_pred"])
            return {"status": "ok", "nisqa_mos": score}
    except Exception as error:
        return unavailable(f"NISQA unavailable: {type(error).__name__}: {error}")


def _nisqa_prediction_args(checkpoint: Path, audio_path: Path) -> dict:
    """Supply runtime-only defaults absent from older local NISQA checkpoints."""
    return {
        "mode": "predict_file",
        "deg": str(audio_path),
        "pretrained_model": str(checkpoint),
        "compile": False,
        "ms_channel": None,
        "tr_parallel": False,
        "tr_bs_val": 1,
        "tr_num_workers": 0,
        "output_dir": None,
        "name": "nisqa_local",
    }


def _nisqa_checkpoint_path() -> Path:
    from core.model_utils import require_local_model_path

    return require_local_model_path(
        "nisqa.tar", env_var="NISQA_MODEL_PATH", default_subpath="nisqa.tar",
        model_name_hint="NISQA", require_file=True,
    )


@lru_cache(maxsize=4)
def _dnsmos_scorer(model_dir: Path):
    from .dnsmos import DNSMOSScorer

    return DNSMOSScorer(model_dir)


def _dnsmos_metrics(left: np.ndarray, right: np.ndarray, model_dir: Path | None) -> dict:
    try:
        scorer = _dnsmos_scorer(Path(model_dir) if model_dir else Path("models/dnsmos"))
        return _combine_channels(scorer.score(left, SAMPLE_RATE), scorer.score(right, SAMPLE_RATE))
    except Exception as error:
        return unavailable(f"DNSMOS unavailable: {type(error).__name__}: {error}")


@lru_cache(maxsize=2)
def _squim_model(device: str):
    import torch
    import torchaudio

    squim_target = _squim_checkpoint_path()
    model = torchaudio.models.squim_objective_base()
    state_dict = torch.load(squim_target, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _squim_checkpoint_path() -> Path:
    from core.model_utils import require_local_model_path

    return require_local_model_path(
        None, env_var="SQUIM_MODEL_PATH", default_subpath="squim_objective_dns2020.pth",
        model_name_hint="SQUIM", require_file=True,
    )


def _squim_many(audios: tuple[np.ndarray, np.ndarray], device: str) -> tuple[dict, dict]:
    """Score equal-length chunks together without changing per-track averaging."""
    try:
        import torch

        model = _squim_model(device)
        chunks: dict[int, list[tuple[int, object]]] = defaultdict(list)
        values = [[], []]
        for channel, audio in enumerate(audios):
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            for start in range(0, waveform.numel(), SQUIM_WINDOW_SAMPLES):
                chunk = waveform[start:start + SQUIM_WINDOW_SAMPLES]
                if chunk.numel() >= SAMPLE_RATE:
                    chunks[chunk.numel()].append((channel, chunk))
        for length_chunks in chunks.values():
            for start in range(0, len(length_chunks), SQUIM_BATCH_SIZE):
                batch_items = length_chunks[start:start + SQUIM_BATCH_SIZE]
                batch = torch.stack([chunk for _, chunk in batch_items]).to(device)
                with torch.inference_mode():
                    stois, pesqs, si_sdrs = model(batch)
                for (channel, _), stoi, pesq, si_sdr in zip(batch_items, stois, pesqs, si_sdrs):
                    values[channel].append((float(stoi.item()), float(pesq.item()), float(si_sdr.item())))
        return tuple(_squim_result(channel_values) for channel_values in values)  # type: ignore[return-value]
    except Exception as error:
        failure = unavailable(f"SQUIM unavailable: {type(error).__name__}: {error}")
        return failure, failure


def _squim_result(values: list[tuple[float, float, float]]) -> dict:
    if not values:
        return unavailable("Audio too short for SQUIM")
    scores = np.asarray(values)
    return {
        "status": "ok", "sq_stoi": float(scores[:, 0].mean()), "sq_pesq": float(scores[:, 1].mean()),
        "sq_si_sdr": float(scores[:, 2].mean()),
    }


def speaker_metrics(left: np.ndarray, right: np.ndarray, device: str) -> dict:
    """ITC/ITD from cached SpeechBrain ECAPA embeddings; no ground truth is used."""
    try:
        left_embeddings, right_embeddings = _embeddings_many((left, right), device)
    except Exception as error:
        failure = unavailable(f"speaker encoder unavailable: {type(error).__name__}: {error}")
        return {"itc": {"left": failure, "right": failure, "mean": failure}, "itd": failure}
    left_itc = _itc(left_embeddings)
    right_itc = _itc(right_embeddings)
    result = {
        "itc": {"left": left_itc, "right": right_itc, "mean": _mean_available(left_itc, right_itc)},
        "itd": unavailable("insufficient_speech"),
    }
    if left_embeddings is not None and right_embeddings is not None:
        from .dynamics import cosine_distinctiveness

        left_centroid = _normalised_mean(left_embeddings)
        right_centroid = _normalised_mean(right_embeddings)
        cosine = float(np.dot(left_centroid, right_centroid))
        result["itd"] = {"status": "ok", "inter_track_cosine_similarity": cosine, "itd": cosine_distinctiveness(left_centroid, right_centroid)}
    return result


@lru_cache(maxsize=2)
def _speaker_encoder(device: str):
    from core.compat import ensure_runtime_compat
    from core.model_utils import enforce_offline_mode, is_offline_mode
    ensure_runtime_compat()
    enforce_offline_mode()
    from speechbrain.inference.speaker import EncoderClassifier

    offline = is_offline_mode()
    if offline:
        local_target = _speaker_model_path()
        return _load_local_speechbrain_encoder(EncoderClassifier, local_target, device)
    else:
        from core.model_utils import resolve_local_model_path
        target, is_dir = resolve_local_model_path(
            "speechbrain/spkrec-ecapa-voxceleb", env_var="SPEECHBRAIN_MODEL_PATH", default_subpath="spkrec-ecapa-voxceleb"
        )
        return EncoderClassifier.from_hparams(
            source=str(target),
            savedir=str(target) if (is_dir or Path(target).is_dir()) else None,
            run_opts={"device": device},
            local_files_only=False,
        )


def _speaker_model_path() -> Path:
    from core.model_utils import assert_local_model_exists, require_local_model_path

    path = require_local_model_path(
        None, env_var="SPEECHBRAIN_MODEL_PATH",
        default_subpath="spkrec-ecapa-voxceleb", model_name_hint="SpeechBrain ECAPA",
    )
    return assert_local_model_exists(
        path,
        required_files=["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt", "mean_var_norm_emb.ckpt"],
        model_name_hint="SpeechBrain ECAPA",
    )


def _load_local_speechbrain_encoder(encoder_classifier, local_target: Path, device: str):
    """Load ECAPA without allowing its YAML pretrainer paths to contact the Hub."""
    from speechbrain.inference import interfaces as sb_interfaces
    from speechbrain.utils import fetching as sb_fetching
    from speechbrain.utils import parameter_transfer as sb_parameter_transfer
    from speechbrain.utils.fetching import FetchFrom, FetchSource

    local_root = local_target.resolve()
    original_interface_fetch = sb_interfaces.fetch
    original_transfer_fetch = sb_parameter_transfer.fetch
    original_fetching_fetch = sb_fetching.fetch

    def fetch_local(filename, _source, *fetch_args, **fetch_kwargs):
        return original_fetching_fetch(
            filename, FetchSource(FetchFrom.LOCAL, str(local_root)), *fetch_args, **fetch_kwargs
        )

    # ECAPA hyperparams may embed a remote repo ID in pretrainer.paths.  Patch
    # every SpeechBrain import binding only for this construction.
    sb_interfaces.fetch = fetch_local
    sb_parameter_transfer.fetch = fetch_local
    sb_fetching.fetch = fetch_local
    try:
        return encoder_classifier.from_hparams(
            source=FetchSource(FetchFrom.LOCAL, str(local_root)),
            savedir=str(local_root),
            run_opts={"device": device},
        )
    finally:
        sb_interfaces.fetch = original_interface_fetch
        sb_parameter_transfer.fetch = original_transfer_fetch
        sb_fetching.fetch = original_fetching_fetch


def _embeddings_many(audios: tuple[np.ndarray, np.ndarray], device: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    import torch

    windows: list[tuple[int, object]] = []
    for channel, audio in enumerate(audios):
        signal = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        for start in range(0, signal.numel() - SPEAKER_WINDOW_SAMPLES + 1, SPEAKER_WINDOW_SAMPLES):
            windows.append((channel, signal[start:start + SPEAKER_WINDOW_SAMPLES]))
    if not windows:
        return None, None
    encoder = _speaker_encoder(device)
    values = [[], []]
    for start in range(0, len(windows), SPEAKER_BATCH_SIZE):
        batch_items = windows[start:start + SPEAKER_BATCH_SIZE]
        batch = torch.stack([window for _, window in batch_items]).to(device)
        with torch.inference_mode():
            embeddings = encoder.encode_batch(batch).detach().cpu().numpy().reshape(len(batch_items), -1)
        for (channel, _), embedding in zip(batch_items, embeddings):
            values[channel].append(embedding)
    return tuple(np.asarray(channel_values) if channel_values else None for channel_values in values)  # type: ignore[return-value]


def _combine_channels(left: dict, right: dict) -> dict:
    result = {"left": left, "right": right}
    if left.get("status") == right.get("status") == "ok":
        shared = set(left) & set(right) - {"status"}
        result["mean"] = {name: float((left[name] + right[name]) / 2) for name in shared}
    return result


def _itc(embeddings: np.ndarray | None) -> dict:
    if embeddings is None or len(embeddings) < 2:
        return unavailable("insufficient_speech")
    centroid = _normalised_mean(embeddings)
    similarities = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) @ centroid
    return {"status": "ok", "itc": float(np.mean(similarities)), "window_count": len(embeddings)}


def _normalised_mean(embeddings: np.ndarray) -> np.ndarray:
    centroid = np.mean(embeddings, axis=0)
    return centroid / np.linalg.norm(centroid)


def _mean_available(left: dict, right: dict) -> dict:
    if left.get("status") != "ok" or right.get("status") != "ok":
        return unavailable("insufficient_speech")
    return {"status": "ok", "itc": float((left["itc"] + right["itc"]) / 2)}
