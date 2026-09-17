"""Purpose: Implement DuplexChat separator model loading and chunk inference.

Inputs: Waveforms, separator backend/model and runtime device settings.
Outputs: Timeline-aligned pair of separated waveform tensors.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as F_audio
from diffusers import DPMSolverMultistepScheduler
from huggingface_hub import hf_hub_download

from .model_options import SEPARATION_MODELS, resolve_model_alias
from core.model_utils import (
    enforce_offline_mode,
    resolve_local_model_path,
)
from core.local_model_validation import (
    LocalModelValidationError,
    validate_dialoguesidon_model as validate_local_dialoguesidon_model,
)

REPO_ID = "sarulab-speech/DialogueSidon"
MODEL_FILES = ["ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"]
SAMPLE_RATE_IN = 16_000
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 10.0

_cache: dict = {}
_cache_lock = threading.Lock()


def validate_dialoguesidon_model(model_dir: str | Path) -> dict[str, str]:
    """Validate an explicit local DialogueSidon bundle without contacting Hugging Face."""
    return validate_local_dialoguesidon_model(model_dir)


def _canonical_export_device(device: str | torch.device) -> torch.device:
    """Use an indexed CUDA spelling for exported graph assertions.

    PyTorch materializes tensors moved to generic ``cuda`` on GPU 0 as
    ``cuda:0``.  DialogueSidon's exported graph guards compare the spelling
    strictly, so retarget the graph and all tensors to the explicit device.
    Explicit nonzero GPUs remain unchanged.
    """
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        return torch.device("cuda:0")
    return requested


def _retarget_exported_module(module, device: torch.device):
    """Replace device literals captured by a torch.export graph."""
    def retarget(value):
        if isinstance(value, torch.device):
            return device
        # torch.export serializes device constants both as torch.device values
        # and as strings.  The latter kept DialogueSidon's original "cuda"
        # literal while chunks were deliberately moved to "cuda:0".
        if isinstance(value, str) and value.startswith("cuda"):
            return str(device)
        if isinstance(value, tuple):
            return tuple(retarget(item) for item in value)
        if isinstance(value, list):
            return [retarget(item) for item in value]
        if isinstance(value, dict):
            return {key: retarget(item) for key, item in value.items()}
        return value

    for node in module.graph.nodes:
        node.args = retarget(node.args)
        node.kwargs = retarget(node.kwargs)
    module.recompile()
    return module


def _load_exported_module(path: str | Path, device: torch.device):
    """Load an exported PT2 module directly via torch.export.load and retarget device."""
    module = torch.export.load(str(path)).module()
    return _retarget_exported_module(module, device).to(device)


def load_separation_models(
    device: str = "cuda",
    backend: str = "dialoguesidon",
    model_id: str | None = None,
) -> dict:
    """Download and load a configured speech-separation backend."""
    enforce_offline_mode()
    backend_norm = backend.strip().lower()
    model_id = resolve_model_alias(model_id, SEPARATION_MODELS)
    if backend_norm == "dialoguesidon":
        return _load_dialoguesidon_models(device, model_id)
    raise ValueError(f"Unsupported separation backend '{backend}'. Use dialoguesidon.")


def _load_dialoguesidon_models(device: str = "cuda", model_id: str | None = None) -> dict:
    """Download and load DialogueSidon models. Cached per device."""
    enforce_offline_mode()
    raw_id = model_id or REPO_ID
    local_target, is_dir = resolve_local_model_path(
        raw_id, env_var="DIALOGUESIDON_MODEL_PATH", default_subpath="DialogueSidon"
    )
    resolved = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    cache_key = ("dialoguesidon", str(local_target), resolved)
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

        if is_dir or Path(local_target).is_dir():
            # An explicit local directory is fully self-contained. Never use a
            # Hub lookup/cache as a fallback when validating or loading it.
            paths = validate_local_dialoguesidon_model(local_target)
        else:
            paths = {
                f: hf_hub_download(repo_id=str(local_target), filename=f, local_files_only=True)
                for f in MODEL_FILES
            }

        with open(paths["metadata.json"]) as fp:
            meta = json.load(fp)

        torch_device = _canonical_export_device(resolved)

        # Enable TF32 for faster float32 matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ssl_encoder = _load_exported_module(paths["ssl_encoder.pt2"], torch_device)
        diffusion_head = _load_exported_module(paths["diffusion_head.pt2"], torch_device)
        vae_decoder = _load_exported_module(paths["vae_decoder.pt2"], torch_device)

        latent_norm_mean = torch.tensor(
            meta["latent_norm_mean"], dtype=torch.float32, device=torch_device
        ).view(1, 1, -1)
        latent_norm_std = torch.tensor(
            meta["latent_norm_std"], dtype=torch.float32, device=torch_device
        ).view(1, 1, -1)

        try:
            scheduler = DPMSolverMultistepScheduler.from_config(
                meta["ddpm_config"],
                algorithm_type="dpmsolver++",
                timestep_spacing="linspace",
            )
        except TypeError:
            scheduler = DPMSolverMultistepScheduler.from_config(meta["ddpm_config"])

        models = {
            "ssl_encoder": ssl_encoder,
            "diffusion_head": diffusion_head,
            "vae_decoder": vae_decoder,
            "latent_norm_mean": latent_norm_mean,
            "latent_norm_std": latent_norm_std,
            "latent_norm_initialized": meta["latent_norm_initialized"],
            "scheduler": scheduler,
            "latent_dim": meta["latent_dim"],
            "sample_rate": meta["sample_rate"],
            "input_sample_rate": SAMPLE_RATE_IN,
            "device": torch_device,
            "backend": "dialoguesidon",
            "model_id": str(local_target),
        }
        _cache[cache_key] = models
        return models


# ── helpers ───────────────────────────────────────────────────────────────────

def _pad_batch(
    features: list[torch.Tensor],
    pad_to_multiple_of: int = 2,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_length = max(f.shape[0] for f in features)
    if pad_to_multiple_of:
        target_length = (
            (target_length + pad_to_multiple_of - 1)
            // pad_to_multiple_of
            * pad_to_multiple_of
        )
    batch_size = len(features)
    feature_dim = features[0].shape[1]
    device = features[0].device
    padded = torch.full(
        (batch_size, target_length, feature_dim), padding_value,
        dtype=torch.float32, device=device,
    )
    mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=device)
    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = 1
    return padded, mask


def _extract_fbank(waveforms: list[torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    features = []
    for wav in waveforms:
        if wav.ndim > 1:
            wav = wav[0]
        # kaldi.fbank requires CPU
        feat = kaldi.fbank(
            wav.cpu().unsqueeze(0),
            sample_frequency=SAMPLE_RATE_IN,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            preemphasis_coefficient=0.97,
            remove_dc_offset=True,
            window_type="povey",
            use_energy=False,
            energy_floor=1.192092955078125e-07,
        )
        mean = feat.mean(0, keepdim=True)
        var = feat.var(0, keepdim=True)
        feat = (feat - mean) / torch.sqrt(var + 1e-5)
        features.append(feat.to(device))

    stride = 2
    input_features, attention_mask = _pad_batch(features)
    b, t, c = input_features.shape
    if t < stride * 2:
        pad_len = stride * 2 - t
        input_features = torch.nn.functional.pad(input_features, (0, 0, 0, pad_len))
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len))
        b, t, c = input_features.shape
    t = (t // stride) * stride
    input_features = input_features[:, :t, :]
    attention_mask = attention_mask[:, :t]
    input_features = input_features.reshape(b, t // stride, c * stride)
    attention_mask = attention_mask[:, 1::stride]
    return {"input_features": input_features, "attention_mask": attention_mask}


def _normalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return ((latents.float() - models["latent_norm_mean"]) / models["latent_norm_std"]).to(latents.dtype)


def _denormalize(latents: torch.Tensor, models: dict) -> torch.Tensor:
    if not models["latent_norm_initialized"]:
        return latents
    return (latents.float() * models["latent_norm_std"] + models["latent_norm_mean"]).to(latents.dtype)


def _channel_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    # Dùng Dot Product (không chia cho mẫu số) để ưu tiên các đoạn có giọng nói lớn (energy cao).
    # Nếu chia cho norm (Pearson correlation), các đoạn im lặng (chỉ có nhiễu noise) 
    # sẽ bị phóng đại và làm đảo lộn logic ghép kênh (swap).
    return float(torch.dot(a, b))


def _maybe_swap(
    prev_overlap: torch.Tensor, curr_chunk: torch.Tensor, overlap_samples: int
) -> tuple[torch.Tensor, bool]:
    if overlap_samples <= 0 or prev_overlap.shape[0] != 2 or curr_chunk.shape[0] != 2:
        return curr_chunk, False
    effective_overlap = min(overlap_samples, prev_overlap.shape[-1], curr_chunk.shape[-1])
    if effective_overlap <= 0:
        return curr_chunk, False
    prev_ov = prev_overlap[:, -effective_overlap:]
    curr_ov = curr_chunk[:, :effective_overlap]
    direct = (
        _channel_similarity(prev_ov[0], curr_ov[0])
        + _channel_similarity(prev_ov[1], curr_ov[1])
    )
    swapped = (
        _channel_similarity(prev_ov[0], curr_ov[1])
        + _channel_similarity(prev_ov[1], curr_ov[0])
    )
    if swapped > direct:
        return curr_chunk[[1, 0], :], True
    return curr_chunk, False


def _blend_overlap(
    stitched: torch.Tensor, pred: torch.Tensor, overlap_samples: int
) -> torch.Tensor:
    """Align and crossfade two chunks using only samples present in both."""
    effective_overlap = min(
        overlap_samples,
        stitched.shape[-1],
        pred.shape[-1],
    )
    if effective_overlap <= 0:
        return torch.cat([stitched, pred], dim=-1)

    previous = stitched[:, -effective_overlap:]
    pred, _ = _maybe_swap(previous, pred, effective_overlap)
    current = pred[:, :effective_overlap]
    fade = torch.linspace(
        0.0,
        1.0,
        effective_overlap,
        device=stitched.device,
        dtype=stitched.dtype,
    ).unsqueeze(0)
    blended = previous * (1 - fade) + current * fade
    return torch.cat(
        [stitched[:, :-effective_overlap], blended, pred[:, effective_overlap:]],
        dim=-1,
    )


@torch.inference_mode()
def _separate_chunk(wav: torch.Tensor, num_steps: int, models: dict) -> torch.Tensor:
    """Separate a single chunk. Input: (1, T) at 16 kHz. Output: (2, T_out) at model sr."""
    # Re-enable TF32 in case pyannote disabled it
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = models["device"]
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_device(device)
    latent_dim = models["latent_dim"]

    noisy_ssl = _extract_fbank([wav.view(-1)], device)

    features, pred0, pred1 = models["ssl_encoder"](
        noisy_ssl["input_features"], noisy_ssl["attention_mask"]
    )

    predicted_latents = torch.cat([pred0, pred1], dim=-1)
    conditioning = torch.cat([_normalize(predicted_latents, models), features], dim=-1)

    seq_len = conditioning.shape[1]
    scheduler = models["scheduler"]
    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(
        (1, seq_len, latent_dim * 2), device=device, dtype=conditioning.dtype
    )
    for t in scheduler.timesteps:
        t_batch = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        latents = scheduler.step(
            models["diffusion_head"](latents, t_batch, conditioning), t, latents
        ).prev_sample

    latents = _denormalize(latents, models)
    spk0 = models["vae_decoder"](latents[:, :, :latent_dim].transpose(1, 2)).squeeze(0)
    spk1 = models["vae_decoder"](latents[:, :, latent_dim:].transpose(1, 2)).squeeze(0)
    return torch.cat([spk0, spk1], dim=0)  # (2, T)


# ── public API ────────────────────────────────────────────────────────────────

def run_separation(
    wav: torch.Tensor,
    sample_rate: int,
    num_steps: int,
    models: dict,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    progress_callback: Callable[[str, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Separate a (1, T) mono waveform into two speaker tracks.

    Returns (spk0, spk1, out_sr) where each track is a (1, T) CPU float32 tensor.
    """
    device = models["device"]
    out_sr: int = models["sample_rate"]
    input_sample_rate = int(models.get("input_sample_rate", SAMPLE_RATE_IN))

    if sample_rate != input_sample_rate:
        wav = F_audio.resample(wav, sample_rate, input_sample_rate)
    # Keep the full source timeline on CPU. Only the active chunk belongs on
    # CUDA; otherwise a small chunk setting still leaves the whole recording
    # (and its growing stitched output) consuming VRAM.
    wav = wav.cpu()

    chunk_samples = int(chunk_seconds * input_sample_rate)
    total_samples = wav.shape[-1]

    if total_samples <= chunk_samples:
        if progress_callback is not None:
            progress_callback("start", 1)
        chunk = wav.to(device)
        max_val = chunk.abs().max().clamp_min(1e-6)
        wav_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))
        separated = _separate_chunk(wav_norm, num_steps, models)
        target_out = max(1, round(total_samples * out_sr / input_sample_rate))
        if separated.shape[-1] > target_out:
            separated = separated[:, :target_out]
        elif separated.shape[-1] < target_out:
            separated = torch.cat(
                [separated, torch.zeros(2, target_out - separated.shape[-1], device=device)],
                dim=-1,
            )
        separated = separated.cpu()
        if progress_callback is not None:
            progress_callback("advance", 1)
    else:
        overlap_samples_in = int(overlap_seconds * input_sample_rate)
        hop_samples = chunk_samples - overlap_samples_in
        starts = list(range(0, total_samples, hop_samples))
        if progress_callback is not None:
            progress_callback("start", len(starts))
        stitched: torch.Tensor | None = None
        prev_end_in = 0

        for start in starts:
            end = min(start + chunk_samples, total_samples)
            chunk = wav[:, start:end].to(device)
            max_val = chunk.abs().max().clamp_min(1e-6)
            chunk_norm = torch.nn.functional.pad(0.9 * chunk / max_val, (160, 160))
            pred = _separate_chunk(chunk_norm, num_steps, models)

            target_out = max(1, round((end - start) * out_sr / input_sample_rate))
            if pred.shape[-1] > target_out:
                pred = pred[:, :target_out]
            elif pred.shape[-1] < target_out:
                pred = torch.cat(
                    [pred, torch.zeros(2, target_out - pred.shape[-1], device=device)], dim=-1
                )

            pred = pred.cpu()
            if stitched is None:
                stitched = pred
                prev_end_in = end
                if progress_callback is not None:
                    progress_callback("advance", 1)
                continue

            overlap_in = max(0, prev_end_in - start)
            overlap_out = max(0, min(
                round(overlap_in * out_sr / input_sample_rate),
                stitched.shape[-1],
                pred.shape[-1],
            ))
            if overlap_out > 0:
                stitched = _blend_overlap(stitched, pred, overlap_out)
            else:
                stitched = torch.cat([stitched, pred], dim=-1)
            prev_end_in = end
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress_callback is not None:
                progress_callback("advance", 1)

        separated = stitched  # type: ignore[assignment]

    spk0 = separated[0:1].cpu()
    spk1 = separated[1:2].cpu()
    if progress_callback is not None:
        progress_callback("close", 0)
    return spk0, spk1, out_sr
