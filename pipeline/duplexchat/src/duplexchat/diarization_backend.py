"""Purpose: Implement DuplexChat diarization model loading and inference.

Inputs: Audio paths, diarization backend/model settings and device.
Outputs: Normalized diarization segment dictionaries.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Protocol

import logging
import os
import warnings

os.environ["MPLBACKEND"] = "Agg"

# Suppress noisy third-party warnings before any heavy imports
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
warnings.filterwarnings("ignore", message=".*Migrating your old cache.*")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


def _suppress_nemo_logs() -> None:
    """Tắt NeMo W/I logs không liên quan đến lỗi thực sự."""
    warnings.filterwarnings("ignore", message=r".*If you intend to do training or fine-tuning.*")
    warnings.filterwarnings("ignore", message=r".*setup_training_data.*")
    warnings.filterwarnings("ignore", message=r".*ModelPT.*")
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
        nemo_logging.set_verbosity(logging.ERROR)
    except Exception:
        pass
    for name in (
        "nemo_logging",
        "nemo",
        "nemo.collections",
        "nemo.core",
        "nemo.utils",
        "pytorch_lightning",
        "lightning",
        "transformers.utils.hub",
        "transformers.configuration_utils",
        "huggingface_hub.utils",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

_suppress_nemo_logs()


import torch
import torch.nn.functional as F
from huggingface_hub import get_token

from .audio import load_wav_tensor
from .model_options import DIARIZATION_MODELS, infer_diarization_backend, resolve_model_alias


DIARIZATION_SAMPLE_RATE = 16_000

if TYPE_CHECKING:
    from pyannote.audio import Pipeline


class FileDiarizationAdapter:
    backend: str

    def diarize_file(self, wav_path: Path) -> list[dict]:
        raise NotImplementedError


def _disable_sortformer_streaming_mode(model: Any) -> None:
    """Tắt cờ streaming_mode trong attribute và _cfg/cfg của NeMo model (tránh NotImplementedError ở NeMo cũ)."""
    if hasattr(model, "streaming_mode"):
        try:
            model.streaming_mode = False
        except Exception:
            pass
    for cfg_attr in ("_cfg", "cfg"):
        c = getattr(model, cfg_attr, None)
        if c is not None:
            try:
                from omegaconf import open_dict
                with open_dict(c):
                    c["streaming_mode"] = False
            except Exception:
                try:
                    c["streaming_mode"] = False
                except Exception:
                    try:
                        setattr(c, "streaming_mode", False)
                    except Exception:
                        pass


class SortformerDiarizationAdapter(FileDiarizationAdapter):
    backend = "sortformer"

    def __init__(self, model: Any, max_chunk_duration: float | str | None = 120.0):
        self.model = model
        if max_chunk_duration is None or str(max_chunk_duration).strip().lower() in ("full", "none", "inf"):
            self.max_chunk_duration = float("inf")
        else:
            try:
                val = float(max_chunk_duration)
                self.max_chunk_duration = float("inf") if val <= 0 else val
            except (ValueError, TypeError):
                self.max_chunk_duration = float("inf")

    def _diarize_single(self, wav_path_str: str) -> list[dict]:
        with torch.inference_mode():
            try:
                predicted = self.model.diarize(audio=wav_path_str, batch_size=1)
            except NotImplementedError as exc:
                if "Streaming mode is not implemented" in str(exc):
                    _disable_sortformer_streaming_mode(self.model)
                    predicted = self.model.diarize(audio=wav_path_str, batch_size=1)
                else:
                    raise
        return _segments_from_sortformer_output(predicted)

    def diarize_file(self, wav_path: Path) -> list[dict]:
        import soundfile as sf
        import tempfile

        try:
            info = sf.info(str(wav_path))
            duration = info.duration
            sample_rate = info.samplerate
        except Exception:
            duration = None
            sample_rate = 16000

        # Nếu max_chunk_duration là inf ("full") hoặc audio ngắn hơn max_chunk_duration, diarize trực tiếp toàn bộ
        if self.max_chunk_duration == float("inf") or (duration is not None and duration <= self.max_chunk_duration):
            return self._diarize_single(str(wav_path))

        # Audio dài (> max_chunk_duration): xử lý theo chunk để tiết kiệm VRAM
        segments: list[dict] = []
        chunk_samples = int(self.max_chunk_duration * sample_rate)

        with tempfile.TemporaryDirectory(prefix="sortformer_chunk_") as tmpdir:
            tmp_dir_path = Path(tmpdir)
            with sf.SoundFile(str(wav_path)) as f:
                offset_samples = 0
                chunk_idx = 0
                while offset_samples < f.frames:
                    f.seek(offset_samples)
                    chunk_data = f.read(chunk_samples)
                    if len(chunk_data) == 0:
                        break

                    offset_sec = offset_samples / sample_rate
                    chunk_file = tmp_dir_path / f"chunk_{chunk_idx:04d}.wav"
                    sf.write(str(chunk_file), chunk_data, sample_rate)

                    chunk_segs = self._diarize_single(str(chunk_file))
                    for seg in chunk_segs:
                        seg["start"] = round(seg["start"] + offset_sec, 3)
                        seg["end"] = round(seg["end"] + offset_sec, 3)
                        segments.append(seg)

                    offset_samples += chunk_samples
                    chunk_idx += 1

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Ghép các segment liền kề cùng speaker bị cắt ngang ở biên chunk
        segments.sort(key=lambda x: x["start"])
        merged: list[dict] = []
        for seg in segments:
            if (
                merged
                and merged[-1]["speaker"] == seg["speaker"]
                and abs(seg["start"] - merged[-1]["end"]) < 0.25
            ):
                merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            else:
                merged.append(seg)
        return merged


class DiariZenDiarizationAdapter(FileDiarizationAdapter):
    backend = "diarizen"

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def diarize_file(self, wav_path: Path) -> list[dict]:
        output = self.pipeline(str(wav_path), sess_name=wav_path.stem)
        return _segments_from_annotation(output)


def load_diarization_pipeline(
    model: str,
    device: str = "cuda",
    backend: str = "auto",
    max_chunk_duration: float | str | None = None,
) -> "Pipeline | FileDiarizationAdapter":
    model = resolve_model_alias(model, DIARIZATION_MODELS) or model
    backend_norm = backend.strip().lower()
    if backend_norm == "auto":
        backend_norm = infer_diarization_backend(model)
    if backend_norm == "sortformer":
        return _load_sortformer_pipeline(model, device, max_chunk_duration=max_chunk_duration)
    if backend_norm == "nemotron":
        return _load_sortformer_pipeline(
            model, device, max_chunk_duration=max_chunk_duration, model_family="nemotron",
        )
    if backend_norm == "diarizen":
        return _load_diarizen_pipeline(model, device)
    if backend_norm != "pyannote":
        raise ValueError("Unsupported diarization backend '%s'. Use auto, pyannote, sortformer, nemotron, or diarizen." % backend)
    return _load_pyannote_pipeline(model, device)


from core.model_utils import (
    assert_local_model_exists,
    enforce_offline_mode,
    is_offline_mode,
    resolve_local_model_path,
)


def _resolve_device(device: str) -> str:
    return device if (device != "cuda" or torch.cuda.is_available()) else "cpu"


def _load_pyannote_pipeline(model: str, device: str = "cuda") -> "Pipeline":
    enforce_offline_mode()
    local_target, is_dir = resolve_local_model_path(
        model, env_var="PYANNOTE_MODEL_PATH", default_subpath="speaker-diarization-community-1"
    )
    token = get_token()

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError("pyannote.audio is required for diarization") from exc

    target_str = str(local_target)
    try:
        pipeline = Pipeline.from_pretrained(target_str, use_auth_token=token or False)
    except TypeError:
        pipeline = Pipeline.from_pretrained(target_str, token=token or False)
    except Exception as exc:
        # Retry with local_files_only=True explicitly if token call failed
        try:
            pipeline = Pipeline.from_pretrained(target_str, local_files_only=True)
        except Exception:
            raise exc

    resolved_device = _resolve_device(device)
    pipeline.to(torch.device(resolved_device))
    return pipeline


def _load_sortformer_pipeline(
    model: str,
    device: str = "cuda",
    max_chunk_duration: float | str | None = None,
    model_family: str = "sortformer",
) -> SortformerDiarizationAdapter:
    enforce_offline_mode()
    is_nemotron = model_family == "nemotron"
    model_env_var = "NEMOTRON_DIARIZATION_MODEL_PATH" if is_nemotron else "SORTFORMER_MODEL_PATH"
    model_subpath = "Nemotron-3-Diarization" if is_nemotron else "diar_streaming_sortformer_4spk-v2.1"
    local_target, is_local = resolve_local_model_path(
        model, env_var=model_env_var, default_subpath=model_subpath
    )
    _suppress_nemo_logs()

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Sortformer diarization requires NVIDIA NeMo (nemo_toolkit[asr]). "
            f"Import failed with error: {exc}. Please install: pip install nemo_toolkit[asr]"
        ) from exc

    if is_nemotron:
        try:
            from nemo.collections.asr import modules as asr_modules
            from nemo.collections.asr.modules.transformer_encoder import (
                TransformerEncoder,
                _SUPPORTED_SELF_ATTENTION_MODELS,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Nemotron-3-Diarization requires the NeMo Speech TransformerEncoder with "
                "feat_in and RoPE support. "
                "Upgrade the DuplexChat environment from its requirements-duplexchat.txt."
            ) from exc
        if "rope" not in _SUPPORTED_SELF_ATTENTION_MODELS:
            raise RuntimeError(
                "Nemotron-3-Diarization requires NeMo Speech RoPE support; the installed "
                "NeMo 3.0.0 release does not include it. Upgrade using requirements-duplexchat.txt."
            )
        # The checkpoint targets nemo.collections.asr.modules.TransformerEncoder.
        if not hasattr(asr_modules, "TransformerEncoder"):
            asr_modules.TransformerEncoder = TransformerEncoder

    def _patch_sortformer_modules():
        """Monkey-patch SortformerModules to filter unknown kwargs like fifo_len, spkcache_len."""
        try:
            import inspect
            from nemo.collections.asr.modules import sortformer_modules

            cls = getattr(sortformer_modules, "SortformerModules", None)
            if cls is not None and not getattr(cls, "_kwargs_filter_patched", False):
                orig_init = cls.__init__
                sig = inspect.signature(orig_init)
                allowed_params = set(sig.parameters.keys())

                def patched(self, *args, **kwargs):
                    safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
                    return orig_init(self, *args, **safe_kwargs)

                cls.__init__ = patched
                cls._kwargs_filter_patched = True
        except Exception:
            pass

    _patch_sortformer_modules()


    target_path = Path(local_target)
    if is_offline_mode() and not is_local and not target_path.exists():
        assert_local_model_exists(
            local_target,
            model_name_hint=(
                "NVIDIA Nemotron-3-Diarization Model (Nemotron-3-Diarization.nemo)"
                if is_nemotron else
                "NVIDIA Sortformer Diarization Model (e.g. diar_streaming_sortformer_4spk-v2.1.nemo)"
            )
        )

    target_str = str(local_target)

    # Nếu target_str là HF repo ID và chưa phải file local, tải file .nemo từ HF Hub
    if not target_path.is_file() and not is_offline_mode() and "/" in target_str:
        try:
            from huggingface_hub import hf_hub_download
            model_subname = target_str.split("/")[-1]
            nemo_filename = f"{model_subname}.nemo"
            downloaded = hf_hub_download(repo_id=target_str, filename=nemo_filename)
            target_str = downloaded
            target_path = Path(downloaded)
        except Exception:
            pass


    def _restore(path_str: str):
        """Load .nemo với fallback patch nếu NeMo version cũ không nhận tham số mới.

        Lỗi thường bị wrap trong Hydra/OmegaConf nên phải kiểm tra toàn bộ exception chain.
        """
        _patch_sortformer_modules()

        def _restore_model(path: str):
            if is_nemotron:
                return SortformerEncLabelModel.restore_from(
                    path, map_location=_resolve_device(device), strict=False,
                )
            return SortformerEncLabelModel.restore_from(path)

        def _has_unsupported_err(exc: BaseException) -> bool:
            """Đệ quy kiểm tra exception chain có chứa lỗi unexpected keyword argument không."""
            seen = set()
            e: BaseException | None = exc
            while e is not None and id(e) not in seen:
                seen.add(id(e))
                msg = str(e)
                if "unexpected keyword argument" in msg or "spkcache_len" in msg or "fifo_len" in msg:
                    return True
                e = e.__cause__ or e.__context__
            return False

        try:
            return _restore_model(path_str)
        except Exception as e:  # noqa: BLE001
            if not _has_unsupported_err(e):
                raise
            # NeMo version cũ không nhận tham số mới — patch bằng cách strip khỏi config YAML
            import tarfile
            import tempfile
            from omegaconf import OmegaConf
            with tempfile.TemporaryDirectory(prefix="nemo_patch_") as tmp:
                tmp_path = Path(tmp)
                # .nemo là tar.gz — thử cả gz lẫn uncompressed
                try:
                    with tarfile.open(path_str, "r:gz") as tar:
                        tar.extractall(tmp_path)
                except tarfile.ReadError:
                    with tarfile.open(path_str, "r:*") as tar:
                        tar.extractall(tmp_path)
                # Tìm model_config.yaml
                cfg_file = next(tmp_path.rglob("model_config.yaml"), None)
                if cfg_file is None:
                    raise RuntimeError(
                        "Sortformer config patch failed: model_config.yaml not found in .nemo archive. "
                        "Please upgrade NeMo: pip install 'nemo_toolkit[asr]>=2.3.0'"
                    ) from e
                cfg = OmegaConf.load(cfg_file)
                # Xóa các key mới khỏi sortformer_modules
                try:
                    sm_node = OmegaConf.select(cfg, "model.sortformer_modules")
                    if sm_node is not None:
                        sm_dict = OmegaConf.to_container(sm_node, resolve=False)
                        for k in ["spkcache_len", "fifo_len", "chunk_len", "total_buffer_in_secs"]:
                            sm_dict.pop(k, None)
                        OmegaConf.update(cfg, "model.sortformer_modules", sm_dict, merge=False)
                except Exception:
                    pass
                OmegaConf.save(cfg, cfg_file)
                # Đóng gói lại thành .nemo tạm
                patched = tmp_path / "patched.nemo"
                with tarfile.open(str(patched), "w:gz") as tar:
                    for f in sorted(tmp_path.rglob("*")):
                        if f != patched and f.is_file():
                            tar.add(f, arcname=str(f.relative_to(tmp_path)))
                return _restore_model(str(patched))


    if target_path.is_file() and target_str.endswith(".nemo"):
        diar_model = _restore(target_str)
    elif target_path.is_dir():
        nemo_files = list(target_path.glob("*.nemo"))
        if nemo_files:
            diar_model = _restore(str(nemo_files[0]))
        else:
            try:
                diar_model = _restore(target_str)
            except Exception:
                diar_model = SortformerEncLabelModel.from_pretrained(target_str)
    else:
        try:
            diar_model = _restore(target_str)
        except Exception:
            diar_model = SortformerEncLabelModel.from_pretrained(target_str)

    # Nếu NeMo version hiện tại không hỗ trợ streaming inference (v2.1/v2.2 ném NotImplementedError trong forward),
    # chủ động tắt cờ streaming_mode trong _cfg để chạy offline forward mà không văng lỗi.
    try:
        import inspect
        forward_src = inspect.getsource(diar_model.forward)
        if not is_nemotron and 'raise NotImplementedError("Streaming mode is not implemented yet.")' in forward_src:
            _disable_sortformer_streaming_mode(diar_model)
    except Exception:
        pass

    if hasattr(diar_model, "eval"):
        diar_model.eval()
    if is_nemotron:
        modules = getattr(diar_model, "sortformer_modules", None)
        if modules is None:
            raise RuntimeError("Nemotron-3-Diarization checkpoint has no sortformer_modules")
        # NVIDIA model card's recommended 30.4-second offline-style configuration.
        modules.chunk_len = 340
        modules.chunk_right_context = 40
        modules.fifo_len = 40
        modules.spkcache_update_period = 300
        check_parameters = getattr(diar_model, "_check_streaming_parameters", None)
        if not callable(check_parameters):
            raise RuntimeError("Installed NeMo version lacks _check_streaming_parameters required by Nemotron")
        check_parameters()
    resolved_device = _resolve_device(device)
    if hasattr(diar_model, "to"):
        diar_model.to(torch.device(resolved_device))
    env_chunk = os.environ.get("SORTFORMER_MAX_CHUNK_SECONDS") or os.environ.get("MAX_CHUNK_SECONDS")
    chunk_dur = max_chunk_duration if max_chunk_duration is not None else env_chunk
    if chunk_dur is None:
        chunk_dur = 120.0
    return SortformerDiarizationAdapter(diar_model, max_chunk_duration=chunk_dur)




def _load_diarizen_pipeline(model: str, device: str = "cuda") -> DiariZenDiarizationAdapter:
    enforce_offline_mode()
    local_target, _ = resolve_local_model_path(
        model, env_var="DIARIZEN_MODEL_PATH", default_subpath="diarizen-wavlm-large-s80-md"
    )
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "DiariZen diarization requires the BUTSpeechFIT/DiariZen package and "
            "its pyannote-compatible environment. Install the diarizen profile in "
            "a separate environment before using BUT-FIT/diarizen-wavlm-large-s80-md."
        ) from exc

    target_str = str(local_target)
    try:
        pipeline = DiariZenPipeline.from_pretrained(target_str, local_files_only=True)
    except TypeError:
        pipeline = DiariZenPipeline.from_pretrained(target_str)

    resolved_device = _resolve_device(device)
    if hasattr(pipeline, "to"):
        pipeline.to(torch.device(resolved_device))
    return DiariZenDiarizationAdapter(pipeline)


def _segments_from_annotation(output: Any) -> list[dict]:
    segments = []
    diarization = output.speaker_diarization if hasattr(output, "speaker_diarization") else output
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {"speaker": str(speaker), "start": float(turn.start), "end": float(turn.end)}
        )
    segments.sort(key=lambda x: x["start"])
    return segments


def _segments_from_sortformer_output(output: Any) -> list[dict]:
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], (list, tuple)):
        output = output[0]
    segments = []
    for item in output:
        parsed = _parse_sortformer_segment(item)
        if parsed is not None:
            segments.append(parsed)
    segments.sort(key=lambda x: x["start"])
    return segments


def _parse_sortformer_segment(item: Any) -> dict | None:
    if isinstance(item, dict):
        start = item.get("start", item.get("begin", item.get("start_time")))
        end = item.get("end", item.get("stop", item.get("end_time")))
        speaker = item.get("speaker", item.get("label", item.get("speaker_id")))
        if start is not None and end is not None and speaker is not None:
            return {"speaker": str(speaker), "start": float(start), "end": float(end)}
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {"speaker": str(item[2]), "start": float(item[0]), "end": float(item[1])}
    if isinstance(item, str):
        parts = re.split(r"[\s,]+", item.strip())
        numbers = []
        speaker = None
        for part in parts:
            try:
                numbers.append(float(part))
            except ValueError:
                if part:
                    speaker = part
        if len(numbers) >= 2:
            if speaker is None and len(numbers) >= 3:
                speaker = str(int(numbers[2]))
            return {
                "speaker": str(speaker or "SPEAKER_00"),
                "start": numbers[0],
                "end": numbers[1],
            }
    return None


class EmbeddingExtractor(Protocol):
    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...


class SpeechBrainEmbeddingExtractor:
    """Public, versioned speaker embedding API used to link diarization chunks."""

    model_id = "speechbrain/spkrec-ecapa-voxceleb"
    sample_rate = 16000
    minimum_samples = 8000

    def __init__(self, device: str) -> None:
        enforce_offline_mode()
        EncoderClassifier = _load_encoder_classifier()
        self.device = device
        local_target, _ = resolve_local_model_path(
            self.model_id, env_var="SPEECHBRAIN_MODEL_PATH", default_subpath="spkrec-ecapa-voxceleb"
        )
        offline = is_offline_mode()
        self.model = EncoderClassifier.from_hparams(
            source=str(local_target),
            savedir=str(local_target) if Path(local_target).is_dir() else None,
            run_opts={"device": self.device},
            local_files_only=offline,
        )

    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        prepared = wav.detach().cpu().float()
        if prepared.ndim > 1:
            prepared = prepared.mean(dim=0)
        if sample_rate != self.sample_rate:
            import torchaudio.functional as F_audio

            prepared = F_audio.resample(
                prepared.unsqueeze(0), sample_rate, self.sample_rate,
            ).squeeze(0)
        if prepared.numel() < self.minimum_samples:
            prepared = F.pad(prepared, (0, self.minimum_samples - prepared.numel()))
        with torch.inference_mode():
            embedding = self.model.encode_batch(prepared.unsqueeze(0).to(self.device))
        emb = embedding.detach().cpu().reshape(-1).float()
        # L2 normalization
        return emb / (torch.norm(emb, p=2) + 1e-8)


class GlobalSpeakerLinker:
    """Assign chunk-local labels to the participants in a DuplexChat recording."""

    def __init__(
        self, extractor: EmbeddingExtractor, similarity_threshold: float = 0.7, expected_speakers: int = 2,
    ) -> None:
        self.extractor = extractor
        self.similarity_threshold = similarity_threshold
        self.expected_speakers = expected_speakers
        self._embeddings: dict[str, list[torch.Tensor]] = {}
        self._next_speaker_index = 0
        self._embedded_labels = 0
        self._matched_labels = 0
        self._new_labels = 0
        self._forced_matches = 0

    def link(
        self, local_segments: list[dict], chunk_waveform: torch.Tensor, sample_rate: int,
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        assigned_in_chunk: set[str] = set()
        for speaker in sorted({segment["speaker"] for segment in local_segments}):
            speaker_segments = [segment for segment in local_segments if segment["speaker"] == speaker]
            # Best practice: Filter out short noise segments (<0.5s) when enough audio is available
            long_segments = [seg for seg in speaker_segments if seg["end"] - seg["start"] >= 0.5]
            target_segments = long_segments if long_segments else [seg for seg in speaker_segments if seg["end"] > seg["start"]]

            embeddings = [
                self.extractor.extract(_slice_segment(chunk_waveform, segment, sample_rate), sample_rate)
                for segment in target_segments
            ]
            if not embeddings:
                raise RuntimeError(
                    "Could not extract a speaker embedding for diarization label "
                    f"{speaker}; refusing to create an unstable fallback label."
                )
            self._embedded_labels += 1
            local_embedding = torch.stack(embeddings).mean(dim=0)
            local_embedding = local_embedding / (torch.norm(local_embedding, p=2) + 1e-8)
            global_speaker = self._best_match(local_embedding, excluded=assigned_in_chunk)
            if global_speaker is None:
                if self.expected_speakers is None or len(self._embeddings) < self.expected_speakers:
                    global_speaker = f"SPEAKER_{self._next_speaker_index:02d}"
                    self._next_speaker_index += 1
                    self._embeddings[global_speaker] = []
                    self._new_labels += 1
                else:
                    # A third local label is diarization fragmentation: force match to nearest speaker.
                    global_speaker = self._best_match(
                        local_embedding, excluded=set(), require_threshold=False,
                    )
                    self._forced_matches += 1
            else:
                self._matched_labels += 1
            self._embeddings[global_speaker].append(local_embedding)
            mapping[speaker] = global_speaker
            assigned_in_chunk.add(global_speaker)
        return mapping

    def diagnostics(self) -> dict:
        return {
            "method": "speechbrain_ecapa_cosine",
            "similarity_threshold": self.similarity_threshold,
            "expected_speakers": self.expected_speakers,
            "embedding_labels": self._embedded_labels,
            "global_speakers": len(self._embeddings),
            "matched_labels": self._matched_labels,
            "new_labels": self._new_labels,
            "forced_matches": self._forced_matches,
        }

    def _best_match(
        self, embedding: torch.Tensor, excluded: set[str], require_threshold: bool = True,
    ) -> str | None:
        best_speaker: str | None = None
        best_score = -1.0
        for speaker, embeddings in self._embeddings.items():
            if speaker in excluded:
                continue
            centroid = torch.stack(embeddings).mean(dim=0)
            score = float(F.cosine_similarity(embedding.reshape(-1), centroid.reshape(-1), dim=0))
            if score > best_score:
                best_speaker, best_score = speaker, score
        if best_speaker is None:
            return None
        return best_speaker if not require_threshold or best_score >= self.similarity_threshold else None


def _slice_segment(waveform: torch.Tensor, segment: dict, sample_rate: int) -> torch.Tensor:
    start = max(0, int(round(segment["start"] * sample_rate)))
    end = min(waveform.shape[-1], int(round(segment["end"] * sample_rate)))
    return waveform[:, start:end]


def _load_encoder_classifier():
    from core.compat import ensure_runtime_compat
    ensure_runtime_compat()
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ModuleNotFoundError as exc:
        if exc.name is None or not (exc.name == "speechbrain" or exc.name.startswith("speechbrain.")):
            raise
        raise RuntimeError(
            "DuplexChat chunked pyannote diarization requires SpeechBrain ECAPA. "
            "Install the DuplexChat runtime dependency with `pip install speechbrain` or `pip install -r requirements.txt`."
        ) from exc
    return EncoderClassifier


def run_diarization(
    pipeline: "Pipeline | FileDiarizationAdapter",
    wav_path: Path,
    max_chunk_dur: float | str | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
    diagnostics: dict | None = None,
) -> list[dict]:
    """
    Chạy diarization bằng cách dùng VAD để cắt audio thành các chunk <= max_chunk_dur,
    sau đó so sánh embedding để gán nhãn speaker globally (giúp tránh OOM).
    Nếu max_chunk_dur là 'full' hoặc None, chạy toàn bộ audio không cắt chunk.
    """
    if isinstance(pipeline, FileDiarizationAdapter):
        if progress_callback is not None:
            progress_callback("start", 1)
        segments = pipeline.diarize_file(wav_path)
        if diagnostics is not None:
            diagnostics.update({"method": "backend_native_labels", "global_speakers": len({s["speaker"] for s in segments})})
        if progress_callback is not None:
            progress_callback("advance", 1)
            progress_callback("close", 0)
        return segments

    waveform, source_sample_rate = load_wav_tensor(wav_path)
    is_full = (
        max_chunk_dur is None
        or str(max_chunk_dur).strip().lower() in ("full", "none", "inf")
        or max_chunk_dur == float("inf")
    )
    if is_full:
        if progress_callback is not None:
            progress_callback("start", 1)
        output = pipeline({"waveform": waveform, "sample_rate": source_sample_rate})
        segments = _segments_from_annotation(output)
        if diagnostics is not None:
            diagnostics.update({"method": "whole_episode", "global_speakers": len({s["speaker"] for s in segments})})
        if progress_callback is not None:
            progress_callback("advance", 1)
            progress_callback("close", 0)
        return segments
    max_chunk_dur = float(max_chunk_dur)

    sample_rate = DIARIZATION_SAMPLE_RATE
    if source_sample_rate != sample_rate:
        target_length = max(1, round(waveform.shape[-1] * sample_rate / source_sample_rate))
        waveform = F.interpolate(
            waveform.unsqueeze(0), size=target_length, mode="linear", align_corners=False,
        ).squeeze(0)
    dur_sec = waveform.shape[1] / sample_rate
    
    # 1. Dùng Silero VAD để lấy các phân đoạn có giọng nói
    try:
        from core.model_utils import load_local_silero_vad
        vad_model, get_speech_timestamps = load_local_silero_vad()
        # Silero VAD yêu cầu tensor 1D và sample_rate=16000
        speech_ts = get_speech_timestamps(waveform[0], vad_model, sampling_rate=sample_rate)
        vad_segments = [(ts["start"]/sample_rate, ts["end"]/sample_rate) for ts in speech_ts]
    except Exception as e:
        import logging
        logging.getLogger("duplexchat").warning("Silero VAD failed (%s); falling back to full audio", e)
        vad_segments = [(0.0, dur_sec)]
        
    if not vad_segments:
        return []

    # 2. Gom nhóm các segment VAD thành các chunk sao cho (end_N - start_1) <= max_chunk_dur
    chunks = []
    curr_start = vad_segments[0][0]
    curr_end = vad_segments[0][1]
    
    for st, en in vad_segments[1:]:
        if en - curr_start <= max_chunk_dur:
            curr_end = en
        else:
            chunks.append((curr_start, curr_end))
            curr_start = st
            curr_end = en
    chunks.append((curr_start, curr_end))
    if progress_callback is not None:
        progress_callback("start", len(chunks))
    if diagnostics is not None:
        diagnostics.update({"chunk_count": len(chunks), "chunks": []})
    
    # 3. Chạy diarization từng chunk và link nhãn local qua public ECAPA API.
    all_segments = []
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
    linker = GlobalSpeakerLinker(SpeechBrainEmbeddingExtractor(embedding_device))
    
    for c_start, c_end in chunks:
        # Mở rộng nhẹ chunk để không cắt gắt
        pad = 0.5
        s_pad = max(0.0, c_start - pad)
        e_pad = min(dur_sec, c_end + pad)
        
        s_idx = int(s_pad * sample_rate)
        e_idx = int(e_pad * sample_rate)
        chunk_wav = waveform[:, s_idx:e_idx]
        
        try:
            output = pipeline({"waveform": chunk_wav, "sample_rate": sample_rate})
        except Exception as e:
            import logging
            logging.getLogger("duplexchat").warning("Diarization failed on chunk %s-%s: %s", s_pad, e_pad, e)
            if progress_callback is not None:
                progress_callback("advance", 1)
            continue
            
        local_segs = []
        iterator = (
            output.speaker_diarization.itertracks(yield_label=True)
            if hasattr(output, "speaker_diarization")
            else output.itertracks(yield_label=True)
        )

        for turn, _, speaker in iterator:
            local_segs.append({
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            })
            
        if not local_segs:
            if progress_callback is not None:
                progress_callback("advance", 1)
            continue
            
        local_to_global = linker.link(local_segs, chunk_wav, sample_rate)
        if diagnostics is not None:
            diagnostics["chunks"].append(
                {
                    "start": s_pad,
                    "end": e_pad,
                    "local_labels": sorted(local_to_global),
                    "global_labels": sorted(set(local_to_global.values())),
                }
            )
        # Cập nhật thời gian thực tế và append
        for s in local_segs:
            all_segments.append({
                "speaker": local_to_global[s["speaker"]],
                "start": s["start"] + s_pad,
                "end": s["end"] + s_pad,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback("advance", 1)

    all_segments.sort(key=lambda x: x["start"])
    if diagnostics is not None:
        diagnostics.update(linker.diagnostics())
    if progress_callback is not None:
        progress_callback("close", 0)
    return all_segments
