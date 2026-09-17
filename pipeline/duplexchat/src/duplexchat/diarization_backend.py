"""Purpose: Implement DuplexChat diarization model loading and inference.

Inputs: Audio paths, diarization backend/model settings and device.
Outputs: Normalized diarization segment dictionaries.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Protocol

import os
os.environ["MPLBACKEND"] = "Agg"
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

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


class SortformerDiarizationAdapter(FileDiarizationAdapter):
    backend = "sortformer"

    def __init__(self, model: Any):
        self.model = model

    def diarize_file(self, wav_path: Path) -> list[dict]:
        predicted = self.model.diarize(audio=str(wav_path), batch_size=1)
        return _segments_from_sortformer_output(predicted)


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
) -> "Pipeline | FileDiarizationAdapter":
    model = resolve_model_alias(model, DIARIZATION_MODELS) or model
    backend_norm = backend.strip().lower()
    if backend_norm == "auto":
        backend_norm = infer_diarization_backend(model)
    if backend_norm == "sortformer":
        return _load_sortformer_pipeline(model, device)
    if backend_norm == "diarizen":
        return _load_diarizen_pipeline(model, device)
    if backend_norm != "pyannote":
        raise ValueError("Unsupported diarization backend '%s'. Use auto, pyannote, sortformer, or diarizen." % backend)
    return _load_pyannote_pipeline(model, device)


from core.model_utils import (
    assert_local_model_exists,
    enforce_offline_mode,
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


def _load_sortformer_pipeline(model: str, device: str = "cuda") -> SortformerDiarizationAdapter:
    enforce_offline_mode()
    local_target, is_local = resolve_local_model_path(
        model, env_var="SORTFORMER_MODEL_PATH", default_subpath="diar_streaming_sortformer_4spk-v2.1"
    )
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Sortformer diarization requires NVIDIA NeMo. Install the sortformer "
            "dependency profile before using an NVIDIA Sortformer model."
        ) from exc

    target_path = Path(local_target)
    if not is_local and not target_path.exists():
        assert_local_model_exists(
            local_target,
            model_name_hint="NVIDIA Sortformer Diarization Model (e.g. diar_streaming_sortformer_4spk-v2.1.nemo)"
        )

    target_str = str(local_target)
    if target_path.is_file() and target_str.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(target_str)
    elif target_path.is_dir():
        nemo_files = list(target_path.glob("*.nemo"))
        if nemo_files:
            diar_model = SortformerEncLabelModel.restore_from(str(nemo_files[0]))
        else:
            try:
                diar_model = SortformerEncLabelModel.restore_from(target_str)
            except Exception:
                diar_model = SortformerEncLabelModel.from_pretrained(target_str)
    else:
        try:
            diar_model = SortformerEncLabelModel.restore_from(target_str)
        except Exception:
            diar_model = SortformerEncLabelModel.from_pretrained(target_str)

    if hasattr(diar_model, "eval"):
        diar_model.eval()
    resolved_device = _resolve_device(device)
    if hasattr(diar_model, "to"):
        diar_model.to(torch.device(resolved_device))
    return SortformerDiarizationAdapter(diar_model)


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
        self.model = EncoderClassifier.from_hparams(
            source=str(local_target),
            savedir=str(local_target) if Path(local_target).is_dir() else None,
            run_opts={"device": self.device},
            local_files_only=True,
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
    max_chunk_dur: float | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
    diagnostics: dict | None = None,
) -> list[dict]:
    """
    Chạy diarization bằng cách dùng VAD để cắt audio thành các chunk <= max_chunk_dur,
    sau đó so sánh embedding để gán nhãn speaker globally (giúp tránh OOM).
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
    if max_chunk_dur is None:
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
