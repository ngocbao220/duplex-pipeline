"""Transcribe the final stereo waveform and retain speaker/coverage provenance."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from core.model_utils import load_local_silero_vad
from core.outputs import turns_from_diarization

from .aligner_engine import WhisperXAligner
from .channel_resolver import duplexchat_mapping, sommelier_mapping
from .region_extractor import asr_regions, uncovered_speech
from .zipformer_engine import ZipformerEngine


def _to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16_000:
        return np.asarray(samples, dtype=np.float32)
    return np.asarray(resample_poly(samples, 16_000, sample_rate), dtype=np.float32)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_fingerprints(stereo: Path, dialogue_dir: Path, pipeline_name: str) -> dict[str, str]:
    index = stereo.stem.removeprefix("stereo_")
    paths = {"stereo_sha256": stereo, "dialogue_sha256": dialogue_dir / f"dialogue_{index}.json"}
    if pipeline_name == "sommelier":
        paths["channel_map_sha256"] = stereo.with_suffix(".channels.json")
    elif pipeline_name == "duplexchat":
        paths["mixture_sha256"] = dialogue_dir / f"dialogue_{index}.wav"
    else:
        raise ValueError(f"Unsupported ASR pipeline: {pipeline_name}")
    return {name: _sha256(path) for name, path in paths.items()}


def speaker_turns_for_asr(metadata: dict) -> tuple[list[dict], str]:
    turns = metadata.get("speaker_turns") or metadata.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Dialogue has no speaker turns")
    source = "speaker_turns" if metadata.get("speaker_turns") else "turns"
    if metadata.get("segments"):
        detailed = turns_from_diarization(metadata["segments"])
        if detailed != turns:
            return detailed, "segments_legacy_recovery"
    return turns, source


class ASRPipeline:
    def __init__(self, cfg: dict, offline: bool, device: str, pipeline_name: str):
        self.cfg = cfg
        self.offline = offline
        if offline:
            os.environ["MODE"] = "sever"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        self.asr = ZipformerEngine(cfg["zipformer"], offline)
        self.aligner = WhisperXAligner(cfg["aligner"], offline, device)
        self.vad_model, self.vad_timestamps = load_local_silero_vad(device="cpu")
        self.extractor = None
        if pipeline_name == "duplexchat":
            from pipeline.duplexchat.src.duplexchat.diarization_backend import SpeechBrainEmbeddingExtractor

            self.extractor = SpeechBrainEmbeddingExtractor(device)
        self.config_signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    def _speech(self, audio: np.ndarray) -> list[tuple[float, float]]:
        samples = _to_16k(audio, 24_000)
        timestamps = self.vad_timestamps(
            torch.from_numpy(samples), self.vad_model, sampling_rate=16_000,
            min_speech_duration_ms=100,
        )
        return [(item["start"] / 16_000, item["end"] / 16_000) for item in timestamps]

    def process(self, stereo: Path, dialogue_dir: Path, pipeline_name: str) -> dict:
        match = re.fullmatch(r"stereo_(\d+)\.wav", stereo.name)
        if match is None:
            raise ValueError(f"Unexpected stereo filename: {stereo}")
        index = match.group(1)
        metadata_path = dialogue_dir / f"dialogue_{index}.json"
        mixture_path = dialogue_dir / f"dialogue_{index}.wav"
        fingerprints = input_fingerprints(stereo, dialogue_dir, pipeline_name)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            turns, turn_source = speaker_turns_for_asr(metadata)
        except ValueError as exc:
            raise ValueError(f"{exc}: {metadata_path}") from exc
        speakers = {str(turn["speaker"]) for turn in turns}
        if len(speakers) != 2:
            raise ValueError(f"Expected two speakers in {metadata_path}, got {speakers}")
        info = sf.info(stereo)
        if info.channels != 2 or info.samplerate != 24_000 or info.frames <= 0:
            raise ValueError(f"Expected nonempty 24 kHz stereo WAV: {stereo}")
        audio, _ = sf.read(stereo, dtype="float32", always_2d=True)
        duration = len(audio) / 24_000
        if pipeline_name == "sommelier":
            mapping = sommelier_mapping(stereo, speakers)
        elif pipeline_name == "duplexchat":
            mapping = duplexchat_mapping(
                audio, 24_000, mixture_path, turns, self.extractor,
                min_score=float(self.cfg["channel_matching"]["min_score"]),
                min_margin=float(self.cfg["channel_matching"]["min_margin"]),
            )
        else:
            raise ValueError(f"Unsupported ASR pipeline: {pipeline_name}")

        channels = []
        merged_timeline = []
        all_uncovered = []
        vad_seconds = 0.0
        uncovered_seconds = 0.0
        for channel, speaker in enumerate(mapping["channel_speakers"]):
            speaker_turns = [(float(t["start"]), float(t["end"])) for t in turns if t["speaker"] == speaker]
            speech = self._speech(audio[:, channel])
            coverage, uncovered = uncovered_speech(speech, speaker_turns)
            vad_seconds += sum(end - start for start, end in speech)
            uncovered_seconds += sum(end - start for start, end in uncovered)
            uncovered_rows = [{"channel": channel, "speaker": speaker, "start": start, "end": end} for start, end in uncovered]
            all_uncovered.extend(uncovered_rows)
            regions = []
            for start, end in asr_regions(turns, speaker, duration,
                                           gap=float(self.cfg["regions"]["merge_gap_seconds"]),
                                           pad=float(self.cfg["regions"]["pad_seconds"])):
                first = round(start * 24_000)
                last = round(end * 24_000)
                region_start = first / 24_000
                region_end = last / 24_000
                samples = _to_16k(audio[first:last, channel], 24_000)
                raw = self.asr.transcribe(samples)
                alignment_text, local_words, status, alignment_error = self.aligner.align(raw, samples)
                words = []
                for word in local_words:
                    global_word = {**word, "start": region_start + word["start"] if word["aligned"] else None,
                                   "end": region_start + word["end"] if word["aligned"] else None}
                    words.append(global_word)
                    if word["aligned"]:
                        merged_timeline.append({**global_word, "channel": channel, "speaker": speaker})
                regions.append({"start": region_start, "end": region_end,
                                "asr_text_raw": raw, "alignment_text": alignment_text,
                                "alignment_status": status,
                                "alignment_error": alignment_error,
                                "aligned_word_ratio": sum(word["aligned"] for word in words) / len(words) if words else None,
                                "words": words})
            channels.append({"channel": channel, "speaker": speaker, "regions": regions,
                             "asr_text_raw": " ".join(region["asr_text_raw"] for region in regions if region["asr_text_raw"]),
                             "alignment_text": " ".join(region["alignment_text"] for region in regions if region["alignment_text"]),
                             "words": [word for region in regions for word in region["words"]],
                             "coverage_ratio": coverage, "uncovered_regions": uncovered_rows})
        merged_timeline.sort(key=lambda row: (row["start"], row["end"], row["channel"]))
        if input_fingerprints(stereo, dialogue_dir, pipeline_name) != fingerprints:
            raise RuntimeError(f"ASR inputs changed during inference: {stereo}")
        return {"schema_version": 1, "stereo_file": stereo.name, "duration": duration,
                "speaker_turns": turns,
                "provenance": {**fingerprints,
                               "config_sha256": self.config_signature,
                               "turn_source": turn_source,
                               "asr": {"model": self.asr.model_id, "backend": "sherpa-onnx", "sample_rate": 16_000,
                                       "decoding_method": self.asr.decoding_method,
                                       "max_active_paths": self.asr.max_active_paths},
                               "aligner": {"model": self.aligner.model_id, "backend": "whisperx", "sample_rate": 16_000},
                               "channel_attribution": mapping},
                "channels": channels, "coverage_ratio": 1 - uncovered_seconds / vad_seconds if vad_seconds else None,
                "uncovered_regions": all_uncovered, "merged_timeline": merged_timeline}
