from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat.diarization_backend import (
    GlobalSpeakerLinker,
    run_diarization,
    FileDiarizationAdapter,
)

class MockExtractor:
    def __init__(self):
        self.vectors = {
            "A": torch.tensor([1.0, 0.0, 0.0]),
            "B": torch.tensor([0.0, 1.0, 0.0]),
        }

    def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Simple mock: return unit vector depending on mock key or signal
        if wav.shape[-1] > 0 and float(wav.sum()) > 0:
            return torch.tensor([0.0, 1.0, 0.0])
        return torch.tensor([1.0, 0.0, 0.0])


def test_global_speaker_linker_links_consistent_speakers():
    linker = GlobalSpeakerLinker(MockExtractor(), similarity_threshold=0.7, expected_speakers=2)
    chunk_wav = torch.zeros(1, 16000 * 10)
    
    # Chunk 1: local speaker 'spk_1' maps to SPEAKER_00, 'spk_2' maps to SPEAKER_01
    segs1 = [
        {"speaker": "spk_1", "start": 0.0, "end": 2.0},
        {"speaker": "spk_2", "start": 2.5, "end": 5.0},
    ]
    # Make segment 2 positive audio sum so MockExtractor yields different vector
    chunk_wav[0, 40000:80000] = 1.0

    mapping1 = linker.link(segs1, chunk_wav, 16000)
    assert mapping1["spk_1"] == "SPEAKER_00"
    assert mapping1["spk_2"] == "SPEAKER_01"

    # Chunk 2: local speaker 'spk_x' has same embedding as 'spk_1' -> SPEAKER_00
    segs2 = [
        {"speaker": "spk_x", "start": 0.0, "end": 2.0},
    ]
    chunk_wav_2 = torch.zeros(1, 16000 * 5)
    mapping2 = linker.link(segs2, chunk_wav_2, 16000)
    assert mapping2["spk_x"] == "SPEAKER_00"


def test_global_speaker_linker_filters_short_noise_segments():
    class NoiseSensitiveExtractor:
        def __init__(self):
            self.calls = []

        def extract(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
            self.calls.append(wav.shape[-1] / sample_rate)
            return torch.tensor([1.0, 0.0, 0.0])

    extractor = NoiseSensitiveExtractor()
    linker = GlobalSpeakerLinker(extractor, similarity_threshold=0.7, expected_speakers=2)
    chunk_wav = torch.zeros(1, 16000 * 10)

    # Has 1 short segment (0.1s) and 1 long segment (2.0s)
    segs = [
        {"speaker": "A", "start": 0.0, "end": 0.1},
        {"speaker": "A", "start": 1.0, "end": 3.0},
    ]
    linker.link(segs, chunk_wav, 16000)
    # Should only call extract on the long segment (2.0s), skipping the 0.1s micro-noise
    assert extractor.calls == [2.0]


class DummyNativeAdapter(FileDiarizationAdapter):
    def diarize_file(self, wav_path: Path) -> list[dict]:
        return [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0},
        ]


def test_run_diarization_adapter_flow(tmp_path):
    adapter = DummyNativeAdapter()
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"dummy")

    results = run_diarization(adapter, wav_file)
    assert len(results) == 2
    assert results[0]["speaker"] == "SPEAKER_00"
    assert results[1]["speaker"] == "SPEAKER_01"


def test_sortformer_adapter_full_chunk_mode(tmp_path):
    import soundfile as sf
    import numpy as np
    from duplexchat.diarization_backend import SortformerDiarizationAdapter

    # Create a 30-second test wav
    wav_path = tmp_path / "long_audio.wav"
    data = np.zeros(16000 * 30, dtype=np.float32)
    sf.write(str(wav_path), data, 16000)

    class MockSortformerModel:
        def __init__(self):
            self.calls = []
        def diarize(self, audio, batch_size=1):
            self.calls.append(audio)
            return ["0.0 5.0 speaker_0", "5.0 10.0 speaker_1"]

    # 1. Full mode: should have max_chunk_duration = inf and call diarize once on the full file
    mock_model = MockSortformerModel()
    adapter_full = SortformerDiarizationAdapter(mock_model, max_chunk_duration="full")
    assert adapter_full.max_chunk_duration == float("inf")

    res = adapter_full.diarize_file(wav_path)
    assert len(mock_model.calls) == 1
    assert mock_model.calls[0] == str(wav_path)
    assert len(res) == 2

    # 2. Case-insensitive "FULL" or None or 0
    adapter_upper = SortformerDiarizationAdapter(mock_model, max_chunk_duration="FULL")
    assert adapter_upper.max_chunk_duration == float("inf")

    adapter_none = SortformerDiarizationAdapter(mock_model, max_chunk_duration=None)
    assert adapter_none.max_chunk_duration == float("inf")

