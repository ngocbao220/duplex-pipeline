from __future__ import annotations

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "duplexchat" / "src"))

from duplexchat.music import DemucsMusicFilter, NoOpMusicFilter, load_music_filter


def test_load_music_filter_disabled_returns_noop():
    filter_inst = load_music_filter(enabled=False)
    assert isinstance(filter_inst, NoOpMusicFilter)
    wav = torch.ones(1, 16000)
    out = filter_inst.filter_waveform(wav, 16000)
    assert torch.equal(wav, out)


def test_noop_music_filter_preserves_waveform():
    noop = NoOpMusicFilter()
    wav = torch.randn(2, 8000)
    out = noop.filter_waveform(wav, 16000)
    assert torch.equal(wav, out)


class MockDemucsModel:
    samplerate = 44100

    def to(self, device):
        return self

    def eval(self):
        return self


def test_demucs_music_filter_mock(monkeypatch):
    class DummyFilter(DemucsMusicFilter):
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = MockDemucsModel()

        def apply_model(self, model, batch, **kwargs):
            # batch is (1, 2, T), return sources tensor (1, 4, 2, T) where index 3 is vocals
            sources = torch.zeros(1, 4, 2, batch.shape[-1])
            sources[0, 3] = batch[0] * 0.9
            return sources

    filt = DummyFilter()
    wav = torch.ones(1, 44100)
    cleaned = filt.filter_waveform(wav, 44100)
    assert cleaned.shape == (1, 44100)
    assert torch.allclose(cleaned, torch.ones(1, 44100) * 0.9, atol=1e-2)
