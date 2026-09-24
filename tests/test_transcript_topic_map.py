from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core import transcript_topic_map as topic_map


class _WhitespaceTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_load_transcripts_uses_sorted_txt_stems_and_skips_empty_files(tmp_path):
    (tmp_path / "zeta.txt").write_text("Cuộc trò chuyện cuối tuần.", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("Chủ đề giáo dục.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text(" \n", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    with pytest.warns(UserWarning, match="Skipping empty transcript"):
        transcripts = topic_map.load_transcripts(tmp_path)

    assert [item.dialogue_id for item in transcripts] == ["alpha", "zeta"]
    assert transcripts[0].text == "Chủ đề giáo dục."


def test_make_token_chunks_obeys_model_token_limit_and_keeps_all_words():
    tokenizer = _WhitespaceTokenizer()
    chunks = topic_map.make_token_chunks("một hai ba bốn năm sáu bảy tám", tokenizer, max_seq_length=6)

    assert [len(token_ids) for token_ids, _weight in chunks] == [4, 4]
    assert [weight for _token_ids, weight in chunks] == [4, 4]
    assert [word for token_ids, _weight in chunks for word in token_ids] == [
        "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
    ]


def test_embed_transcripts_requires_existing_local_model_before_importing_runtime(tmp_path):
    transcript = topic_map.Transcript("one", tmp_path / "one.txt", "Xin chào.")

    with pytest.raises(FileNotFoundError, match="Local Vietnamese bi-encoder model directory"):
        topic_map._embed_transcripts([transcript], tmp_path / "missing-model")


def test_embed_transcripts_uses_local_transformers_and_averages_long_transcript_chunks(tmp_path, monkeypatch):
    import torch

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 5}), encoding="utf-8",
    )
    calls = []

    class FakeTokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

        def encode(self, text, add_special_tokens=False):
            return [len(word) for word in text.split()]

        def prepare_for_model(self, token_ids, add_special_tokens=True, return_attention_mask=True):
            ids = [101, *token_ids, 102]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        def pad(self, rows, padding=True, return_tensors="pt"):
            max_length = max(len(row["input_ids"]) for row in rows)
            ids = [row["input_ids"] + [0] * (max_length - len(row["input_ids"])) for row in rows]
            masks = [row["attention_mask"] + [0] * (max_length - len(row["attention_mask"])) for row in rows]
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

        def __call__(self, input_ids, attention_mask):
            hidden = torch.stack((input_ids.float(), torch.ones_like(input_ids).float()), dim=-1)
            return types.SimpleNamespace(last_hidden_state=hidden)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kwargs: calls.append(("tokenizer", path, kwargs)) or FakeTokenizer())
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=lambda path, **kwargs: calls.append(("model", path, kwargs)) or FakeModel())
    pyvi = types.ModuleType("pyvi")
    pyvi.ViTokenizer = types.SimpleNamespace(tokenize=lambda text: text)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "pyvi", pyvi)

    embeddings = topic_map._embed_transcripts(
        [topic_map.Transcript("long", tmp_path / "long.txt", "mot hai ba bon nam sau")],
        model_dir,
        device="cpu",
    )

    assert embeddings.shape == (1, 2)
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert {call[0] for call in calls} == {"model", "tokenizer"}


def test_create_topic_map_writes_reproducible_png_and_csv(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    for name, content in (("a", "Nội dung A"), ("b", "Nội dung B"), ("c", "Nội dung C")):
        (input_dir / f"{name}.txt").write_text(content, encoding="utf-8")

    monkeypatch.setattr(
        topic_map,
        "_embed_transcripts",
        lambda transcripts, _model_dir, device="auto": np.asarray([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ], dtype=np.float32),
    )

    png_path, csv_path = topic_map.create_topic_map(input_dir, tmp_path / "local-model", tmp_path / "out")

    assert png_path.is_file() and png_path.stat().st_size > 0
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["dialogue_id"] for row in rows] == ["a", "b", "c"]
    assert all(row["source_file"].endswith(f"{row['dialogue_id']}.txt") for row in rows)
    assert all(np.isfinite([float(row["x"]), float(row["y"])]).all() for row in rows)


def test_create_topic_map_requires_three_nonempty_dialogues(tmp_path, monkeypatch):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("A.", encoding="utf-8")
    (input_dir / "b.txt").write_text("B.", encoding="utf-8")
    monkeypatch.setattr(topic_map, "_embed_transcripts", lambda *_args, **_kwargs: pytest.fail("must fail before model load"))

    with pytest.raises(ValueError, match="at least 3"):
        topic_map.create_topic_map(input_dir, tmp_path / "model", tmp_path / "out")
