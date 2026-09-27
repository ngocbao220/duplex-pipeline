"""Sherpa ONNX transducer adapter for the configured Vietnamese Zipformer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


class ZipformerEngine:
    sample_rate = 16_000

    def __init__(self, cfg: dict, offline: bool):
        import sentencepiece as spm
        import sherpa_onnx

        names = {key: str(cfg[key]) for key in ("encoder", "decoder", "joiner", "bpe_model")}
        decoding_method = str(cfg.get("decoding_method", "greedy_search"))
        if decoding_method not in {"greedy_search", "modified_beam_search"}:
            raise ValueError(f"Unsupported Zipformer decoding method: {decoding_method}")
        max_active_paths = int(cfg.get("max_active_paths", 4))
        if max_active_paths < 1:
            raise ValueError("Zipformer max_active_paths must be >= 1")
        if offline:
            root = Path(cfg["local_dir"]).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"Zipformer local directory is missing: {root}")
        else:
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(repo_id=str(cfg["model_id"]), allow_patterns=list(names.values())))
        paths = {key: root / name for key, name in names.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Zipformer model files are missing: {missing}")

        tokenizer = spm.SentencePieceProcessor(model_file=str(paths["bpe_model"]))
        if tokenizer.id_to_piece(0) != "<blk>":
            raise ValueError("Zipformer bpe.model must assign <blk> token ID 0")
        self._temporary = tempfile.TemporaryDirectory(prefix="duplex-asr-tokens-")
        tokens = Path(self._temporary.name) / "tokens.txt"
        tokens.write_text("".join(f"{tokenizer.id_to_piece(index)} {index}\n" for index in range(tokenizer.vocab_size())), encoding="utf-8")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(paths["encoder"]), decoder=str(paths["decoder"]), joiner=str(paths["joiner"]),
            tokens=str(tokens), num_threads=1, sample_rate=self.sample_rate, feature_dim=80,
            decoding_method=decoding_method, max_active_paths=max_active_paths,
        )
        self.model_id = str(cfg["model_id"])
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, np.asarray(samples, dtype=np.float32))
        self.recognizer.decode_stream(stream)
        return str(stream.result.text).strip()
