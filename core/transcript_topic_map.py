"""Create a two-dimensional topic-distribution view from dialogue transcripts."""
from __future__ import annotations

import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Transcript:
    dialogue_id: str
    path: Path
    text: str


def load_transcripts(input_dir: Path) -> list[Transcript]:
    """Read one UTF-8 TXT transcript per dialogue, using each stem as its ID."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory does not exist: {input_dir}")

    transcripts = []
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            warnings.warn(f"Skipping empty transcript: {path}", stacklevel=2)
            continue
        transcripts.append(Transcript(dialogue_id=path.stem, path=path, text=text))

    if not transcripts:
        raise ValueError(f"No non-empty .txt transcripts found in {input_dir}")
    return transcripts


def make_token_chunks(text: str, tokenizer, max_seq_length: int) -> list[tuple[list[int], int]]:
    """Pack sentence token IDs into windows that fit the model's sequence limit."""
    max_body_tokens = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if max_body_tokens < 1:
        raise ValueError(f"Model sequence limit is too small: {max_seq_length}")

    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if part.strip()]
    chunks: list[tuple[list[int], int]] = []
    current: list[int] = []

    def flush() -> None:
        if current:
            chunks.append((current.copy(), len(current)))
            current.clear()

    for sentence in sentences:
        token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if not token_ids:
            continue
        if len(token_ids) > max_body_tokens:
            flush()
            for start in range(0, len(token_ids), max_body_tokens):
                chunk = token_ids[start:start + max_body_tokens]
                chunks.append((chunk, len(chunk)))
        elif len(current) + len(token_ids) <= max_body_tokens:
            current.extend(token_ids)
        else:
            flush()
            current.extend(token_ids)
    flush()
    return chunks


def _device_name(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _embed_transcripts(
    transcripts: list[Transcript], model_dir: Path, device: str = "auto", batch_size: int = 16,
) -> np.ndarray:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local Vietnamese bi-encoder model directory does not exist: {model_dir}")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from pyvi import ViTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Topic-map imports failed: "
            f"{error}. Install requirements-topic-map.txt in the active DuplexChat environment "
            "and keep its configured Torch/Transformers runtime compatible."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    device = _device_name(torch, device)
    model.to(device)

    settings_path = model_dir / "sentence_bert_config.json"
    max_seq_length = 256
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        max_seq_length = int(settings.get("max_seq_length", max_seq_length))

    encoded_chunks: list[dict] = []
    chunk_owners: list[int] = []
    chunk_weights: list[int] = []
    for owner, transcript in enumerate(transcripts):
        segmented = ViTokenizer.tokenize(transcript.text)
        chunks = make_token_chunks(segmented, tokenizer, max_seq_length)
        if not chunks:
            raise ValueError(f"Transcript has no model tokens after word segmentation: {transcript.path}")
        for token_ids, weight in chunks:
            encoded_chunks.append(tokenizer.prepare_for_model(
                token_ids, add_special_tokens=True, return_attention_mask=True,
            ))
            chunk_owners.append(owner)
            chunk_weights.append(weight)

    chunk_vectors = []
    for start in range(0, len(encoded_chunks), batch_size):
        batch = tokenizer.pad(
            encoded_chunks[start:start + batch_size], padding=True, return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            token_embeddings = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        chunk_vectors.extend(pooled.cpu().numpy())

    vectors = np.asarray(chunk_vectors)
    vectors_by_document = [[] for _ in transcripts]
    weights_by_document = [[] for _ in transcripts]
    for vector, owner, weight in zip(vectors, chunk_owners, chunk_weights, strict=True):
        vectors_by_document[owner].append(vector)
        weights_by_document[owner].append(weight)
    documents = [
        np.average(vectors, axis=0, weights=np.asarray(weights, dtype=np.float64))
        for vectors, weights in zip(vectors_by_document, weights_by_document, strict=True)
    ]
    return np.asarray(documents, dtype=np.float32)


def create_topic_map(
    input_dir: Path,
    model_dir: Path,
    output_dir: Path,
    device: str = "auto",
) -> tuple[Path, Path]:
    """Embed the transcript corpus and write a reproducible t-SNE PNG and CSV."""
    transcripts = load_transcripts(input_dir)
    if len(transcripts) < 3:
        raise ValueError(
            f"t-SNE needs at least 3 non-empty dialogue transcripts; found {len(transcripts)}"
        )
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError("Topic mapping requires scikit-learn") from error
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Topic mapping requires matplotlib") from error

    embeddings = _embed_transcripts(transcripts, model_dir, device=device)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(transcripts):
        raise ValueError(f"Expected one embedding per transcript, got shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Transcript embeddings contain non-finite values")
    perplexity = min(30.0, max(1.0, (len(transcripts) - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=42,
        n_jobs=1,
    ).fit_transform(embeddings)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "transcript_tsne.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dialogue_id", "source_file", "x", "y"])
        writer.writerows(
            (transcript.dialogue_id, str(transcript.path), float(point[0]), float(point[1]))
            for transcript, point in zip(transcripts, coordinates, strict=True)
        )

    png_path = output_dir / "transcript_tsne.png"
    figure, axis = plt.subplots(figsize=(12, 9))
    axis.scatter(coordinates[:, 0], coordinates[:, 1], s=18, alpha=0.75)
    axis.set_title(f"Dialogue transcript distribution (t-SNE, n={len(transcripts)})")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    figure.tight_layout()
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return png_path, csv_path
