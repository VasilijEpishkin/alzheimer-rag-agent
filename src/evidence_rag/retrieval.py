import gc
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from evidence_rag.models import Document


@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, retriever: bm25s.BM25) -> None:
        self.retriever = retriever

    @classmethod
    def build(cls, documents: Sequence[Document]) -> "BM25Index":
        corpus = [{"doc_id": document.doc_id} for document in documents]
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(bm25s.tokenize([document.retrieval_text for document in documents]))
        return cls(retriever)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(path, corpus=self.retriever.corpus)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls(bm25s.BM25.load(path, load_corpus=True, mmap=True))

    def search(self, query: str, k: int) -> list[SearchResult]:
        documents, scores = self.retriever.retrieve(
            bm25s.tokenize([query]), k=min(k, len(self.retriever.corpus))
        )
        return [
            SearchResult(
                str(document["doc_id"] if isinstance(document, Mapping) else document),
                float(score),
            )
            for document, score in zip(documents[0], scores[0], strict=True)
        ]


class DenseIndex:
    def __init__(self, path: Path) -> None:
        metadata = json.loads((path / "metadata.json").read_text())
        self.doc_ids: list[str] = metadata["doc_ids"]
        self.dimension = int(metadata["dimension"])
        self.embeddings = np.load(path / "embeddings.npy", mmap_mode="r")

    @staticmethod
    def _device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        path: Path,
        model_name: str,
        batch_size: int = 32,
    ) -> "DenseIndex":
        import torch
        from transformers import AutoModel, AutoTokenizer

        path.mkdir(parents=True, exist_ok=True)
        device = cls._device()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        if device == "cuda":
            model.half()
        dimension = int(model.config.hidden_size)
        embeddings = np.lib.format.open_memmap(
            path / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(documents), dimension)
        )
        with torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                inputs = tokenizer(
                    [document.title for document in batch],
                    [document.text for document in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                vectors = model(**inputs).last_hidden_state[:, 0, :]
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                embeddings[start : start + len(batch)] = vectors.float().cpu().numpy()
                embeddings.flush()
                del inputs, vectors
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "doc_ids": [document.doc_id for document in documents],
                    "dimension": dimension,
                    "model": model_name,
                }
            )
        )
        del embeddings, model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return cls(path)

    def search_vector(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        scores = self.embeddings @ query_vector.astype(np.float32, copy=False)
        count = min(k, len(self.doc_ids))
        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [SearchResult(self.doc_ids[index], float(scores[index])) for index in indices]


class MedCPTQueryEncoder:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = DenseIndex._device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> Iterable[np.ndarray]:
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                vectors = self.model(**inputs).last_hidden_state[:, 0, :]
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
                yield from vectors.float().cpu().numpy()
                del inputs, vectors

    def close(self) -> None:
        del self.model, self.tokenizer
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
