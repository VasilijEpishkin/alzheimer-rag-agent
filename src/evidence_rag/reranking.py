import gc
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evidence_rag.models import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EVIDENCE_LEVELS = {
    "meta-analysis": 5.0,
    "systematic review": 5.0,
    "randomized controlled trial": 4.0,
    "clinical trial": 3.0,
    "observational study": 2.0,
    "case reports": 1.0,
}

FEATURE_NAMES = (
    "bm25_score",
    "dense_score",
    "bm25_reciprocal_rank",
    "dense_reciprocal_rank",
    "cross_encoder_score",
    "title_token_overlap",
    "mesh_token_overlap",
    "evidence_level",
    "year_scaled",
    "evidence_missing",
    "year_missing",
    "mesh_missing",
)
RELEVANCE_FEATURES = FEATURE_NAMES[:6]


@dataclass(frozen=True, slots=True)
class Candidate:
    query_id: str
    doc_id: str
    bm25_score: float
    dense_score: float
    bm25_rank: int
    dense_rank: int
    cross_score: float = 0.0


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _evidence_level(document: Document) -> float:
    normalized = {value.lower() for value in document.publication_types}
    return max((_EVIDENCE_LEVELS.get(value, 0.0) for value in normalized), default=0.0)


def build_features(query: str, document: Document, candidate: Candidate) -> np.ndarray:
    mesh_text = " ".join(document.mesh_headings)
    year = float(document.year - 1950) / 100.0 if document.year is not None else 0.0
    return np.asarray(
        [
            candidate.bm25_score,
            candidate.dense_score,
            1.0 / candidate.bm25_rank if candidate.bm25_rank else 0.0,
            1.0 / candidate.dense_rank if candidate.dense_rank else 0.0,
            candidate.cross_score,
            _overlap(query, document.title),
            _overlap(query, mesh_text),
            _evidence_level(document),
            year,
            float(not document.publication_types),
            float(document.year is None),
            float(not document.mesh_headings),
        ],
        dtype=np.float32,
    )


class MedCPTCrossEncoder:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cpu().eval()

    def score(self, query: str, documents: Sequence[Document], batch_size: int = 16) -> list[float]:
        if batch_size > 16:
            raise ValueError("Cross-encoder batch size must not exceed 16")
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                inputs = self.tokenizer(
                    [query] * len(batch),
                    [document.retrieval_text for document in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = self.model(**inputs).logits.reshape(-1)
                scores.extend(float(value) for value in logits.cpu())
                del inputs, logits
        return scores

    def close(self) -> None:
        del self.model, self.tokenizer
        gc.collect()
