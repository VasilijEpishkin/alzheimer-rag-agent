from collections.abc import Iterable
from urllib.parse import urlparse

import ir_datasets

from evidence_rag.models import Benchmark, Document

_DATASETS = {
    "scifact": {"train": "beir/scifact/train", "test": "beir/scifact/test"},
    "nfcorpus": {
        "train": "beir/nfcorpus/train",
        "dev": "beir/nfcorpus/dev",
        "test": "beir/nfcorpus/test",
    },
}


def available_datasets() -> tuple[str, ...]:
    return tuple(_DATASETS)


def load_beir(name: str, split: str = "test") -> Benchmark:
    try:
        dataset_id = _DATASETS[name][split]
    except KeyError as error:
        choices = ", ".join(_DATASETS.get(name, {}))
        if not choices:
            available = ", ".join(_DATASETS)
            raise ValueError(f"Unknown dataset {name!r}; choose from {available}") from error
        raise ValueError(f"Unknown {name!r} split {split!r}; choose from {choices}") from error
    dataset = ir_datasets.load(dataset_id)
    documents = [
        Document(
            doc_id=str(document.doc_id),
            title=getattr(document, "title", "") or "",
            text=getattr(document, "text", "") or "",
            pubmed_id=_pubmed_id(getattr(document, "url", "") or ""),
        )
        for document in dataset.docs_iter()
    ]
    queries = {str(query.query_id): query.text for query in dataset.queries_iter()}
    qrels: dict[str, dict[str, int]] = {query_id: {} for query_id in queries}
    for qrel in dataset.qrels_iter():
        relevance = int(qrel.relevance)
        if relevance > 0:
            qrels.setdefault(str(qrel.query_id), {})[str(qrel.doc_id)] = relevance
    evaluated_queries = {
        query_id: text for query_id, text in queries.items() if qrels.get(query_id)
    }
    return Benchmark(
        name=name,
        split=split,
        documents=documents,
        queries=evaluated_queries,
        qrels=qrels,
    )


def _pubmed_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname not in {
        "www.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
    }:
        return None
    parts = parsed.path.rstrip("/").split("/")
    if len(parts) < 3 or parts[-2].casefold() != "pubmed":
        return None
    identifier = parts[-1]
    return identifier if identifier.isdecimal() else None


def iter_document_texts(documents: Iterable[Document]) -> Iterable[str]:
    for document in documents:
        yield document.retrieval_text
