from collections.abc import Iterable

import ir_datasets

from evidence_rag.models import Benchmark, Document

_DATASETS = {
    "scifact": "beir/scifact/test",
    "nfcorpus": "beir/nfcorpus/test",
}


def available_datasets() -> tuple[str, ...]:
    return tuple(_DATASETS)


def load_beir(name: str) -> Benchmark:
    try:
        dataset_id = _DATASETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(_DATASETS)}") from error
    dataset = ir_datasets.load(dataset_id)
    documents = [
        Document(
            doc_id=str(document.doc_id),
            title=getattr(document, "title", "") or "",
            text=getattr(document, "text", "") or "",
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
    return Benchmark(name=name, documents=documents, queries=evaluated_queries, qrels=qrels)


def iter_document_texts(documents: Iterable[Document]) -> Iterable[str]:
    for document in documents:
        yield document.retrieval_text
