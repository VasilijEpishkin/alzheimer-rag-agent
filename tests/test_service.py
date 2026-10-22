import asyncio

from httpx2 import ASGITransport, AsyncClient

from evidence_rag import service
from evidence_rag.config import Settings
from evidence_rag.models import Benchmark, Document
from evidence_rag.reranking import Candidate
from evidence_rag.retrieval import SearchResult


def _configure_indexes(monkeypatch, tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        index_dir=tmp_path / "indexes",
        results_dir=tmp_path / "results",
        pubmed_db=tmp_path / "pubmed.sqlite3",
        ncbi_cache_db=tmp_path / "cache" / "ncbi.sqlite3",
    )
    settings.ensure_directories()
    (settings.index_dir / "scifact").mkdir(exist_ok=True)
    monkeypatch.setattr(service, "Settings", lambda: settings)
    monkeypatch.setattr(
        service,
        "load_beir",
        lambda dataset: Benchmark(dataset, [Document("123", "title", "abstract")], {}, {}),
    )
    monkeypatch.setattr(service, "_load_indexes", lambda *_args: (object(), object()))
    monkeypatch.setattr(
        service,
        "_retrieve_and_rerank",
        lambda *_args: (
            [Document("123", "title", "abstract")],
            [Candidate("service_query", "123", 1.0, 1.0, 1, 1, 1.0)],
            [1.0],
            [],
        ),
    )

def _post(payload: dict) -> object:
    async def request():
        async with AsyncClient(
            transport=ASGITransport(app=service.app), base_url="http://testserver"
        ) as client:
            return await client.post("/query", json=payload)

    return asyncio.run(request())


def test_query_endpoint_returns_generated_answer_and_verifies_citations(
    monkeypatch, tmp_path
) -> None:
    _configure_indexes(monkeypatch, tmp_path)
    monkeypatch.setattr(service, "_compute_answerability", lambda *_args: ([0.9], {}))
    monkeypatch.setattr(
        service, "generate_answer", lambda *_args: "Claim supported by [PMID:123]."
    )
    response = _post({"query": "question", "dataset": "scifact"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Claim supported by [PMID:123]."
    assert response.json()["verification_unverified_pmids"] == []


def test_query_endpoint_returns_503_when_answerability_model_is_missing(
    monkeypatch, tmp_path
) -> None:
    _configure_indexes(monkeypatch, tmp_path)

    def missing_model(*_args):
        raise FileNotFoundError("answerability model is not trained")

    monkeypatch.setattr(service, "_compute_answerability", missing_model)
    response = _post({"query": "question", "dataset": "scifact"})

    assert response.status_code == 503
    assert response.json()["detail"] == "answerability model is not trained"


def test_query_endpoint_rejects_unknown_dataset() -> None:
    response = _post({"query": "question", "dataset": "unknown"})

    assert response.status_code == 422


def test_retracted_documents_are_excluded_from_answer_context(monkeypatch, tmp_path) -> None:
    class FakeBM25:
        def search(self, _query, _count):
            return [SearchResult("retracted", 2.0), SearchResult("safe", 1.0)]

    class FakeDense:
        def search_vector(self, _query, _count):
            return [SearchResult("retracted", 2.0), SearchResult("safe", 1.0)]

    class FakeEncoder:
        def __init__(self, _model):
            pass

        def encode(self, _texts, _batch_size):
            yield [0.0]

        def close(self):
            pass

    class FakeCrossEncoder:
        def __init__(self, _model):
            pass

        def score(self, _query, documents, _batch_size):
            return [0.9 if document.doc_id == "retracted" else 0.8 for document in documents]

        def close(self):
            pass

    monkeypatch.setattr(service, "MedCPTQueryEncoder", FakeEncoder)
    monkeypatch.setattr(service, "MedCPTCrossEncoder", FakeCrossEncoder)
    settings = Settings(data_dir=tmp_path, ltr_candidate_k=10)
    benchmark = Benchmark(
        "pubmed",
        [
            Document("retracted", "Old", "Text", is_retracted=True),
            Document("safe", "Current", "Text"),
        ],
        {},
        {},
    )

    documents, candidates, scores, excluded = service._retrieve_and_rerank(
        "query", benchmark, FakeBM25(), FakeDense(), settings, top_k=1
    )

    assert [document.doc_id for document in documents] == ["safe"]
    assert [candidate.doc_id for candidate in candidates] == ["safe"]
    assert scores == [0.8]
    assert [document.doc_id for document, _ in excluded] == ["retracted"]
