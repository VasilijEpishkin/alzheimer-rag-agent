from types import SimpleNamespace

from evidence_rag import datasets


class FakeDataset:
    def docs_iter(self):
        yield SimpleNamespace(doc_id="d1", title="Title", text="Abstract")

    def queries_iter(self):
        yield SimpleNamespace(query_id="q1", text="Query")
        yield SimpleNamespace(query_id="q2", text="No labels")

    def qrels_iter(self):
        yield SimpleNamespace(query_id="q1", doc_id="d1", relevance=2)


def test_loader_keeps_only_evaluated_queries(monkeypatch) -> None:
    monkeypatch.setattr(datasets.ir_datasets, "load", lambda _: FakeDataset())
    benchmark = datasets.load_beir("scifact")
    assert benchmark.queries == {"q1": "Query"}
    assert benchmark.qrels == {"q1": {"d1": 2}, "q2": {}}
    assert benchmark.documents[0].retrieval_text == "Title Abstract"
