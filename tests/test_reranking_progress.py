from contextlib import nullcontext
from types import SimpleNamespace

from evidence_rag.models import Document
from evidence_rag.reranking import MedCPTCrossEncoder


def test_cross_encoder_logs_every_ten_batches(caplog) -> None:
    class FakeTensor:
        def reshape(self, *_shape):
            return self

        def cpu(self):
            return [1.0]

    scorer = object.__new__(MedCPTCrossEncoder)
    scorer.torch = SimpleNamespace(inference_mode=nullcontext)
    scorer.tokenizer = lambda *_args, **_kwargs: {"input_ids": [1]}
    scorer.model = lambda **_kwargs: SimpleNamespace(logits=FakeTensor())

    with caplog.at_level("INFO", logger="evidence_rag.reranking"):
        scores = scorer.score("query", [Document(str(i), "title", "text") for i in range(11)], 1)

    assert len(scores) == 11
    assert "batch 10/11" in caplog.text
    assert "batch 11/11" in caplog.text
