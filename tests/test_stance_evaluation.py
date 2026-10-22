import json

from evidence_rag.models import Document
from evidence_rag.stance_evaluation import _metrics, _stance_examples


def test_stance_examples_map_scipact_labels_and_empty_evidence(tmp_path):
    claims_path = tmp_path / "claims_dev.jsonl"
    claims_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "claim": "Claim one",
                        "evidence": {"d1": [{"label": "SUPPORT", "sentences": [0]}]},
                        "cited_doc_ids": ["d1"],
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "claim": "Claim two",
                        "evidence": {},
                        "cited_doc_ids": ["d2"],
                    }
                ),
            ]
        )
    )
    documents = {"d1": Document("d1", "Title", "Text"), "d2": Document("d2", "Title", "Text")}

    examples = _stance_examples(claims_path, documents)

    assert [(example[0], example[3]) for example in examples] == [
        ("1", "SUPPORTS"),
        ("2", "NOT_ENOUGH_INFO"),
    ]


def test_stance_metrics_cover_all_three_labels():
    metrics = _metrics(
        ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
        ["SUPPORTS", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"],
    )

    assert metrics["accuracy"] == 2 / 3
    assert 0 <= metrics["macro_f1"] <= 1
