import sys
from types import SimpleNamespace

from evidence_rag import evidence_layer


def test_stance_model_is_pinned_and_maps_model_labels(monkeypatch) -> None:
    calls = {}

    def pipeline(task, **kwargs):
        calls["load"] = (task, kwargs)

        def classify(pair, **options):
            calls["input"] = (pair, options)
            return [
                {"label": "NOT_ENOUGH_INFO", "score": 0.5},
                {"label": "SUPPORTS", "score": 0.3},
                {"label": "REFUTES", "score": 0.2},
            ]

        return classify

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    monkeypatch.setattr(evidence_layer, "_STANCE_MODEL", None)

    model = evidence_layer._load_stance_model()

    assert calls["load"] == (
        "text-classification",
        {
            "model": evidence_layer._STANCE_MODEL_ID,
            "revision": evidence_layer._STANCE_MODEL_REVISION,
            "device": -1,
        },
    )
    assert model("claim", "evidence") == [0.3, 0.2, 0.5]
    assert calls["input"] == (
        {"text": "claim", "text_pair": "evidence"},
        {"top_k": None, "truncation": True, "max_length": 512},
    )
