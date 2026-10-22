import pytest

from evidence_rag.config import Settings
from evidence_rag.generation import build_generation_prompt, generate_answer, verify_citations
from evidence_rag.models import Document


def test_citation_verifier_accepts_pmid_format_and_flags_unknown_ids() -> None:
    documents = [Document("123", "title", "abstract")]

    assert verify_citations("Supported [PMID:123] and [123]", documents) == []
    assert verify_citations("Unsupported [PMID:456]", documents) == ["456"]


def test_prompt_uses_the_same_citation_format_as_the_verifier() -> None:
    prompt = build_generation_prompt("query", [Document("123", "title", "abstract")], [], 1.0)

    assert "[PMID:123]" in prompt
    assert "[PMID:123]" in prompt.split("Answer:", maxsplit=1)[0]


def test_generation_abstains_without_an_api_key_when_probability_is_low(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    answer = generate_answer("query", [], [], 0.2, Settings(openrouter_api_key=None))

    assert answer == "I do not have enough evidence to answer this question."


def test_generation_never_returns_a_placeholder_when_the_api_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is required"):
        generate_answer("query", [], [], 0.9, Settings(openrouter_api_key=None))
