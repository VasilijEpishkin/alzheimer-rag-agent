import pytest
from pydantic import ValidationError

from evidence_rag.config import Settings


def test_embedding_batch_defaults_to_memory_safe_limit() -> None:
    assert Settings().embedding_batch_size == 16


def test_embedding_batch_rejects_values_above_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(embedding_batch_size=32)
