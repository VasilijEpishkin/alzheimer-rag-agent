from evidence_rag.config import Settings


def test_settings_use_bounded_embedding_batch() -> None:
    settings = Settings(embedding_batch_size=32)
    assert settings.embedding_batch_size == 32
