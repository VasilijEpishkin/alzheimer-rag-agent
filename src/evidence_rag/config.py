from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    index_dir: Path = Path("data/indexes")
    results_dir: Path = Path("data/results")
    pubmed_db: Path = Path("data/pubmed.sqlite3")
    ncbi_cache_db: Path = Path("data/cache/ncbi.sqlite3")
    ncbi_api_key: str | None = Field(default=None, repr=False)
    ncbi_email: str = ""
    ncbi_tool: str = "evidence-rag"
    medcpt_article_model: str = "ncbi/MedCPT-Article-Encoder"
    medcpt_query_model: str = "ncbi/MedCPT-Query-Encoder"
    embedding_batch_size: int = Field(default=16, ge=1, le=16)
    retrieval_k: int = Field(default=100, ge=10, le=1000)
    rrf_k: int = Field(default=60, ge=1)
    bootstrap_samples: int = Field(default=10_000, ge=100)
    permutation_samples: int = Field(default=10_000, ge=100)
    random_seed: int = 42
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "phase-1-retrieval"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.cache_dir, self.index_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)
