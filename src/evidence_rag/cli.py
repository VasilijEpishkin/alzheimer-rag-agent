from typing import Annotated

import typer

from evidence_rag.config import Settings
from evidence_rag.datasets import available_datasets

app = typer.Typer(no_args_is_help=True, help="Evidence-aware biomedical retrieval tools.")


@app.command()
def ingest(
    query: Annotated[str, typer.Argument(help="PubMed query")],
    limit: Annotated[int, typer.Option(min=1, max=10_000)] = 10_000,
) -> None:
    """Fetch PubMed records into the local SQLite article store."""
    from evidence_rag.pubmed import PubMedClient, store_documents

    settings = Settings()
    settings.ensure_directories()
    client = PubMedClient(
        settings.ncbi_cache_db,
        api_key=settings.ncbi_api_key,
        email=settings.ncbi_email,
        tool=settings.ncbi_tool,
    )
    try:
        pmids = client.search(query, limit)
        count = store_documents(settings.pubmed_db, client.fetch(pmids))
    finally:
        client.close()
    typer.echo(f"Stored {count} PubMed records in {settings.pubmed_db}")


@app.command()
def index(
    dataset: Annotated[str, typer.Argument(help=f"One of: {', '.join(available_datasets())}")],
    method: Annotated[str, typer.Option(help="bm25, dense, or all")] = "all",
) -> None:
    """Build disk-backed indexes for a BEIR dataset."""
    from evidence_rag.datasets import load_beir
    from evidence_rag.retrieval import BM25Index, DenseIndex

    if method not in {"bm25", "dense", "all"}:
        raise typer.BadParameter("method must be bm25, dense, or all")
    settings = Settings()
    settings.ensure_directories()
    benchmark = load_beir(dataset)
    base = settings.index_dir / dataset
    if method in {"bm25", "all"}:
        typer.echo(f"Building BM25 for {dataset} ({len(benchmark.documents)} documents)")
        BM25Index.build(benchmark.documents).save(base / "bm25")
    if method in {"dense", "all"}:
        typer.echo(
            f"Encoding {dataset} in batches of {settings.embedding_batch_size}; "
            "the article encoder is released after indexing"
        )
        DenseIndex.build(
            benchmark.documents,
            base / "dense",
            settings.medcpt_article_model,
            settings.embedding_batch_size,
        )
    typer.echo(f"Indexes written to {base}")


@app.command()
def evaluate(
    dataset: Annotated[str, typer.Argument(help=f"One of: {', '.join(available_datasets())}")],
) -> None:
    """Evaluate BM25, dense, and hybrid retrieval and log to MLflow."""
    from evidence_rag.evaluation import evaluate_dataset

    output = evaluate_dataset(dataset, Settings())
    typer.echo(f"Results written to {output}")


@app.command()
def phase2(
    dataset: Annotated[str, typer.Argument(help=f"One of: {', '.join(available_datasets())}")],
) -> None:
    """Train and evaluate cross-encoder and learning-to-rank methods."""
    from evidence_rag.phase2 import evaluate_phase2

    output = evaluate_phase2(dataset, Settings())
    typer.echo(f"Phase 2 results written to {output}")


if __name__ == "__main__":
    app()
