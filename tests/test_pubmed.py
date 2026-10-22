from pathlib import Path

from evidence_rag.config import Settings
from evidence_rag.models import Benchmark, Document
from evidence_rag.pubmed import (
    PubMedClient,
    enrich_benchmark_with_pubmed,
    load_pubmed_documents,
    parse_pubmed_xml,
    store_documents,
)

XML = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>123</PMID><Article><Journal><JournalIssue><PubDate><Year>2024</Year>
</PubDate></JournalIssue></Journal><ArticleTitle>Trial <i>title</i></ArticleTitle>
<Abstract><AbstractText Label="A">First.</AbstractText>
<AbstractText Label="B">Second.</AbstractText></Abstract><PublicationTypeList>
<PublicationType>Randomized Controlled Trial</PublicationType>
<PublicationType>Retracted Publication</PublicationType></PublicationTypeList></Article>
<MeshHeadingList><MeshHeading><DescriptorName>Alzheimer Disease</DescriptorName>
</MeshHeading></MeshHeadingList></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


class FakeResponse:
    content = b'{"esearchresult":{"idlist":["123"]}}'

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, *args: object, **kwargs: object) -> FakeResponse:
        self.calls += 1
        return FakeResponse()

    def close(self) -> None:
        return None


def test_parse_pubmed_xml_preserves_evidence_metadata() -> None:
    document = parse_pubmed_xml(XML)[0]
    assert document.doc_id == "123"
    assert document.title == "Trial title"
    assert document.text == "First. Second."
    assert document.year == 2024
    assert document.mesh_headings == ("Alzheimer Disease",)
    assert document.publication_types == ("Randomized Controlled Trial", "Retracted Publication")
    assert document.is_retracted is True


def test_client_uses_cache_without_second_request(tmp_path: Path) -> None:
    session = FakeSession()
    client = PubMedClient(tmp_path / "cache.sqlite3", session=session)  # type: ignore[arg-type]
    client.rate_limiter.minimum_interval = 0
    assert client.search("test") == ["123"]
    assert client.search("test") == ["123"]
    assert session.calls == 1
    client.close()


def test_enrich_benchmark_joins_cached_pubmed_metadata(tmp_path: Path) -> None:
    settings = Settings(pubmed_db=tmp_path / "pubmed.sqlite3")
    store_documents(
        settings.pubmed_db,
        [
            Document(
                "25329299",
                "Cached title",
                "Cached abstract",
                2014,
                ("Alzheimer Disease",),
                ("Clinical Trial",),
            )
        ],
    )
    benchmark = Benchmark(
        "nfcorpus",
        [Document("MED-10", "BEIR title", "BEIR abstract", pubmed_id="25329299")],
        {"q1": "query"},
        {"q1": {}},
    )

    coverage = enrich_benchmark_with_pubmed(benchmark, settings, fetch_missing=False)

    enriched = benchmark.documents_by_id["MED-10"]
    assert enriched.title == "BEIR title"
    assert enriched.year == 2014
    assert enriched.mesh_headings == ("Alzheimer Disease",)
    assert enriched.publication_types == ("Clinical Trial",)
    assert coverage["metadata_records"] == 1
    assert coverage["metadata_coverage"] == 1.0


def test_load_pubmed_documents_preserves_pmid_and_fetch_time(tmp_path: Path) -> None:
    path = tmp_path / "pubmed.sqlite3"
    store_documents(path, parse_pubmed_xml(XML))

    documents = load_pubmed_documents(path)

    assert documents[0].doc_id == "123"
    assert documents[0].pubmed_id == "123"
    assert documents[0].metadata_fetched_at is not None
