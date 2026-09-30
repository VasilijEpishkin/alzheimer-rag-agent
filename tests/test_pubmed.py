from pathlib import Path

from evidence_rag.pubmed import PubMedClient, parse_pubmed_xml

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
