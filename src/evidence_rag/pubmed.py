import hashlib
import json
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from evidence_rag.models import Document

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_RETRACTION_TYPES = {"retracted publication", "retraction of publication"}
_RETRACTION_REFS = {"retractionin", "retractionof"}


class SqliteResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, body BLOB NOT NULL, created_at REAL NOT NULL)"
        )

    def get(self, key: str) -> bytes | None:
        row = self.connection.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def put(self, key: str, body: bytes) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, body, time.time())
            )

    def close(self) -> None:
        self.connection.close()


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.minimum_interval = 1.0 / requests_per_second
        self._last_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self.minimum_interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()


class PubMedClient:
    def __init__(
        self,
        cache_path: Path,
        *,
        api_key: str | None = None,
        email: str = "",
        tool: str = "evidence-rag",
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.email = email
        self.tool = tool
        self.cache = SqliteResponseCache(cache_path)
        self.session = session or self._build_session()
        self.rate_limiter = RateLimiter(10.0 if api_key else 3.0)

    @staticmethod
    def _build_session() -> requests.Session:
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _get(self, endpoint: str, params: dict[str, Any]) -> bytes:
        request_params = {"tool": self.tool, **params}
        if self.email:
            request_params["email"] = self.email
        if self.api_key:
            request_params["api_key"] = self.api_key
        encoded = json.dumps([endpoint, sorted(request_params.items())], separators=(",", ":"))
        key = hashlib.sha256(encoded.encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.rate_limiter.wait()
        response = self.session.get(f"{_EUTILS}/{endpoint}", params=request_params, timeout=60)
        response.raise_for_status()
        self.cache.put(key, response.content)
        return response.content

    def search(self, query: str, limit: int = 10_000) -> list[str]:
        payload = self._get(
            "esearch.fcgi",
            {"db": "pubmed", "term": query, "retmode": "json", "retmax": min(limit, 10_000)},
        )
        return [str(item) for item in json.loads(payload)["esearchresult"]["idlist"]]

    def fetch(self, pmids: Iterable[str], batch_size: int = 200) -> Iterable[Document]:
        identifiers = list(pmids)
        for start in range(0, len(identifiers), batch_size):
            payload = self._get(
                "efetch.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(identifiers[start : start + batch_size]),
                    "retmode": "xml",
                },
            )
            yield from parse_pubmed_xml(payload)

    def close(self) -> None:
        self.cache.close()
        self.session.close()


def _text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def _year(article: ET.Element) -> int | None:
    for xpath in (
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
    ):
        value = article.findtext(xpath)
        if value and value.isdigit():
            return int(value)
    medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate", "")
    for token in medline_date.split():
        if len(token) >= 4 and token[:4].isdigit():
            return int(token[:4])
    return None


def parse_pubmed_xml(payload: bytes) -> list[Document]:
    root = ET.fromstring(payload)
    documents: list[Document] = []
    for record in root.findall(".//PubmedArticle"):
        citation = record.find("MedlineCitation")
        if citation is None:
            continue
        article = citation.find("Article")
        pmid = citation.findtext("PMID")
        if article is None or not pmid:
            continue
        publication_type_elements = article.findall(".//PublicationTypeList/PublicationType")
        publication_types = tuple(filter(None, map(_text, publication_type_elements)))
        references = {
            item.attrib.get("RefType", "").casefold()
            for item in record.findall(".//CommentsCorrections")
        }
        is_retracted = bool(
            {item.casefold() for item in publication_types} & _RETRACTION_TYPES
            or references & _RETRACTION_REFS
        )
        abstract = " ".join(
            filter(None, (_text(item) for item in article.findall(".//Abstract/AbstractText")))
        )
        documents.append(
            Document(
                doc_id=str(pmid),
                title=_text(article.find("ArticleTitle")),
                text=abstract,
                year=_year(record),
                mesh_headings=tuple(
                    filter(None, map(_text, citation.findall(".//MeshHeading/DescriptorName")))
                ),
                publication_types=publication_types,
                is_retracted=is_retracted,
            )
        )
    return documents


def store_documents(path: Path, documents: Iterable[Document]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS articles ("
        "pmid TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL, year INTEGER, "
        "mesh_headings TEXT NOT NULL, publication_types TEXT NOT NULL, "
        "is_retracted INTEGER NOT NULL)"
    )
    count = 0
    with connection:
        for document in documents:
            connection.execute(
                "INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.doc_id,
                    document.title,
                    document.text,
                    document.year,
                    json.dumps(document.mesh_headings),
                    json.dumps(document.publication_types),
                    int(document.is_retracted),
                ),
            )
            count += 1
    connection.close()
    return count
