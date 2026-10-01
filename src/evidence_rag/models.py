from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    year: int | None = None
    mesh_headings: tuple[str, ...] = ()
    publication_types: tuple[str, ...] = ()
    is_retracted: bool = False

    @property
    def retrieval_text(self) -> str:
        return f"{self.title} {self.text}".strip()


@dataclass(slots=True)
class Benchmark:
    name: str
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    split: str = "test"
    _documents_by_id: dict[str, Document] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._documents_by_id = {document.doc_id: document for document in self.documents}

    @property
    def documents_by_id(self) -> dict[str, Document]:
        return self._documents_by_id
