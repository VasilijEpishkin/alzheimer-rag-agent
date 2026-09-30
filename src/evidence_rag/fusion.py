from collections import defaultdict
from collections.abc import Sequence

from evidence_rag.retrieval import SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], *, k: int = 60, limit: int = 100
) -> list[SearchResult]:
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.doc_id] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [SearchResult(doc_id, score) for doc_id, score in ordered]
