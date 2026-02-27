# Evaluation logic for RAG performance
from typing import List, Dict
import numpy as np

def calculate_mrr(relevance_scores: List[List[int]]) -> float:
    rr = []
    for scores in relevance_scores:
        for i, score in enumerate(scores):
            if score > 0:
                rr.append(1 / (i + 1))
                break
        else:
            rr.append(0)
    return np.mean(rr)

def calculate_hit_rate(relevance_scores: List[List[int]], k: int = 5) -> float:
    hits = 0
    for scores in relevance_scores:
        if any(s > 0 for s in scores[:k]):
            hits += 1
    return hits / len(relevance_scores)
