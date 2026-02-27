import json
import asyncio
from rag.main import AlzheimerRAG
from rag.evaluation import calculate_mrr, calculate_hit_rate
import os

async def main():
    rag = AlzheimerRAG()
    
    with open("evaluation/test_queries.json", "r") as f:
        test_data = json.load(f)
    
    results = []
    relevance_scores = []
    
    for item in test_data:
        query = item["query"]
        expected_pmid = item["expected_pmid"]
        
        print(f"Evaluating: {query}")
        result = rag.query(query)
        
        # Binary relevance: is the expected PMID in the sources?
        found_pmids = [doc.metadata.get("pmid") for doc in result["source_documents"]]
        scores = [1 if str(pmid) == str(expected_pmid) else 0 for pmid in found_pmids]
        
        relevance_scores.append(scores)
        results.append({
            "query": query,
            "expected_pmid": expected_pmid,
            "found_pmids": found_pmids,
            "answer": result["answer"]
        })
    
    metrics = {
        "mrr": calculate_mrr(relevance_scores),
        "hit_rate@5": calculate_hit_rate(relevance_scores, k=5)
    }
    
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/eval_results.json", "w") as f:
        json.dump({"metrics": metrics, "detailed_results": results}, f, indent=2)
    
    print(f"Evaluation complete. Metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())
