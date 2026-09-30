# Phase 1: hybrid retrieval evaluation

## Scope and environment

Phase 1 evaluates BM25, dense MedCPT retrieval, and reciprocal-rank fusion (RRF) on the official BEIR test splits exposed by `ir_datasets`:

- SciFact: 5,183 documents, 300 evaluated queries
- NFCorpus: 3,633 documents, 323 evaluated queries

The run used Python 3.11, `bm25s 0.2.14`, `ir-datasets 0.5.11`, `transformers 4.57.6`, CPU-only `torch 2.14.1+cpu`, `numpy 2.4.6`, and `mlflow 3.16.1`. Dense retrieval used `ncbi/MedCPT-Article-Encoder` and `ncbi/MedCPT-Query-Encoder`, normalized CLS embeddings, batches of 16, and disk-backed NumPy embeddings. Hybrid retrieval used RRF with `k=60`. Every method retrieved at most 100 documents.

All model loading, indexing, evaluation, and test runs were executed through the 2.6 GB `~/.local/bin/heavy` memory scope. PyTorch reported `cuda=False`; only one model was resident at a time.

## Reproduction

From a clean checkout with Python 3.11 and `uv`:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev]'

~/.local/bin/heavy .venv/bin/evidence-rag index scifact --method all
~/.local/bin/heavy .venv/bin/evidence-rag evaluate scifact
~/.local/bin/heavy .venv/bin/evidence-rag index nfcorpus --method all
~/.local/bin/heavy .venv/bin/evidence-rag evaluate nfcorpus
~/.local/bin/heavy bash -lc '.venv/bin/python -m pytest -q && .venv/bin/ruff check .'
```

The completed run reused the already-built SciFact BM25 index, rebuilt its dense index, and built both NFCorpus indexes. Result files are `data/results/scifact.json` and `data/results/nfcorpus.json`; indexes and result JSON are intentionally gitignored.

## Methodology

Metrics are macro-averaged over evaluated queries. Recall is reported at 10 and 100; reciprocal rank and nDCG are truncated at 10. Relevance labels greater than zero count as relevant for Recall and MRR; nDCG retains graded gains.

The 95% confidence intervals are query-level percentile bootstrap intervals with 10,000 resamples and seed 42. Pairwise p-values use a two-sided paired random-sign permutation test over per-query metric differences with 10,000 samples, seed 42, and the finite-sample correction `(extreme + 1) / (samples + 1)`. The significance threshold is 0.05; no multiple-comparison correction was applied, so p-values should be interpreted as exploratory pairwise tests.

## Results

Values are point estimate with bootstrap 95% CI.

### SciFact

| Method | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.7739 [0.7270, 0.8186] | 0.8759 [0.8389, 0.9109] | 0.6312 [0.5826, 0.6779] | 0.6617 [0.6150, 0.7064] |
| Dense MedCPT | 0.8728 [0.8361, 0.9082] | 0.9743 [0.9553, 0.9900] | 0.6867 [0.6420, 0.7310] | 0.7276 [0.6875, 0.7674] |
| Hybrid RRF | 0.8554 [0.8159, 0.8923] | 0.9800 [0.9633, 0.9933] | 0.6912 [0.6445, 0.7357] | 0.7256 [0.6830, 0.7659] |

Paired permutation p-values:

| Comparison | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense vs BM25 | 0.0001 | 0.0001 | 0.0059 | 0.0007 |
| Hybrid vs BM25 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| Hybrid vs Dense | 0.3588 | 0.2495 | 0.7596 | 0.8768 |

Dense MedCPT and hybrid RRF both significantly outperform BM25 on all four SciFact metrics. Hybrid is not significantly different from dense on any metric. In particular, hybrid has slightly higher MRR@10 and Recall@100 but slightly lower Recall@10 and nDCG@10; none of those differences supports an improvement claim.

### NFCorpus

| Method | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.1450 [0.1203, 0.1717] | 0.2387 [0.2104, 0.2687] | 0.5186 [0.4701, 0.5672] | 0.3073 [0.2734, 0.3418] |
| Dense MedCPT | 0.1653 [0.1393, 0.1928] | 0.3158 [0.2849, 0.3480] | 0.5128 [0.4640, 0.5613] | 0.3315 [0.2976, 0.3665] |
| Hybrid RRF | 0.1651 [0.1388, 0.1931] | 0.3141 [0.2839, 0.3460] | 0.5393 [0.4913, 0.5866] | 0.3402 [0.3051, 0.3754] |

Paired permutation p-values:

| Comparison | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense vs BM25 | 0.0082 | 0.0001 | 0.7770 | 0.0293 |
| Hybrid vs BM25 | 0.0001 | 0.0001 | 0.0945 | 0.0001 |
| Hybrid vs Dense | 0.9767 | 0.7654 | 0.1121 | 0.2831 |

Dense MedCPT significantly improves Recall@10, Recall@100, and nDCG@10 over BM25, but not MRR@10. Hybrid significantly improves both Recall metrics and nDCG@10 over BM25, but its MRR@10 difference is not significant. Hybrid is not significantly different from dense on any metric.

## MLflow evidence

MLflow experiment `phase-1-retrieval` has experiment ID `500054413502491248` and six finished runs. Each run logs the method's four aggregate metrics and carries the full dataset JSON as an artifact, including per-query metrics, bootstrap intervals, and all paired p-values.

| Run | Run ID | Artifact |
|---|---|---|
| scifact-bm25 | `608bba7300174199a1686c663c4f6e99` | `scifact.json` |
| scifact-dense | `b3c3d82de2564178a3d1d7d5f7b4d2a0` | `scifact.json` |
| scifact-hybrid | `838a08e2a7854d4883a3603864223d38` | `scifact.json` |
| nfcorpus-bm25 | `45d16ae4d0eb41de99821ba00f55f0e3` | `nfcorpus.json` |
| nfcorpus-dense | `101e263952a54eb0a7a1cc66a87f3f93` | `nfcorpus.json` |
| nfcorpus-hybrid | `2c9a4c37221b43ddb0ba5bf0c08e4b8a` | `nfcorpus.json` |

The local store is `mlruns/` and is intentionally gitignored. MLflow metric names replace `@` with `_at_`; the JSON artifact retains the display names such as `recall@10`.

## Validation

The final test and lint command completed successfully:

```text
............                                                             [100%]
All checks passed!
```

The added evaluation test uses an isolated local MLflow store and fake in-memory retrievers; it verifies the three result sections, confidence intervals, paired comparisons, three finished runs, and MLflow-safe metric names without network or model access.

## Open problems

1. RRF does not significantly outperform dense MedCPT on either benchmark. Phase 2 should treat dense as a serious baseline rather than assume fusion helps.
2. NFCorpus remains difficult: all Recall@10 estimates are below 0.17. Cross-encoder and LTR gains must be measured against this low-recall candidate set, and increasing the candidate depth should be evaluated without test-set tuning.
3. The p-values are unadjusted across metrics and comparisons. A later consolidated report should either control family-wise/FDR error or explicitly retain the exploratory label.
4. Local MLflow 3.16.1 places its filesystem backend in maintenance mode. The phase keeps the specified local `mlruns/` store via `MLFLOW_ALLOW_FILE_STORE=true`; a future migration to local SQLite can be considered without changing the recorded Phase 1 results.
5. Phase 1 covers only public retrieval benchmarks. The adopted evidence-safety set, retraction policy, claim-evidence ledger, and evidence-quality-aware counterfactual ranking belong to later phases and were not implemented here.
