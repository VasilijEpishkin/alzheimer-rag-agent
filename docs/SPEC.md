# Evidence-aware biomedical RAG — specification

## Why this project exists (the differentiator)

Existing biomedical RAG systems (MedRAG / MIRAGE benchmark, PaperQA2) retrieve papers and cite them,
but treat every retrieved abstract as equally trustworthy. In biomedicine that is wrong in three ways:

1. **Evidence strength.** A meta-analysis or RCT outweighs a case report or an opinion piece.
   PubMed already tags this (`PublicationType`); retrieval and answers must use it.
2. **Retracted papers.** Retracted publications keep getting cited by LLM answers. PubMed marks them
   (`Retracted Publication`, `Retraction of Publication`); they must be excluded or loudly flagged.
3. **Conflicting findings.** Biomedical literature disagrees. An answer must show supporting AND
   contradicting evidence, not a confident one-sided summary.

Plus **calibrated abstention**: when the evidence is weak or absent, the system says so. A gradient
boosting model (CatBoost / XGBoost) predicts "is there enough evidence to answer" from retrieval
features; its calibration is measured (reliability curve, ECE), not assumed.

Domain: general PubMed literature. The demo corpus is neurodegeneration (Alzheimer's, Parkinson's),
but nothing is Alzheimer-specific.

## Architecture (backend-first; the UI is a thin afterthought)

```
PubMed E-utilities ──► ingest (cache, retries, rate limit) ──► Parquet/SQLite store
                                                             (abstract, title, year, MeSH,
                                                              PublicationType, retraction flag)
                                    │
                    ┌───────────────┴────────────────┐
                 BM25 (bm25s)                  dense: MedCPT query/article encoders → ChromaDB (or FAISS)
                    └──────────── RRF fusion ────────┘
                                    │
                    cross-encoder rerank (MedCPT-Cross-Encoder)
                                    │
                    learning-to-rank (CatBoost YetiRank vs XGBoost rank:pairwise)
                    features: bm25, dense, cross-encoder, evidence level, year, title/MeSH overlap
                                    │
                    evidence layer: study-design weight, retraction filter,
                                    stance (supports / contradicts / neutral) via biomedical NLI
                                    │
                    answerability model (CatBoost, calibrated) ──► abstain or answer
                                    │
                    LLM (OpenRouter, OpenAI-compatible client) with mandatory [PMID] citations
                    + citation verifier: every cited PMID must be in the retrieved context
                                    │
                    FastAPI service (JSON API); minimal UI optional
```

Engineering requirements: Python 3.11, `src/` layout package, pydantic-settings config (.env),
typed code, pytest with fixtures that never hit the network, a CLI (typer) for ingest / index /
evaluate / serve, Dockerfile + docker-compose, ruff. Secrets only from env (OPENROUTER_API_KEY,
NCBI_API_KEY optional). Machine limits: 7 GB RAM, RTX 3060 6 GB — keep corpora small
(≤ 10k abstracts), batch embeddings, no model > 1B params loaded locally.

## Evaluation (honest numbers only)

- Retrieval: Recall@k, MRR@10, nDCG@10 on public labelled biomedical sets (BEIR SciFact and
  NFCorpus via `ir_datasets` or Hugging Face) + a hand-checked neurodegeneration question set with
  real PMIDs (≥ 30 questions, every PMID verified to exist).
- Compare configs: BM25 / dense / hybrid RRF / + cross-encoder / + LTR (CatBoost vs XGBoost).
- Statistics: bootstrap 95% CIs per metric; paired permutation test (or Wilcoxon) between configs,
  report p-values; no "improvement" claimed without significance.
- Generation (later phase): citation precision (cited PMIDs ⊆ retrieved), faithfulness via
  LLM-as-a-judge on a small set; abstention calibration (ECE, reliability plot).
- MLflow: every evaluation run and every LTR / answerability training is an MLflow run
  (params, metrics, artifacts). Local file store `mlruns/`.
- README results table is generated from the eval output, never typed by hand.

## Phases

1. **Data + hybrid retrieval + evaluation harness.** Package skeleton, config, PubMed ingest with
   PublicationType + retraction flag, BM25 + MedCPT dense + RRF, BEIR loaders, metrics, bootstrap
   CIs, paired tests, MLflow logging, CLI, tests. Deliverable: `evaluate` produces a results table
   with CIs for BM25 / dense / hybrid on SciFact + NFCorpus.
2. **Reranking + learning-to-rank.** Cross-encoder, feature builder, CatBoost vs XGBoost LTR trained
   on train qrels, evaluated on test with significance tests.
3. **Evidence layer + answerability + generation.** Study-design weighting, retraction filter,
   stance detection, calibrated answerability model, OpenRouter generation with citation verifier,
   FastAPI service, Docker, README with generated results.

## Rules for the implementing agent

- Work only inside this repository. Commit locally at the end of each phase with clear messages.
  **Never `git push`**, never change the remote, never rewrite history.
- Never fabricate data, PMIDs or metrics. If a run could not be done, say so in the report.
- Stop after the phase you were asked to do and write `docs/reports/phase-N.md`: what was built,
  commands to reproduce, real results, open problems.

## Research proposals (2026-09-30) — ADOPTED by the lead on 2026-09-30

Adopted: the claim-evidence ledger, versioned retraction policy with `metadata_fetched_at`, the evidence-safety evaluation set and non-clinical scope, and evidence-quality-aware LTR with counterfactual reporting (relevance-only ranking retained for audit). Phases 2 and 3 must implement them; safety metrics are reported next to nDCG/MRR, never folded into one score.

### Original proposals

These proposals are research input only; they do not alter the committed phase scope until the owner
accepts them.

1. Add a versioned **claim-evidence ledger** as an internal/API artefact.  For every generated claim
   retain the cited PMID and passage, retrieval/rerank scores, publication types, retraction status
   and fetch time, evidence-policy weight, stance probabilities, and final answer/qualified/abstain
   decision.  This makes evidence decisions auditable rather than merely producing citations.
2. Make retraction behaviour a policy with a safe default: keep an excluded result in the audit log
   with `excluded_reason=retracted`, prohibit it from supporting a claim, and retain the linked
   notice.  Refresh PubMed metadata before answer/export; an explicit audit mode may display it only
   with a conspicuous label.
3. Add a small, manually reviewed safety challenge after Phase 1: real PMIDs pre-labelled for
   retracted/corrected status, publication-type cases, and support/refute/insufficient evidence.
   Report retraction-safety and support/refute balance separately from MRR/nDCG; describe it as a
   project test set, not a clinical gold standard.
4. Treat study-design weight and NLI stance as uncertain features, not ground truth.  Preserve a
   relevance-only ranking for counterfactual audit, tune/calibrate without test leakage, and emit
   `unknown` where metadata or stance confidence is inadequate.
