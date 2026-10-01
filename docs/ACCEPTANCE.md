# Acceptance criteria (set by the lead, 2026-10-01)

Phases 2 and 3 were closed with scaffolding, not results: Phase 2 "evaluated" 2 of 300 test queries with
5 candidates and 2 training iterations; Phase 3 ships a uniform-placeholder stance model and an untrained
answerability model. Their review cards left no verdict. A card is DONE only when every box below is true,
checked by the reviewer against files on disk — otherwise the reviewer REJECTS it with the failing boxes.

## Phase 2b — real reranking / LTR evaluation
- [ ] Full official test splits: SciFact 300 queries, NFCorpus 323 queries. No query subsetting.
- [ ] `ltr_candidate_k` = 100 (hybrid RRF candidates); cross-encoder scores every candidate.
- [ ] LTR trained on the official train split with real iterations (>= 300) and early stopping on dev, never on test.
- [ ] Evidence features are REAL on NFCorpus: its documents carry PubMed URLs/PMIDs — fetch PublicationType,
      year, MeSH and retraction status via the existing PubMed ingest (cached), join by PMID, report coverage.
      SciFact has no PMIDs: there, run relevance-only features and say so.
- [ ] Compare: hybrid RRF / cross-encoder / CatBoost-relevance / CatBoost-evidence / XGBoost-relevance /
      XGBoost-evidence, with bootstrap CIs and paired permutation p-values; counterfactual report
      (which results moved and why) for the evidence variants.
- [ ] Every run in MLflow; numbers in docs/reports/phase-2.md regenerated from the result JSON.
- [ ] CPU only, all heavy commands via `~/.local/bin/heavy`; if the 2.6 GB cap is hit, lower batch size, not scope.

## Phase 3b — real evidence layer and answer pipeline
- [ ] Stance: a real model (a biomedical/scientific NLI or claim-verification model, e.g. trained on SciFact)
      evaluated on SciFact claim labels (support/refute/NEI) with accuracy/macro-F1 and CI. No placeholder.
- [ ] Answerability: CatBoost trained on an explicitly declared weak proxy label (document it), calibrated
      (isotonic or Platt) on a held-out split; report ECE and a reliability plot artifact.
- [ ] Claim-evidence ledger persisted per answer (SQLite/Parquet) with the fields in SPEC "Research proposals".
- [ ] Generation via OpenRouter on >= 20 real questions over the PubMed demo corpus: citation precision
      (cited PMIDs ⊆ retrieved), share of answers that abstain, and a small LLM-as-judge faithfulness score.
      If OpenRouter fails, say so — never fake outputs.
- [ ] FastAPI endpoints tested; Dockerfile builds (`docker build`), README results table generated from JSON.

## Reviews
Reviewer runs the tests, reopens the result JSON, recounts queries/candidates, and posts ACCEPT or REJECT
with the box-by-box checklist as a card comment. No comment = not reviewed.
