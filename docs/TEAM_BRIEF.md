# Team brief: evidence-aware biomedical RAG pet project

Owner: human (Vasilij). Organizer: `default`. Research: `res`. Implementation + review: `dev`.
Repository: ~/workspace/alzheimer-rag-agent (local; push is disabled until human review).
Spec: docs/SPEC.md. Phase 1 code is committed (package `evidence_rag`); its evaluation run is unfinished.

## Direction
A portfolio-grade, genuinely current pet project for a Data Scientist / bioinformatics CV:
retrieval-augmented QA over biomedical literature that weighs EVIDENCE, not just similarity —
study-design hierarchy, retraction awareness, supporting vs contradicting evidence, calibrated
abstention, verified citations. Backend and architecture matter most; UI is minimal.
It must honestly use and demonstrate: learning-to-rank with CatBoost and XGBoost, MLflow experiment
tracking, statistical testing (bootstrap CIs, paired tests, calibration/ECE). LLM via OpenRouter.

## What we want from the team
1. `res`: current state of the art (2025–2026) — what exists (MedRAG/MIRAGE, PaperQA2, OpenScholar,
   Elicit/Consensus-style tools, retraction-aware / evidence-grading work), what is NOT solved, which
   open datasets/benchmarks and models (retrievers, rerankers, biomedical NLI for stance) are usable
   on this machine. Deliver docs/research/landscape.md with links; recommend 1–2 differentiators that
   make the project clearly current and not a clone. Update SPEC.md proposals in a separate section,
   do not silently rewrite it.
2. `default`: turn SPEC.md + research into a Kanban graph on board `rag-pet`: finish phase 1 eval,
   phase 2 (cross-encoder + LTR CatBoost vs XGBoost), phase 3 (evidence layer, answerability,
   generation with citation verifier, FastAPI, Docker, README with generated results), and a review
   card for each implementation card (review by a subagent on a different model).
3. `dev`: implement cards, commit locally per milestone, write docs/reports/phase-N.md with REAL numbers.

## Hard rules
- Never push, never change git remotes, never rewrite history.
- Never fabricate data, PMIDs or metrics.
- MEMORY: the machine has ~2 GB free RAM; a model-loading run crashed the whole WSL VM twice.
  Heavy runs (torch, embedding models, indexing) are BLOCKED until the human confirms more memory.
  Until then: research, design, code and unit tests only. When unblocked: batch small, fp16 on GPU,
  one model in memory at a time.
- Dispatch of `dev` cards requires the human's "ok" in Orchestration; the `res` card may start now.
