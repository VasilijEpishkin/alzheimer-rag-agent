# Biomedical evidence-aware RAG: landscape and project position

Research date: 2026-09-30.  This is a design review, not a claim that the proposed stack has been run; heavy model loading is currently blocked by the machine-free-memory constraint in `docs/TEAM_BRIEF.md`.

## Executive position

Retrieval, citation-backed scientific synthesis, and even basic retraction checks are no longer novel by themselves. The defensible project gap is an *auditable evidence-decision layer* between retrieval and generation: each answer claim is tied to passages, their PubMed metadata, a transparent study-design policy, retraction state and a calibrated support/refute/insufficient-evidence decision. The evaluation must directly test those safety behaviours rather than only ranking accuracy.

Two recommended differentiators are:

1. **Versioned claim-evidence ledger and safety evaluation.** Persist a per-answer-claim ledger: cited PMID, passage, retrieval/rerank scores, publication types, retraction status and source timestamp, study-design weight, stance probabilities, and the rule/model outcome (answer / qualified answer / abstain). Test (a) retraction exclusion/flagging, (b) support–refute balance, and (c) abstention calibration. This goes beyond citing papers or merely checking a retraction at indexing time.
2. **Evidence-quality-aware LTR with counterfactual reporting.** Treat relevance and evidentiary suitability as distinct. Compare ordinary LTR with a policy-aware reranking feature set; report whether a result changed because it was retracted, lower on the declared hierarchy, or contradicted by other retrieved work. Retain the relevance-only result for audit rather than silently suppressing it. This makes CatBoost/XGBoost and the evidence layer a measurable, inspectable contribution.

These are project differentiators, not clinical decision support. Publication type is imperfect metadata and is not a universal measure of methodological quality; a visible, configurable policy is preferable to treating its weight as truth.

## What current systems establish

| System / product | What it establishes | Consequence for this project |
|---|---|---|
| MedRAG + MIRAGE (2024) | MIRAGE has 7,663 questions from five medical QA sets; MedRAG evaluates corpora, retrievers and LLMs and shows that medical RAG configuration materially affects QA accuracy. | Use it as a medical-RAG baseline reference, not as evidence that retrieval scores demonstrate safety or evidence quality. It does not remove the need for claim-level provenance and calibration evaluation. |
| PaperQA2 | Open scientific-RAG package for PDFs and other documents; its documented workflow includes retrieval, evidence gathering, scored summaries and final answer. It also documents metadata/retraction checking during ingestion. | Do not market “scientific RAG with citations” or “a retraction check” as novel. Add auditable use of status in ranking/generation and direct adversarial evaluation. |
| OpenScholar (Nature, 2026) | Fully open scientific literature synthesis system with a 45M-paper open-access store, specialized retrievers and iterative self-feedback. ScholarQABench contains 2,967 expert-written queries and 208 long-form answers across four domains including biomedicine. | It is a strong current reference for citation-faithful multi-paper synthesis. Its 8B generator and 45M store are outside this project’s local machine envelope; borrow evaluation ideas, not the runtime stack. |
| Elicit-style workflows | Elicit’s current product documents search, screening, extraction and report production for systematic-review workflows. | The UX precedent is structured evidence extraction. The local project should expose its machine-readable ledger/API rather than attempt a proprietary review product. |
| Consensus-style claim answers | Product category expectation is rapid paper-backed, directional evidence summaries. | The proposed project can be more rigorous by making direction, exclusions, uncertainty and study policy explicit and testable rather than presenting an opaque “consensus.” |

### Retraction status is necessary but insufficient

NLM states that `Retracted Publication [PT]` is assigned to the retracted article and that retraction notices are linked citations. PubMed metadata therefore provides a reliable *first-party input* for a compact PubMed corpus. Store both (1) the article’s own publication type/status and (2) linked retraction/erratum records plus the fetch date. Do not infer retraction from title text.

A retraction filter alone has known limitations: indexing can lag; a paper may be corrected rather than retracted; and excluding it without recording the reason makes later review impossible. The proposed ledger should retain an excluded candidate with `excluded_reason=retracted` and forbid it from supporting a generated biomedical claim by default. An explicit research/audit mode may surface it only with a conspicuous label and linked notice.

## Open evaluation data: fit and limits

| Resource | Open/use | Best use here | Important limit |
|---|---|---|---|
| [BEIR](https://github.com/beir-cellar/beir): [SciFact](https://github.com/allenai/scifact) | Public download; SciFact is scientific claim verification over 5,183 abstracts and includes support/refute evidence annotations. | Primary shared testbed for retrieval and passage/abstract-level support-vs-refute stance. It is the best fit for the evidence layer among the current spec’s datasets. | Claims/corpus are limited and not a real-time PubMed feed. Hold out its test split from any stance calibration/tuning. |
| [BEIR NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | Public; 323 queries, 3.6K corpus documents in BEIR’s biomedical-retrieval task. | Retrieval and LTR evaluation with train/dev/test splits. | Relevance, not evidence direction or study quality. It cannot validate stance, retraction handling or answerability. |
| [TREC-COVID / BEIR](https://huggingface.co/datasets/BeIR/trec-covid) | Public; 50 test queries, ~171K CORD-19 corpus documents. | Optional stress test for a larger biomedical/scientific corpus when memory permits. | Test-only and much larger than the project’s small-corpus operating target; not appropriate for local routine runs under the current memory constraint. |
| [BioASQ](https://bioasq.org/) | Biomedical semantic indexing/QA challenge with its own terms and data access route. | Later external validity check for biomedical retrieval. | Verify the current task-specific licence/download conditions before including it in automated CI or redistributing fixtures. |
| [MedNLI](https://physionet.org/content/mednli/) | Clinical NLI dataset through PhysioNet credentialed access. | Optional transfer baseline only. | Clinical note inference is not paper-abstract stance; access requirements mean it must not be a default reproducible dependency. |
| PubMed E-utilities | Public API with rate limits and record metadata. | Demo-corpus metadata, publication types and linked retraction notices. | Metadata is not a study-quality adjudication; save raw response/provenance to allow refresh and audit. |

Evaluation addition recommended: create a small, human-reviewed *safety challenge* only after Phase 1 infrastructure works. It should contain real PMIDs and pre-declared labels for retracted, corrected/erratum, high-/low-priority publication types, and support/refute/insufficient evidence. Keep it separate from retrieval qrels; report reviewer agreement and do not call it a clinical gold standard.

## Models and the actual resource envelope

The team brief is the operational constraint: about 2 GB free RAM and prior WSL crashes mean no model-load/indexing run now. The SPEC’s RTX 3060 6 GB and <=1B-model target apply only after human approval and with one model loaded at a time.

| Component | Candidate | Why it fits the intended later design | Constraint / validation gate |
|---|---|---|---|
| Sparse retrieval | `bm25s` BM25 | No neural model; strong transparent baseline and low operational risk. | Can be built only on small corpora while memory restriction stands. |
| Dense retriever | [NCBI MedCPT query/article encoders](https://huggingface.co/ncbi/MedCPT-Query-Encoder), [article encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) | Public PubMedBERT-base dual encoders trained from PubMed search logs; intended for zero-shot biomedical IR. | BERT-base class (<1B) but still requires torch/RAM. Do not load now; benchmark against BM25 on a small fixed corpus once approved. |
| Reranker | [NCBI MedCPT Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder) | Domain-specific cross-encoder companion to MedCPT. | Run only over a small top-k and one model at a time; evaluate latency/memory in the real environment, do not assert fit from parameter count alone. |
| Stance / NLI baseline | [PubMedBERT fine-tuned on MNLI](https://huggingface.co/lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli) and optionally [PubMedBERT MNLI+MedNLI](https://huggingface.co/pritamdeka/PubMedBERT-MNLI-MedNLI) | Both expose entailment/neutral/contradiction labels and are small enough in model-family terms for the <=1B requirement. | Community checkpoints are baselines, not validated biomedical evidence judges. Map labels only after checking label order; calibrate on development data and evaluate untouched SciFact test data. |
| LTR / answerability | CatBoost and XGBoost | CPU-friendly models over saved retrieval/evidence features; directly serves the stated portfolio requirement. | Prevent leakage: split by query and keep test qrels/test safety cases untouched during feature-policy selection and calibration. |

For each neural candidate, record exact revision, tokenizer, device, batch size, peak RAM/VRAM and elapsed time in MLflow. “Fits” is an empirical result, not a design assumption.

## Unsolved gaps relevant to this build

1. **Citation correctness is not evidentiary sufficiency.** A citation can be real and relevant yet support neither the exact claim nor the direction/magnitude stated. OpenScholar’s citation-accuracy progress makes this gap more visible, not solved for a compact PubMed QA system.
2. **Study design is structured but nuanced.** PubMed publication types are useful features, but studies may have multiple types, protocol/method detail is often absent from abstracts, and hierarchy varies by question (etiology, diagnosis, intervention, mechanistic biology). A policy must be question-type-aware eventually; Phase 3 should begin with a documented conservative mapping and emit `unknown` rather than overclaim.
3. **Stance models do not equal evidence synthesis.** SciFact assesses claim–abstract support/refute, whereas biomedical questions often require PICO constraints, effect size, population and certainty. NLI should be a triage signal with thresholded uncertainty, not a sole arbiter.
4. **Retraction state needs time-aware provenance.** A cached article can later change status. Persist `metadata_fetched_at` and refresh before generation/export; evaluate both exclusion and correct visible flagging.
5. **Ranking optimisation can hide safety trade-offs.** nDCG/MRR alone can reward a retracted or weak-design result. Safety metrics need to be reported alongside relevance rather than folded into a single opaque score.
6. **Abstention labels are expensive.** Retrieval features can model answerability, but realistic “enough evidence” labels need a pre-declared human rubric. Start with an explicitly weak proxy only if labelled as such; do not call it clinical calibration.

## Concrete proposals for `docs/SPEC.md` (not applied silently)

See the separately appended `## Research proposals (2026-09-30; not yet adopted)` section in `docs/SPEC.md`. It proposes a claim-evidence ledger, a versioned retraction policy, an evidence-safety evaluation set, and explicit non-clinical scope. The existing architecture and phase commitments are unchanged.

## Primary sources

1. Xiong et al. (2024), [Benchmarking Retrieval-Augmented Generation for Medicine / MedRAG and MIRAGE](https://aclanthology.org/2024.findings-acl.372/), DOI: 10.18653/v1/2024.findings-acl.372.
2. Asai et al. (2026), [OpenScholar](https://www.nature.com/articles/s41586-025-10072-4), DOI: 10.1038/s41586-025-10072-4; [official code](https://github.com/AkariAsai/OpenScholar).
3. Future House, [PaperQA2 repository and documentation](https://github.com/Future-House/paper-qa).
4. Jin et al. (2023), [MedCPT official repository](https://github.com/ncbi/MedCPT) and [Bioinformatics article](https://doi.org/10.1093/bioinformatics/btad651).
5. Wadden et al. (2020), [SciFact paper](https://arxiv.org/abs/2004.14974) and [official data/code](https://github.com/allenai/scifact).
6. Thakur et al. (2021), [BEIR](https://arxiv.org/abs/2104.08663) and [implementation/data index](https://github.com/beir-cellar/beir).
7. National Library of Medicine, [Errata, Retractions, and Other Linked Citations in PubMed](https://www.nlm.nih.gov/bsd/policy/errata.html).
8. Elicit Help Center, [Systematic Reviews in Elicit](https://support.elicit.com/en/articles/14759154-systematic-reviews-in-elicit).
9. [MedNLI dataset page](https://physionet.org/content/mednli/) (access conditions) and cited public NLI checkpoints in the model table.
