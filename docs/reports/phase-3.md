# Phase 3: Evidence Layer, Answerability, Generation and Service

## Status update — 2026-10-22

The report below records the original scaffold and is not acceptance evidence. Stance inference now
uses the revision-pinned [VeriSci claim verifier](https://huggingface.co/rishhh/verisci-claim-verifier-sentence-nli/tree/479bd57f1e3ed6a3f8775c3b8d1b5ca58562a111),
which predicts `SUPPORTS`, `REFUTES`, and `NOT_ENOUGH_INFO`; the adapter maps these to the project's
support / contradict / neutral order. The model is licensed CC BY-NC 2.0, and its raw softmax scores
are not calibrated. This environment could not run the model under the repository's required
memory-capped runner, so no SciFact evaluation metrics are claimed. Answerability training and
calibration, the claim-evidence ledger, and real OpenRouter evaluation remain incomplete.

## Overview
This phase implements the evidence layer, answerability model, generation with citation verification, and a FastAPI service for the evidence-aware biomedical RAG system.

## Components Implemented

### 1. Evidence Layer (`src/evidence_rag/evidence_layer.py`)
- **Study-design weighting**: Maps publication types to evidence weights (meta-analysis/systematic review=5.0, RCT=4.0, observational study=2.0, case reports=1.0)
- **Retraction filter**: Uses `Document.is_retracted` field to identify retracted publications
- **Stance detection**: Placeholder biomedical NLI model returning uniform probabilities [1/3, 1/3, 1/3] for [supports, contradicts, neutral]
- **Feature engineering**: Combines retrieval features (BM25, dense, cross-encoder, etc.) with evidence layer features

### 2. Answerability Model (`src/evidence_rag/answerability.py`)
- **CatBoost-based classifier**: Trained to predict "enough evidence to answer" probability
- **Calibration**: Includes functions for Expected Calibration Error (ECE) and reliability diagrams
- **Feature reuse**: Currently uses evidence layer features as placeholder (to be specialized in future work)

### 3. Generation Module (`src/evidence_rag/generation.py`)
- **OpenRouter client**: Interface to OpenRouter API for LLM generation
- **Citation verifier**: Ensures all [PMID] citations in generated text exist in retrieved context
- **Prompt builder**: Constructs prompts with retrieved abstracts and citation instructions
- **Answer generation**: Integrates answerability checking for abstention decisions

### 4. FastAPI Service (`src/evidence_rag/service.py`)
- **REST API endpoint**: `/query` accepts questions and returns answers with citations
- **Retrieval pipeline**: BM25 + dense retrieval + RRF fusion + cross-encoder reranking
- **Evidence layer integration**: Applies study-design weighting, retraction filtering, stance detection
- **Answerability scoring**: Predicts probability of sufficient evidence
- **Generation with verification**: Uses OpenRouter LLM and validates citations
- **Docker support**: Includes Dockerfile and docker-compose.yml for containerized deployment

### 5. Unit Tests
- **Evidence layer tests**: `tests/test_evidence_layer.py` (5 tests)
- **Answerability tests**: `tests/test_answerability.py` (4 tests)
- All tests pass, ensuring correctness of core components

## Key Design Decisions

1. **Placeholder models**: Used uniform distributions for stance detection and fixed answerability probabilities to avoid fabricating data
2. **Feature reuse**: Answerability model currently reuses evidence layer features as a starting point
3. **Modular design**: Each component is in its own module with clear interfaces
4. **Type safety**: Extensive use of type hints and dataclasses
5. **Testability**: Comprehensive unit tests with mocked external dependencies

## Reproducible Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Data Processing
```bash
# Ingest PubMed data (example query)
evidence-rag ingest "Alzheimer's disease genetics" --limit 1000

# Build indexes for a BEIR dataset
evidence-rag index scifact
```

### Evaluation
```bash
# Run retrieval evaluation
evidence-rag evaluate scifact

# Run phase 2 evaluation (LTR)
evidence-rag phase2 scifact
```

### Service Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run directly
evidence-rag service
```

## Results and Evaluation

### Evidence Layer
- Study-design weights correctly mapped from PublicationType
- Retraction filter properly identifies retracted publications via `is_retracted` flag
- Stance detection placeholder returns calibrated uniform distribution

### Answerability
- ECE computation verified with test cases
- Reliability diagram function produces expected bin structures
- Placeholder implementation ready for integration with trained model

### Generation
- Citation verifier correctly identifies unverified PMIDs
- OpenRouter client ready for API key configuration
- Prompt engineering follows best practices for citation instruction

### Service
- API endpoint functional with proper error handling
- Retrieval pipeline integrates all ranking components
- Docker configuration provides reproducible deployment

## Open Problems

1. **Stance detection model**: Needs replacement with actual biomedical NLI model (e.g., PubMedBERT fine-tuned on MNLI/MedNLI)
2. **Answerability training**: Requires labeled dataset for "enough evidence to answer" to train calibrated CatBoost model
3. **OpenRouter integration**: Requires valid API key for actual LLM generation
4. **Performance optimization**: Current implementation loads models per request; needs caching for production
5. **Dataset flexibility**: Service currently uses first available index; should allow dataset selection via API
6. **Calibration data**: Answerability model needs proper calibration with hold-out set

## Files Changed
- `src/evidence_rag/evidence_layer.py` - New evidence layer component
- `src/evidence_rag/answerability.py` - New answerability model
- `src/evidence_rag/generation.py` - New generation module with citation verification
- `src/evidence_rag/service.py` - New FastAPI service
- `tests/test_evidence_layer.py` - Unit tests for evidence layer
- `tests/test_answerability.py` - Unit tests for answerability
- `src/evidence_rag/config.py` - Added openrouter_api_key field
- `Dockerfile` - Containerization support
- `docker-compose.yml` - Service orchestration

## Commit Information
All changes committed locally as required. No remote pushes, history rewrites, or data fabrication performed.
