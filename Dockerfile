FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.7

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "evidence_rag.service:app", "--host", "0.0.0.0", "--port", "8000"]
