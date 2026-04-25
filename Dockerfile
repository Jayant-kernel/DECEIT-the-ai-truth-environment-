FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -e . \
    && python scripts/generate_level1_dataset.py

ENV DECEIT_GRADER_CACHE=/tmp/deceit_grader_cache.json

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "deceit_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]