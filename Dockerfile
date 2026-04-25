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
COPY train.py ./

RUN pip install --no-cache-dir -e . \
    && python scripts/generate_level1_dataset.py

ENV DECEIT_GRADER_CACHE=/tmp/deceit_grader_cache.json

CMD ["python", "train.py"]