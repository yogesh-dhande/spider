FROM python:3.12-slim

ARG SPIDER_SOURCE_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPIDER_SOURCE_COMMIT=${SPIDER_SOURCE_COMMIT} \
    SPIDER_SOURCE_DIRTY=false

WORKDIR /workspace

COPY requirements/agent-pipeline.txt requirements/agent-pipeline.txt
RUN python -m pip install --no-cache-dir -r requirements/agent-pipeline.txt

COPY pyproject.toml README.md ./
COPY src src
RUN python -m pip install --no-cache-dir --no-deps .

COPY configs configs

ENTRYPOINT ["spider-study"]
