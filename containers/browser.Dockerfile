FROM mcr.microsoft.com/playwright/python:v1.61.0-noble

ARG SPIDER_SOURCE_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPIDER_SOURCE_COMMIT=${SPIDER_SOURCE_COMMIT} \
    SPIDER_SOURCE_DIRTY=false

WORKDIR /workspace

COPY requirements/agent-pipeline.txt requirements/agent-pipeline.txt
COPY requirements/agent-browser.txt requirements/agent-browser.txt
RUN python -m pip install --no-cache-dir \
    -r requirements/agent-pipeline.txt \
    -r requirements/agent-browser.txt

COPY pyproject.toml README.md ./
COPY src src
RUN python -m pip install --no-cache-dir --no-deps .

COPY configs configs

ENTRYPOINT ["spider-study"]
