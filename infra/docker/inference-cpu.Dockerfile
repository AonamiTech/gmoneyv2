FROM python:3.12-slim

ARG GMONEY_BUILD_REVISION=unknown
LABEL org.opencontainers.image.revision=${GMONEY_BUILD_REVISION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_MODEL_SOURCE=BOS \
    GMONEY_BUILD_REVISION=${GMONEY_BUILD_REVISION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[inference-cpu]'

RUN useradd --create-home --uid 10001 modelworker
USER modelworker

ENTRYPOINT ["python", "-m", "gmoney.inference.benchmark"]
