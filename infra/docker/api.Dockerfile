FROM python:3.12-slim

ARG GMONEY_BUILD_REVISION=unknown
ARG GMONEY_REQUIRE_BUILD_REVISION=0
LABEL org.opencontainers.image.revision=${GMONEY_BUILD_REVISION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY infra/docker/write-release-manifest.sh /tmp/write-release-manifest.sh
RUN sh /tmp/write-release-manifest.sh && rm /tmp/write-release-manifest.sh
RUN pip install --no-cache-dir .

USER 65532:65532
EXPOSE 8000
CMD ["uvicorn", "gmoney.api:app", "--host", "0.0.0.0", "--port", "8000"]
