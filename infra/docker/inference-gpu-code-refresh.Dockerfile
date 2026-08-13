# Reuse an already verified GPU worker when only application code has changed.
# The caller must pin GMONEY_GPU_BASE_IMAGE to the exact prior release image and
# must use the full inference-gpu.Dockerfile whenever dependencies change.
ARG GMONEY_GPU_BASE_IMAGE
FROM ${GMONEY_GPU_BASE_IMAGE}

ARG GMONEY_BUILD_REVISION=unknown
ARG GMONEY_REQUIRE_BUILD_REVISION=0
LABEL org.opencontainers.image.revision=${GMONEY_BUILD_REVISION}

USER root
WORKDIR /app

RUN rm -rf /app/src
COPY pyproject.toml README.md ./
COPY src ./src
COPY infra/docker/write-release-manifest.sh /tmp/write-release-manifest.sh
RUN sh /tmp/write-release-manifest.sh \
    && rm /tmp/write-release-manifest.sh \
    && pip install --no-cache-dir --no-deps --force-reinstall .

USER modelworker
ENTRYPOINT ["python", "-m", "gmoney.inference.benchmark"]
