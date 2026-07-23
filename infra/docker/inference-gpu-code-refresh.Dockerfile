# Reuse an already verified GPU worker when only application code has changed.
# The caller must pin GMONEY_GPU_BASE_IMAGE to the exact prior release image and
# must use the full inference-gpu.Dockerfile whenever dependencies change.
ARG GMONEY_GPU_BASE_IMAGE
FROM ${GMONEY_GPU_BASE_IMAGE}

USER root
WORKDIR /app

RUN rm -rf /app/src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .

USER modelworker
ENTRYPOINT ["python", "-m", "gmoney.inference.benchmark"]
