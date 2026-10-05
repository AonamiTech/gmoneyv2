# TeleOCR table reader (StarDoc-AI/TeleOCR weights, Apache-2.0) on one NVIDIA GPU.
# Weights are mounted read-only at /models/teleocr; nothing is downloaded at runtime.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TELEOCR_MODEL_PATH=/models/teleocr

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY services/teleocr/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "from importlib.metadata import version; assert version('transformers') == '4.57.1'"
COPY services/teleocr/teleocr_service.py ./teleocr_service.py

RUN useradd --create-home --uid 10001 modelworker
USER modelworker
EXPOSE 8112
CMD ["uvicorn", "teleocr_service:app", "--host", "0.0.0.0", "--port", "8112", "--workers", "1"]
