from __future__ import annotations

import hashlib
from typing import Any

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.geometry.transform import Matrix, apply_matrix, identity


def paddle_ocr_tokens(
    output: dict[str, Any],
    page_number: int,
    artifact_sha256: str,
    crop_to_page: Matrix | None = None,
) -> tuple[OcrToken, ...]:
    pages = output.get("pages") or []
    if not pages:
        raise ValueError("OCR output contains no pages")
    result = pages[0].get("res") or {}
    polygons = result.get("rec_polys") or result.get("dt_polys") or []
    texts = result.get("rec_texts") or []
    scores = result.get("rec_scores") or []
    if not (len(polygons) == len(texts) == len(scores)):
        raise ValueError("OCR polygon, text, and score arrays differ in length")
    transform = crop_to_page or identity()
    tokens: list[OcrToken] = []
    for index, (polygon, text, score) in enumerate(zip(polygons, texts, scores, strict=True)):
        mapped = apply_matrix(transform, ((float(point[0]), float(point[1])) for point in polygon))
        token_hash = hashlib.sha256(
            f"{artifact_sha256}:{page_number}:{index}:{text}".encode()
        ).hexdigest()[:24]
        tokens.append(
            OcrToken(
                token_id=f"ocr-{token_hash}",
                page_number=page_number,
                text=str(text),
                confidence=float(score),
                polygon=Polygon(points=tuple(Point(x=max(0, x), y=max(0, y)) for x, y in mapped)),
                artifact_sha256=artifact_sha256,
                model_name="PP-OCRv6-medium",
                model_version="PP-OCRv6",
            )
        )
    return tuple(tokens)

