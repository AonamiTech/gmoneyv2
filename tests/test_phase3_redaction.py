from pathlib import Path

import cv2
import numpy as np

from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.inference.redaction import redact_crop


def token(token_id: str, text: str, box: tuple[int, int, int, int]) -> OcrToken:
    left, top, right, bottom = box
    return OcrToken(
        token_id=token_id,
        page_number=1,
        text=text,
        confidence=0.99,
        polygon=Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        ),
        artifact_sha256="a" * 64,
        model_name="test",
        model_version="1",
    )


def write_image(path: Path) -> None:
    assert cv2.imwrite(str(path), np.full((100, 200, 3), 255, dtype=np.uint8))


def test_redaction_masks_identifiers_and_sanitizes_tokens(tmp_path) -> None:
    source = tmp_path / "crop.png"
    output = tmp_path / "masked.png"
    write_image(source)
    result = redact_crop(
        source,
        output,
        (
            token("label", "Patient Name", (20, 20, 70, 35)),
            token("name", "Ada Lovelace", (75, 20, 140, 35)),
            token("service", "Blood Test", (20, 60, 80, 75)),
            token("amount", "120.00", (150, 60, 190, 75)),
        ),
        (10, 10, 210, 110),
    )
    assert result.safe
    assert set(result.masked_token_ids) == {"label", "name"}
    sanitized = {item.token_id: item.text for item in result.tokens}
    assert sanitized["label"] == sanitized["name"] == "[REDACTED]"
    assert sanitized["service"] == "Blood Test"
    image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert image is not None
    assert image[15, 15].tolist() == [0, 0, 0]


def test_redaction_fails_closed_without_ocr_coverage(tmp_path) -> None:
    source = tmp_path / "crop.png"
    write_image(source)
    result = redact_crop(source, tmp_path / "masked.png", (), (0, 0, 200, 100))
    assert not result.safe
    assert result.reasons == ("no_ocr_coverage",)


def test_redaction_masks_sensitive_value_on_following_line(tmp_path) -> None:
    source = tmp_path / "crop.png"
    write_image(source)
    result = redact_crop(
        source,
        tmp_path / "masked.png",
        (
            token("label", "Patient Name", (20, 10, 90, 22)),
            token("name", "Ada Lovelace", (20, 30, 100, 42)),
            token("service", "Blood Test", (20, 70, 90, 82)),
        ),
        (0, 0, 200, 100),
    )
    assert set(result.masked_token_ids) == {"label", "name"}
