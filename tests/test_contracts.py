from decimal import Decimal

import pytest
from pydantic import ValidationError

from gmoney.contracts.evidence import Point, Polygon, TransformChain
from gmoney.contracts.gold import GoldAnnotation
from gmoney.settings import Settings


def test_gold_annotation_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GoldAnnotation.model_validate(
            {
                "bill_file": "bill.pdf",
                "rows": [{"page_number": 1, "description": "Test", "amount": 1}],
                "legacy_schema": "must-not-cross-boundary",
            }
        )


def test_transform_requires_three_by_three_matrices() -> None:
    polygon = Polygon(points=(Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1)))
    assert len(polygon.points) == 3
    with pytest.raises(ValidationError):
        TransformChain(
            page_number=1,
            source_width=100,
            source_height=100,
            derived_width=100,
            derived_height=100,
            forward_matrix=((1, 0), (0, 1)),
            inverse_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )


def test_gold_amount_is_decimal() -> None:
    annotation = GoldAnnotation.model_validate(
        {
            "bill_file": "bill.pdf",
            "rows": [{"page_number": 1, "description": "Test", "amount": "10.25"}],
        }
    )
    assert annotation.rows[0].amount == Decimal("10.25")


def test_settings_accepts_standard_gemini_key_without_exposing_it(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key == "secret-key"
    assert "secret-key" not in repr(settings)
