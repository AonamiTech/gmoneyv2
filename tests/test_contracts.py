from decimal import Decimal

import pytest
from pydantic import ValidationError

from gmoney.contracts.evidence import Point, Polygon, TransformChain
from gmoney.contracts.extraction import SourceCell, SourceColumn, SourceTable
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


def test_non_empty_source_cell_requires_token_grounding() -> None:
    with pytest.raises(ValidationError, match="grounded OCR evidence"):
        SourceCell(column_id="c1", raw_value="invented")


@pytest.mark.parametrize("token_id", ["", "   "])
def test_non_empty_source_cell_rejects_blank_token_ids(token_id: str) -> None:
    with pytest.raises(ValidationError, match="non-blank"):
        SourceCell.model_validate(
            {
                "column_id": "c1",
                "raw_value": "invented",
                "evidence": [
                    {
                        "page_number": 1,
                        "polygon": {
                            "points": [
                                {"x": 1, "y": 1},
                                {"x": 10, "y": 1},
                                {"x": 10, "y": 10},
                                {"x": 1, "y": 10},
                            ]
                        },
                        "artifact_sha256": "a" * 64,
                        "token_ids": [token_id],
                    }
                ],
            }
        )


def test_real_source_header_requires_token_grounding() -> None:
    with pytest.raises(ValidationError, match="grounded OCR evidence"):
        SourceColumn(id="c1", label="Amount", order=0)


def test_synthetic_source_header_is_explicitly_flagged() -> None:
    column = SourceColumn(
        id="c1",
        label="Column 1",
        order=0,
        validation_flags=("synthetic_header",),
    )
    assert column.evidence == ()


def test_source_table_requires_exactly_one_cell_per_column() -> None:
    grounded = {
        "page_number": 1,
        "table_id": "p1-t1",
        "polygon": {
            "points": [
                {"x": 1, "y": 1},
                {"x": 10, "y": 1},
                {"x": 10, "y": 10},
                {"x": 1, "y": 10},
            ]
        },
        "artifact_sha256": "a" * 64,
        "token_ids": ["token-1"],
    }
    with pytest.raises(ValidationError, match="exactly one cell per column"):
        SourceTable.model_validate(
            {
                "id": "p1-t1-s1",
                "page_number": 1,
                "table_id": "p1-t1",
                "columns": [
                    {
                        "id": "c1",
                        "label": "First",
                        "order": 0,
                        "evidence": [grounded],
                    },
                    {
                        "id": "c2",
                        "label": "Second",
                        "order": 1,
                        "evidence": [grounded],
                    },
                ],
                "rows": [
                    {
                        "id": "r1",
                        "order": 0,
                        "cells": [
                            {
                                "column_id": "c1",
                                "raw_value": "value",
                                "evidence": [grounded],
                            }
                        ],
                    }
                ],
            }
        )


def test_settings_accepts_standard_gemini_key_without_exposing_it(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key == "secret-key"
    assert "secret-key" not in repr(settings)
