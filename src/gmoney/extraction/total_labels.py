"""Printed total/sub-total label vocabulary shared by validation and reconciliation."""

from __future__ import annotations

import re

TOTAL_LABELS = frozenset(
    {
        "bill amount",
        "bill total",
        "total",
        "totals",
        "sub total",
        "subtotal",
    }
)
TOTAL_PREFIXES = (
    "grand total",
    "gross bill amount",
    "net bill amount",
    "net medical amount",
    "net payable",
    "total bill amount",
    "total gross bill value",
    "total payable amount",
)


def normalized_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def is_section_subtotal_label(value: str) -> bool:
    return value in {"bill total", "sub total", "subtotal"} or value.startswith(
        ("sub total ", "subtotal ")
    )


def is_total_label(value: str) -> bool:
    """Return whether a normalized label names any printed total."""
    return value in TOTAL_LABELS or value.startswith(TOTAL_PREFIXES)
