from __future__ import annotations

import re
import unicodedata

HEADER_NORMALIZER_VERSION = "header_normalizer_v1"
HOSPITAL_NORMALIZER_VERSION = "hospital_name_normalizer_v1"


def _ascii_tokens(value: str) -> str:
    compatible = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", compatible))


def normalize_header(value: str) -> str:
    """Normalize trained and runtime headers with one versioned algorithm."""
    return _ascii_tokens(value)


def normalize_hospital_name(value: str) -> str:
    """Normalize hospital identity separately so its policy can evolve independently."""
    return _ascii_tokens(value)
