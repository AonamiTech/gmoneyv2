from gmoney.contracts.evidence import OcrToken, Point, Polygon
from gmoney.extraction.hospital import _clean_name, detect_hospital


def token(index: int, text: str, box: tuple[float, float, float, float]) -> OcrToken:
    left, top, right, bottom = box
    return OcrToken(
        token_id=f"header-{index}",
        page_number=1,
        text=text,
        confidence=0.98,
        polygon=Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        ),
        artifact_sha256="a" * 64,
        model_name="fixture",
        model_version="1",
    )


def test_detects_uppercase_hospital_header_with_evidence() -> None:
    identity = detect_hospital(
        (
            token(0, "50", (100, 80, 180, 150)),
            token(1, "50 years of Clínical Excellence", (600, 80, 1100, 110)),
            token(2, "VIJAYA GROUP OF HOSPITALS", (600, 130, 1250, 180)),
            token(3, "BILL OF SUPPLY", (600, 260, 1000, 300)),
        ),
        page_width=1600,
        page_height=2400,
    )
    assert identity is not None
    assert identity["name"] == "Vijaya Group of Hospitals"
    assert "header-2" in identity["evidence"]["token_ids"]


def test_joins_split_hospital_header_and_rejects_invoice_metadata() -> None:
    identity = detect_hospital(
        (
            token(0, "Dr.Kamakshi", (100, 80, 360, 125)),
            token(1, "Memorial Hospitals Pvt. Ltd.", (100, 135, 650, 180)),
            token(2, "Credit Invoice", (100, 280, 400, 320)),
            token(3, "Patient Name", (100, 340, 350, 380)),
        ),
        page_width=1200,
        page_height=2000,
    )
    assert identity is not None
    assert identity["name"] == "Dr.Kamakshi Memorial Hospitals Pvt. Ltd."
    assert identity["evidence"]["token_ids"] == ["header-0", "header-1"]


def test_does_not_promote_patient_or_invoice_text_as_hospital() -> None:
    assert (
        detect_hospital(
            (
                token(0, "Patient Name", (100, 80, 350, 120)),
                token(1, "Hospital Bill No 1234", (100, 140, 500, 180)),
            ),
            page_width=1200,
            page_height=2000,
        )
        is None
    )


def test_cleans_joined_brand_unit_tagline_and_duplicate_logo_text() -> None:
    assert (
        _clean_name(
            "RUBAN RUBAN MEMORIÁL HOSPITAL redefining health "
            "(A Unit of Ruban Patliputra Hospital Pvt. Ltd.)"
        )
        == "Ruban Memoriál Hospital"
    )
    assert (
        _clean_name(
            "RUBAN (A Unit of Ruban Patliputra Hospital Pvt. Ltd.) "
            "NABH 19 Patliputra Colony"
        )
        == "Ruban Patliputra Hospital Pvt. Ltd."
    )
    assert (
        _clean_name("Quality CARE India Limited CARE Hospital Banjara Hills")
        == "CARE Hospital Banjara Hills"
    )
