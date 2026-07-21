from __future__ import annotations

import hashlib
from uuid import NAMESPACE_URL, uuid5

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    EvidenceRef,
    PageType,
    ReviewDisposition,
    RowRole,
    TableType,
)
from gmoney.extraction.spatial import AlignedLedgerRow
from gmoney.extraction.typed_values import parse_service_date

BILLABLE_ROLES = {RowRole.DETAIL, RowRole.REFUND, RowRole.CATEGORY_ROLLUP}
PUBLISHED_ROLES = {*BILLABLE_ROLES, RowRole.INFORMATIONAL}


def _page_type(table_type: TableType) -> PageType:
    return {
        TableType.PHARMACY: PageType.PHARMACY,
        TableType.LABORATORY: PageType.LABORATORY,
        TableType.CATEGORY_SUMMARY: PageType.CATEGORY_SUMMARY,
        TableType.PACKAGE_SUMMARY: PageType.CATEGORY_SUMMARY,
        TableType.PAYMENT: PageType.RECEIPT_PAYMENT,
        TableType.METADATA: PageType.METADATA,
        TableType.ITEM_LEDGER: PageType.ITEMIZED_CHARGES,
    }.get(table_type, PageType.MIXED)


def _evidence_ref(
    *,
    page_number: int,
    table_id: str,
    artifact_sha256: str,
    box: tuple[float, float, float, float],
    token_ids: tuple[str, ...],
) -> EvidenceRef:
    left, top, right, bottom = box
    return EvidenceRef(
        page_number=page_number,
        table_id=table_id,
        polygon=Polygon(
            points=(
                Point(x=left, y=top),
                Point(x=right, y=top),
                Point(x=right, y=bottom),
                Point(x=left, y=bottom),
            )
        ),
        artifact_sha256=artifact_sha256,
        token_ids=token_ids,
    )


def canonicalize_rows(
    document_id: str,
    page_number: int,
    table_id: str,
    artifact_sha256: str,
    aligned: tuple[AlignedLedgerRow, ...],
    starting_order: int = 0,
) -> tuple[CanonicalRow, ...]:
    """Publish billable and informational rows with field-level OCR evidence.

    Provider output is a proposal, never evidence. Optional provider fields are
    retained only when their own OCR tokens were aligned. This prevents a
    plausible but column-shifted VLM value from silently entering the ledger.
    """
    output: list[CanonicalRow] = []
    for row in aligned:
        candidate = row.candidate
        if candidate.role not in PUBLISHED_ROLES:
            continue
        if not candidate.description or not row.evidence_box:
            continue

        description_ids = row.field_token_ids.get("description", ())
        amount_ids = row.field_token_ids.get("amount", ())
        if not description_ids:
            continue
        if candidate.role in BILLABLE_ROLES and (candidate.amount is None or not amount_ids):
            continue
        if candidate.role is RowRole.INFORMATIONAL and not any(
            row.field_token_ids.get(field)
            for field in ("service_date", "request_no", "service_code", "hsn_code")
        ):
            continue
        all_token_ids = tuple(
            dict.fromkeys(
                token_id for token_ids in row.field_token_ids.values() for token_id in token_ids
            )
        )
        evidence = (
            _evidence_ref(
                page_number=page_number,
                table_id=table_id,
                artifact_sha256=artifact_sha256,
                box=row.evidence_box,
                token_ids=all_token_ids,
            ),
        )
        field_evidence = {
            field: (
                _evidence_ref(
                    page_number=page_number,
                    table_id=table_id,
                    artifact_sha256=artifact_sha256,
                    box=row.evidence_box,
                    token_ids=token_ids,
                ),
            )
            for field, token_ids in row.field_token_ids.items()
            if token_ids
        }
        identity = hashlib.sha256(
            f"{document_id}:{page_number}:{table_id}:{candidate.source_row}:"
            f"{','.join(all_token_ids)}".encode()
        ).hexdigest()[:24]

        quantity = candidate.quantity if "quantity" in field_evidence else None
        rate = candidate.rate if "rate" in field_evidence else None
        gross_amount = (
            candidate.gross_amount if "gross_amount" in field_evidence else None
        )
        discount = candidate.discount if "discount" in field_evidence else None
        service_date = candidate.service_date if "service_date" in field_evidence else None
        request_no = candidate.request_no if "request_no" in field_evidence else None
        service_code = candidate.service_code if "service_code" in field_evidence else None
        hsn_code = candidate.hsn_code if "hsn_code" in field_evidence else None

        output.append(
            CanonicalRow(
                id=uuid5(NAMESPACE_URL, f"gmoney:canonical-row:{identity}"),
                contract_version="canonical_row_v2",
                document_id=document_id,
                page_number=page_number,
                table_id=table_id,
                page_type=_page_type(candidate.table_type),
                table_type=candidate.table_type,
                row_order=starting_order + len(output),
                role=candidate.role,
                review_disposition=ReviewDisposition.ACCEPTED,
                section=candidate.category or candidate.section,
                description=candidate.description,
                service_date_raw=service_date,
                service_date_iso=parse_service_date(service_date),
                request_no=request_no,
                service_code=service_code,
                hsn_code=hsn_code,
                quantity_raw=str(quantity) if quantity is not None else None,
                quantity=quantity,
                unit_price_raw=str(rate) if rate is not None else None,
                unit_price=rate,
                gross_amount_raw=(str(gross_amount) if gross_amount is not None else None),
                gross_amount=gross_amount,
                discount_raw=str(discount) if discount is not None else None,
                discount=discount,
                net_amount_raw=(str(candidate.amount) if candidate.amount is not None else None),
                net_amount=candidate.amount,
                evidence=evidence,
                field_evidence=field_evidence,
                candidate_ids=(f"candidate-{identity}",),
                source_routes=row.source_routes,
                validation_flags=candidate.validation_flags,
            )
        )
    return tuple(output)
