from __future__ import annotations

import hashlib

from gmoney.contracts.evidence import Point, Polygon
from gmoney.contracts.extraction import (
    CanonicalRow,
    EvidenceRef,
    ReviewDisposition,
    RowRole,
)
from gmoney.extraction.spatial import AlignedLedgerRow


def canonicalize_rows(
    document_id: str,
    page_number: int,
    table_id: str,
    artifact_sha256: str,
    aligned: tuple[AlignedLedgerRow, ...],
    starting_order: int = 0,
) -> tuple[CanonicalRow, ...]:
    output: list[CanonicalRow] = []
    for offset, row in enumerate(aligned):
        if row.candidate.role not in {RowRole.DETAIL, RowRole.REFUND}:
            continue
        if not row.candidate.description or row.candidate.amount is None:
            continue
        evidence: tuple[EvidenceRef, ...] = ()
        if row.evidence_box:
            left, top, right, bottom = row.evidence_box
            evidence = (
                EvidenceRef(
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
                    token_ids=row.evidence_token_ids,
                ),
            )
        accepted = bool(
            evidence
            and row.grounding_ratio >= 0.75
        )
        identity = hashlib.sha256(
            f"{document_id}:{page_number}:{table_id}:{row.candidate.source_row}".encode()
        ).hexdigest()[:24]
        quantity_raw = str(row.candidate.quantity) if row.candidate.quantity is not None else None
        rate_raw = str(row.candidate.rate) if row.candidate.rate is not None else None
        discount_raw = str(row.candidate.discount) if row.candidate.discount is not None else None
        amount_raw = str(row.candidate.amount) if row.candidate.amount is not None else None
        output.append(
            CanonicalRow(
                contract_version="canonical_row_v1",
                document_id=document_id,
                page_number=page_number,
                table_id=table_id,
                row_order=starting_order + offset,
                role=row.candidate.role,
                review_disposition=(
                    ReviewDisposition.ACCEPTED if accepted else ReviewDisposition.PENDING
                ),
                description=row.candidate.description,
                service_date_raw=row.candidate.service_date,
                request_no=row.candidate.request_no,
                service_code=row.candidate.service_code,
                hsn_code=row.candidate.hsn_code,
                quantity_raw=quantity_raw,
                quantity=row.candidate.quantity,
                unit_price_raw=rate_raw,
                unit_price=row.candidate.rate,
                discount_raw=discount_raw,
                discount=row.candidate.discount,
                net_amount_raw=amount_raw,
                net_amount=row.candidate.amount,
                evidence=evidence,
                candidate_ids=(f"candidate-{identity}",),
                validation_flags=(() if accepted else ("insufficient_grounding",)),
            )
        )
    return tuple(output)
