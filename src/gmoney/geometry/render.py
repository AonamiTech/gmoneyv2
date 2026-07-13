from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz

from gmoney.contracts.evidence import PageAsset, PageQuality
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.quality import assess_quality

RENDERER_VERSION = fitz.VersionBind


@dataclass(frozen=True)
class RenderManifest:
    document_sha256: str
    source_path: str
    source_size_bytes: int
    dpi: int
    pages: tuple[PageAsset, ...]
    quality: tuple[PageQuality, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": "render_manifest_v1",
            "document_sha256": self.document_sha256,
            "source_path": self.source_path,
            "source_size_bytes": self.source_size_bytes,
            "dpi": self.dpi,
            "pages": [page.model_dump(mode="json") for page in self.pages],
            "quality": [quality.model_dump(mode="json") for quality in self.quality],
        }


def render_pdf(source: Path, artifact_root: Path, dpi: int = 300) -> RenderManifest:
    if dpi < 72 or dpi > 600:
        raise ValueError("render DPI must be between 72 and 600")
    source = source.resolve()
    before_hash = sha256_file(source)
    source_size = source.stat().st_size
    destination = artifact_root / before_hash / "render-v1" / f"{dpi}dpi"
    destination.mkdir(parents=True, exist_ok=True)
    pages: list[PageAsset] = []
    qualities: list[PageQuality] = []
    scale = dpi / 72
    with fitz.open(source) as document:
        for index, page in enumerate(document):
            page_number = index + 1
            output = destination / f"page-{page_number:04d}.png"
            if not output.exists():
                temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pixmap.save(temporary)
                temporary.replace(output)
            artifact_hash = sha256_file(output)
            with fitz.open(source) as verification_document:
                rect = verification_document[index].rect
            pages.append(
                PageAsset(
                    document_sha256=before_hash,
                    page_number=page_number,
                    artifact_sha256=artifact_hash,
                    relative_path=str(output.relative_to(artifact_root)),
                    width=round(rect.width * scale),
                    height=round(rect.height * scale),
                    dpi=dpi,
                    renderer="PyMuPDF",
                    renderer_version=RENDERER_VERSION,
                )
            )
            qualities.append(assess_quality(output, page_number, dpi))
    if sha256_file(source) != before_hash or source.stat().st_size != source_size:
        raise RuntimeError("source PDF changed during rendering")
    manifest = RenderManifest(
        document_sha256=before_hash,
        source_path=str(source),
        source_size_bytes=source_size,
        dpi=dpi,
        pages=tuple(pages),
        quality=tuple(qualities),
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return manifest
