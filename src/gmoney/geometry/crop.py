from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz

from gmoney.contracts.evidence import TransformChain
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.transform import compose, invert, translation


@dataclass(frozen=True)
class CropResult:
    output_path: Path
    artifact_sha256: str
    transform: TransformChain


@dataclass(frozen=True)
class PhotometricVariant:
    output_path: Path
    artifact_sha256: str


def crop_region(
    source: Path,
    output: Path,
    page_number: int,
    box: tuple[int, int, int, int],
) -> CropResult:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read page image: {source}")
    height, width = image.shape[:2]
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"crop box {box} is outside {width}x{height}")
    cropped = image[top:bottom, left:right]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), cropped):
        raise RuntimeError(f"failed to write crop: {temporary}")
    temporary.replace(output)
    forward = translation(-left, -top)
    transform = TransformChain(
        page_number=page_number,
        source_width=width,
        source_height=height,
        derived_width=right - left,
        derived_height=bottom - top,
        forward_matrix=forward,
        inverse_matrix=invert(forward),
        operations=(f"crop:{left},{top},{right},{bottom}",),
    )
    return CropResult(
        output_path=output,
        artifact_sha256=sha256_file(output),
        transform=transform,
    )


def render_pdf_region(
    source: Path,
    output: Path,
    page_number: int,
    box: tuple[int, int, int, int],
    *,
    source_dpi: int = 300,
    output_dpi: int = 400,
) -> CropResult:
    """Rerender one PDF region at higher resolution with an invertible transform."""
    document = fitz.open(source)
    try:
        page = document[page_number - 1]
        scale_to_points = 72 / source_dpi
        left, top, right, bottom = box
        clip = fitz.Rect(
            left * scale_to_points,
            top * scale_to_points,
            right * scale_to_points,
            bottom * scale_to_points,
        )
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(output_dpi / 72, output_dpi / 72),
            clip=clip,
            alpha=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
        pixmap.save(temporary)
        temporary.replace(output)
        page_rect = page.rect
        source_width = round(page_rect.width * source_dpi / 72)
        source_height = round(page_rect.height * source_dpi / 72)
    finally:
        document.close()
    scale = output_dpi / source_dpi
    scale_matrix = ((scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, 1.0))
    forward = compose(translation(-left, -top), scale_matrix)
    transform = TransformChain(
        page_number=page_number,
        source_width=source_width,
        source_height=source_height,
        derived_width=pixmap.width,
        derived_height=pixmap.height,
        forward_matrix=forward,
        inverse_matrix=invert(forward),
        operations=(f"crop:{left},{top},{right},{bottom}", f"render_dpi:{output_dpi}"),
    )
    return CropResult(output, sha256_file(output), transform)


def clahe_variant(source: Path, output: Path) -> PhotometricVariant:
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read crop: {source}")
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), enhanced):
        raise RuntimeError(f"failed to write CLAHE crop: {temporary}")
    temporary.replace(output)
    return PhotometricVariant(output, sha256_file(output))


def color_overlay_suppressed_variant(
    source: Path,
    output: Path,
) -> PhotometricVariant:
    """Fade chromatic overlays while retaining neutral dark printed text."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read crop: {source}")
    suppressed = image.max(axis=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    if not cv2.imwrite(str(temporary), suppressed):
        raise RuntimeError(f"failed to write color-suppressed crop: {temporary}")
    temporary.replace(output)
    return PhotometricVariant(output, sha256_file(output))
