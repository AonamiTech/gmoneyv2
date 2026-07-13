from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from gmoney.contracts.evidence import TransformChain
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.transform import invert, translation


@dataclass(frozen=True)
class CropResult:
    output_path: Path
    artifact_sha256: str
    transform: TransformChain


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
