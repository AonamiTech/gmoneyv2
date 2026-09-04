"""Auditable UVDoc shadow inference with exact backward-grid capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import Field, model_validator

from gmoney.contracts.common import ContractModel
from gmoney.contracts.v6 import DenseBackwardGridMapping, canonical_sha256
from gmoney.evaluation.corpus import sha256_file
from gmoney.geometry.dense import DenseGridDiagnostics, analyze_dense_grid, write_dense_grid

SHA256_PATTERN = r"^[a-f0-9]{64}$"
UVDOC_ADAPTER_VERSION = "gmoney_uvdoc_shadow_v1"
UVDOC_MODEL_REPOSITORY = "PaddlePaddle/UVDoc_safetensors"
UVDOC_MODEL_REVISION = "7b8c629d7a15656889d0b21c73df206ac8a732b5"
EXPECTED_VERSIONS = {
    "paddle": "3.2.2",
    "paddleocr": "3.7.0",
    "paddlex": "3.7.2",
}
EXPECTED_MODEL_CONFIG = {
    "model_type": "uvdoc",
    "num_filter": 32,
    "in_channels": 3,
    "kernel_size": 5,
    "block_stride_values": [1, 2, 2],
    "feature_map_multipliers": [1, 2, 4],
    "block_counts_per_stage": [3, 4, 6],
    "dilation_values": [
        [1],
        [2],
        [5],
        [8, 3, 2],
        [12, 7, 4],
        [18, 12, 6],
    ],
    "padding_mode": "reflect",
}
EXPECTED_UPSAMPLE_SIZE = [712, 488]
EXPECTED_UPSAMPLE_MODE = "bilinear"
EXPECTED_OUT_POINT_POSITIONS = [[128, 32], [32, 2]]


class UvdocTarget(ContractModel):
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    page_number: int = Field(ge=1)
    cohort: Literal["curved", "flat"]
    failure_ids: tuple[str, ...] = ()


class UvdocPreregistration(ContractModel):
    manifest_version: Literal["gmoney_uvdoc_preregistration_v1"] = "gmoney_uvdoc_preregistration_v1"
    release_corpus_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    audited_gold_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluator_name: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    evaluator_sha256: str = Field(pattern=SHA256_PATTERN)
    model_repository: Literal["PaddlePaddle/UVDoc_safetensors"] = UVDOC_MODEL_REPOSITORY
    model_revision: Literal["7b8c629d7a15656889d0b21c73df206ac8a732b5"] = UVDOC_MODEL_REVISION
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_config_sha256: str = Field(pattern=SHA256_PATTERN)
    targets: tuple[UvdocTarget, ...]

    @model_validator(mode="after")
    def require_unique_targets_and_both_cohorts(self) -> UvdocPreregistration:
        identities = {(item.source_sha256, item.page_number) for item in self.targets}
        if len(identities) != len(self.targets):
            raise ValueError("UVDoc preregistration targets must be unique")
        if {item.cohort for item in self.targets} != {"curved", "flat"}:
            raise ValueError("UVDoc preregistration requires curved and flat targets")
        if any(item.cohort == "curved" and not item.failure_ids for item in self.targets):
            raise ValueError("curved UVDoc targets require preregistered failure IDs")
        return self

    def eligible(self, source_sha256: str, page_number: int) -> bool:
        return any(
            item.source_sha256 == source_sha256 and item.page_number == page_number
            for item in self.targets
        )


def load_preregistration(path: Path) -> tuple[UvdocPreregistration, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("UVDoc preregistration must be a regular non-symlink file")
    payload = path.read_bytes()
    parsed = UvdocPreregistration.model_validate_json(payload)
    return parsed, hashlib.sha256(payload).hexdigest()


def _tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("UVDoc model directory must be a regular non-symlink directory")
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts
    )
    if not files:
        raise ValueError("UVDoc model directory is empty")
    for path in files:
        if path.is_symlink():
            raise ValueError("UVDoc model directory may not contain symlinks")
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class UvdocCompatibility:
    model_sha256: str
    model_config_sha256: str
    adapter_config_sha256: str
    paddle_version: str
    paddleocr_version: str
    paddlex_version: str


@dataclass(frozen=True)
class UvdocPreparedRun:
    page_number: int
    status: Literal["valid", "invalid", "failed", "ineligible"]
    reason_code: str
    parent_image_sha256: str | None = None
    image_relative_path: str | None = None
    image_sha256: str | None = None
    enhanced_relative_path: str | None = None
    enhanced_sha256: str | None = None
    grid_relative_path: str | None = None
    grid_sha256: str | None = None
    grid_shape: tuple[int, int, int] | None = None
    width: int | None = None
    height: int | None = None
    compatibility: UvdocCompatibility | None = None
    reproduction_max_error_by_channel: tuple[int, int, int] | None = None
    transform_metrics: dict[str, Any] | None = None
    probe_metrics: dict[str, Any] | None = None


class UvdocModel(Protocol):
    backbone: Any
    head: Any
    upsample_size: Any
    upsample_mode: Any

    def eval(self) -> None: ...


def _atomic_rgb_png(path: Path, image: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        Image.fromarray(image, mode="RGB").save(temporary, format="PNG", compress_level=9)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _enhance_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)


def replay_uvdoc(parent_bgr: NDArray[np.uint8], grid: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Independently reproduce Paddle grid_sample with bounded zero-padded chunks."""

    height, width = parent_bgr.shape[:2]
    source = parent_bgr.astype(np.float64) / 255.0
    output = np.empty((*grid.shape[:2], 3), dtype=np.uint8)
    for start in range(0, grid.shape[0], 128):
        end = min(start + 128, grid.shape[0])
        chunk = grid[start:end]
        x = (chunk[..., 0].astype(np.float64) + 1.0) * 0.5 * (width - 1)
        y = (chunk[..., 1].astype(np.float64) + 1.0) * 0.5 * (height - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = x - x0
        wy = y - y0
        sampled = np.zeros((*chunk.shape[:2], 3), dtype=np.float64)
        for xi, yi, weight in (
            (x0, y0, (1.0 - wx) * (1.0 - wy)),
            (x1, y0, wx * (1.0 - wy)),
            (x0, y1, (1.0 - wx) * wy),
            (x1, y1, wx * wy),
        ):
            valid = (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
            clipped_x = np.clip(xi, 0, width - 1)
            clipped_y = np.clip(yi, 0, height - 1)
            sampled += source[clipped_y, clipped_x] * (weight * valid)[..., None]
        output[start:end] = np.clip(sampled[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
    return output


def _bounded_grid_diagnostics(
    mapping: DenseBackwardGridMapping, grid: NDArray[np.float32]
) -> DenseGridDiagnostics:
    """Scan full-grid safety while bounding percentile working memory."""

    out_of_bounds_samples = int(
        np.count_nonzero(
            (grid[..., 0] < -1.0)
            | (grid[..., 0] > 1.0)
            | (grid[..., 1] < -1.0)
            | (grid[..., 1] > 1.0)
        )
    )
    foldovers = 0
    for start in range(0, grid.shape[0] - 1, 128):
        end = min(start + 129, grid.shape[0])
        chunk = grid[start:end].astype(np.float64, copy=False)
        top_left = chunk[:-1, :-1]
        top_right = chunk[:-1, 1:]
        bottom_left = chunk[1:, :-1]
        bottom_right = chunk[1:, 1:]
        bilinear = bottom_right - top_right - bottom_left + top_left
        for u, v in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
            derivative_x = top_right - top_left + v * bilinear
            derivative_y = bottom_left - top_left + u * bilinear
            determinants = (
                derivative_x[..., 0] * derivative_y[..., 1]
                - derivative_x[..., 1] * derivative_y[..., 0]
            )
            foldovers += int(np.count_nonzero(determinants <= 0))
    y_indexes = np.unique(np.linspace(0, grid.shape[0] - 1, min(257, grid.shape[0])).astype(int))
    x_indexes = np.unique(np.linspace(0, grid.shape[1] - 1, min(257, grid.shape[1])).astype(int))
    sampled = np.ascontiguousarray(grid[np.ix_(y_indexes, x_indexes)])
    sampled_mapping = DenseBackwardGridMapping(
        grid_relative_path=mapping.grid_relative_path,
        grid_sha256=mapping.grid_sha256,
        grid_shape=tuple(sampled.shape),
        child_width=mapping.child_width,
        child_height=mapping.child_height,
        parent_width=mapping.parent_width,
        parent_height=mapping.parent_height,
        padding_mode="zeros",
    )
    diagnostics = analyze_dense_grid(sampled_mapping, sampled)
    return DenseGridDiagnostics(
        **{
            **asdict(diagnostics),
            "foldover_count": foldovers,
            "out_of_bounds_rate": out_of_bounds_samples / (grid.shape[0] * grid.shape[1]),
        }
    )


class PaddleUvdocAdapter:
    """Pinned dynamic UVDoc forward path that exposes its exact sampling grid."""

    def __init__(self, model_dir: Path, device: str = "cpu", model: UvdocModel | None = None):
        self.model_dir = Path(model_dir)
        self.device = device
        self.compatibility = self._probe()
        if model is None:
            import paddle
            from paddlex.inference.models.image_unwarping.modeling import UVDocNet

            paddle.set_device(device)
            model = UVDocNet.from_pretrained(str(self.model_dir))
        self.model = model
        self._validate_runtime_model()
        self.model.eval()

    def _probe(self) -> UvdocCompatibility:
        import paddle

        versions = {
            "paddle": str(paddle.__version__),
            "paddleocr": importlib.metadata.version("paddleocr"),
            "paddlex": importlib.metadata.version("paddlex"),
        }
        if versions != EXPECTED_VERSIONS:
            raise RuntimeError(f"uvdoc_incompatible_versions:{versions}")
        config_path = self.model_dir / "config.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError("UVDoc model requires a regular config.json")
        config = json.loads(config_path.read_text())
        if config != EXPECTED_MODEL_CONFIG:
            raise RuntimeError("uvdoc_incompatible_model_config")
        model_sha256 = _tree_sha256(self.model_dir)
        model_config_sha256 = sha256_file(config_path)
        adapter_config_sha256 = canonical_sha256(
            {
                "adapter_version": UVDOC_ADAPTER_VERSION,
                "versions": versions,
                "model_sha256": model_sha256,
                "model_config_sha256": model_config_sha256,
                "coordinate_domain": "normalized_minus_one_to_one",
                "interpolation": "bilinear",
                "align_corners": True,
                "padding_mode": "zeros",
                "input": "bgr_float32_0_1_nchw",
                "output": "rgb_uint8_png",
            }
        )
        return UvdocCompatibility(
            model_sha256=model_sha256,
            model_config_sha256=model_config_sha256,
            adapter_config_sha256=adapter_config_sha256,
            paddle_version=versions["paddle"],
            paddleocr_version=versions["paddleocr"],
            paddlex_version=versions["paddlex"],
        )

    def _validate_runtime_model(self) -> None:
        if list(getattr(self.model, "upsample_size", ())) != EXPECTED_UPSAMPLE_SIZE:
            raise RuntimeError("uvdoc_incompatible_runtime_upsample_size")
        if getattr(self.model, "upsample_mode", None) != EXPECTED_UPSAMPLE_MODE:
            raise RuntimeError("uvdoc_incompatible_runtime_upsample_mode")
        config = getattr(self.model, "config", None)
        if (
            config is None
            or getattr(config, "out_point_positions2D", None)
            != EXPECTED_OUT_POINT_POSITIONS
        ):
            raise RuntimeError("uvdoc_incompatible_runtime_output_points")

    def _forward(self, batch: NDArray[np.float32]) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        import paddle
        import paddle.nn.functional as functional

        with paddle.no_grad():
            image = paddle.to_tensor(batch)
            height, width = image.shape[2:]
            resized = functional.interpolate(
                image,
                size=self.model.upsample_size,
                mode=self.model.upsample_mode,
                align_corners=True,
            )
            feature_maps = self.model.backbone(resized)
            predicted = self.model.head(paddle.concat(feature_maps, axis=1))
            grid = functional.interpolate(
                predicted,
                size=(height, width),
                mode=self.model.upsample_mode,
                align_corners=True,
            ).transpose([0, 2, 3, 1])
            result = functional.grid_sample(
                image,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        rgb = result.cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1]
        return (np.clip(rgb * 255.0, 0, 255).astype(np.uint8), grid.cpu().numpy()[0])

    def predict(
        self,
        image_path: Path,
        artifact_root: Path,
        *,
        page_number: int,
    ) -> UvdocPreparedRun:
        parent = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if parent is None:
            raise ValueError("uvdoc_input_unreadable")
        height, width = parent.shape[:2]
        batch = np.ascontiguousarray(parent.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output, grid = self._forward(batch)
        if grid.shape != (height, width, 2) or grid.dtype != np.float32:
            raise RuntimeError("uvdoc_incompatible_output_tensor")
        mapping_template = {
            "grid_relative_path": f"lineage/uvdoc/page-{page_number:04d}.grid.npz",
            "grid_sha256": "0" * 64,
            "grid_shape": tuple(grid.shape),
            "child_width": width,
            "child_height": height,
            "parent_width": width,
            "parent_height": height,
            "padding_mode": "zeros",
        }
        provisional = DenseBackwardGridMapping(**mapping_template)
        diagnostics = _bounded_grid_diagnostics(provisional, grid)
        if diagnostics.foldover_count:
            raise ValueError("uvdoc_grid_foldover")
        if diagnostics.out_of_bounds_rate > 0.005:
            raise ValueError("uvdoc_grid_out_of_bounds")
        replay = replay_uvdoc(parent, grid)
        errors = tuple(
            int(np.max(np.abs(replay[..., channel].astype(int) - output[..., channel].astype(int))))
            for channel in range(3)
        )
        if any(value > 1 for value in errors):
            raise ValueError("uvdoc_grid_reproduction_failed")
        image_relative = Path("lineage/uvdoc") / f"page-{page_number:04d}.png"
        enhanced_relative = Path("lineage/uvdoc") / f"page-{page_number:04d}.enhanced.png"
        _atomic_rgb_png(artifact_root / image_relative, output)
        _atomic_rgb_png(artifact_root / enhanced_relative, _enhance_rgb(output))
        stored = write_dense_grid(artifact_root, mapping_template["grid_relative_path"], grid)
        return UvdocPreparedRun(
            page_number=page_number,
            status="valid",
            reason_code="uvdoc_shadow_valid",
            parent_image_sha256=sha256_file(image_path),
            image_relative_path=image_relative.as_posix(),
            image_sha256=sha256_file(artifact_root / image_relative),
            enhanced_relative_path=enhanced_relative.as_posix(),
            enhanced_sha256=sha256_file(artifact_root / enhanced_relative),
            grid_relative_path=stored.relative_path,
            grid_sha256=stored.sha256,
            grid_shape=stored.shape,
            width=width,
            height=height,
            compatibility=self.compatibility,
            reproduction_max_error_by_channel=errors,
            transform_metrics=asdict(diagnostics),
            probe_metrics={},
        )


__all__ = [
    "EXPECTED_MODEL_CONFIG",
    "EXPECTED_VERSIONS",
    "PaddleUvdocAdapter",
    "UVDOC_ADAPTER_VERSION",
    "UVDOC_MODEL_REPOSITORY",
    "UVDOC_MODEL_REVISION",
    "UvdocCompatibility",
    "UvdocPreparedRun",
    "UvdocPreregistration",
    "UvdocTarget",
    "load_preregistration",
    "replay_uvdoc",
]
