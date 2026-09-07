"""Content-addressed authority corpus vault and milestone gate CLI.

The authority vault is deliberately separate from the repository.  It stores
immutable PDF, review, render, and report objects addressed by SHA-256 while
the repository only contains this workflow and its tests.  The CLI is strict
by design: a missing or ambiguous identity is a hold, never an implicit pass.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import fitz
import typer
from PIL import Image
from pydantic import ValidationError

from gmoney.evaluation.corpus import sha256_file

try:  # The authority contracts land alongside this CLI but remain optional for legacy imports.
    from gmoney.contracts.authority import (
        BaselineManifestV1,
        GoldDocumentV2,
        M0M4GateReportV1,
        ReviewDecision,
        ReviewRecordV1,
        ReviewState,
    )
except ImportError:  # pragma: no cover - exercised only by an older checkout.
    BaselineManifestV1 = None  # type: ignore[assignment,misc]
    GoldDocumentV2 = None  # type: ignore[assignment,misc]
    M0M4GateReportV1 = None  # type: ignore[assignment,misc]
    ReviewDecision = None  # type: ignore[assignment,misc]
    ReviewRecordV1 = None  # type: ignore[assignment,misc]
    ReviewState = None  # type: ignore[assignment,misc]

app = typer.Typer(no_args_is_help=True)

DEFAULT_VAULT = Path("/home/azureuser/gmoney-corpus-vault/authoritative-v1")
VAULT_VERSION = "gmoney_authority_vault_v1"
INVENTORY_VERSION = "gmoney_authority_inventory_v1"
RENDER_VERSION = "gmoney_authority_render_v1"
REVIEW_VERSION = "gmoney_authority_review_v1"
SEAL_VERSION = "gmoney_authority_seal_v1"
BASELINE_VERSION = "gmoney_authority_baseline_v1"
EVALUATION_VERSION = "gmoney_authority_evaluation_v1"
GATE_VERSION = "gmoney_authority_gate_m0_m4_v1"
COHORT_COUNTS = {"production14": 14, "passing36": 36, "staging159": 159}
MASTER_COHORT = "staging159"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class AuthorityError(RuntimeError):
    """A fail-closed authority workflow error."""


def _repository_root() -> Path | None:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.exists():
            return candidate
    return None


def _assert_external(path: Path, *, label: str) -> Path:
    """Reject a vault/output path inside the repository."""

    # Resolve only after checking the lexical path.  Resolving first would
    # silently follow an output symlink and could turn a client-controlled
    # path into an unexpected write target.
    candidate = path.expanduser()
    _assert_no_symlink_ancestors(candidate)
    resolved = candidate.resolve()
    repository = _repository_root()
    if repository is not None:
        try:
            resolved.relative_to(repository)
        except ValueError:
            pass
        else:
            raise AuthorityError(f"{label} must be outside the repository: {resolved}")
    return resolved


def _vault_path(value: Path | None, *, create: bool = False) -> Path:
    vault = _assert_external(value or DEFAULT_VAULT, label="authority vault")
    if create:
        vault.mkdir(parents=True, exist_ok=True)
        if vault.is_symlink():
            raise AuthorityError("authority vault cannot be a symlink")
    return vault


def _assert_no_symlink_ancestors(path: Path, *, root: Path | None = None) -> None:
    current = path.expanduser()
    stop = root.expanduser().resolve() if root is not None else None
    while True:
        if current.exists() and current.is_symlink():
            raise AuthorityError(f"path contains a symlink: {current}")
        if stop is not None:
            try:
                if current.resolve() == stop:
                    break
            except OSError as error:
                raise AuthorityError(f"cannot inspect path: {current}") from error
        if current.parent == current:
            break
        current = current.parent


def _safe_existing_file(path: Path, *, root: Path | None = None, label: str = "file") -> Path:
    _assert_no_symlink_ancestors(path, root=root)
    if path.is_symlink():
        raise AuthorityError(f"{label} may not be a symlink: {path}")
    if not path.is_file():
        raise AuthorityError(f"{label} is not a regular file: {path}")
    resolved = path.resolve()
    if root is not None:
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as error:
            raise AuthorityError(f"{label} escapes its root: {path}") from error
    return resolved


def _safe_directory(path: Path, *, label: str) -> Path:
    _assert_no_symlink_ancestors(path)
    if path.is_symlink():
        raise AuthorityError(f"{label} may not be a symlink: {path}")
    if not path.is_dir():
        raise AuthorityError(f"{label} is not a directory: {path}")
    return path.resolve()


def _iter_regular_files(root: Path) -> list[Path]:
    root = _safe_directory(root, label="source root")
    output: list[Path] = []
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise AuthorityError(f"cannot inspect source root: {current}") from error
        for child in children:
            if child.is_symlink():
                # Source discovery is read-only and must never follow links.
                # Ignoring unrelated links lets a project/archive root be
                # inventoried without treating node_modules-style links as
                # candidate authority material.
                continue
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                output.append(child)
            else:
                raise AuthorityError(f"source tree contains a non-regular entry: {child}")
    return sorted(output)


def _assert_relative(value: str, *, label: str) -> str:
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise AuthorityError(f"{label} contains traversal: {value}")
    return candidate.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _write_immutable(path: Path, payload: bytes) -> str:
    """Write an object once; a changed object is a hard failure."""

    path = _assert_external(path, label="authority output")
    _assert_no_symlink_ancestors(path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise AuthorityError(f"authority output is not a regular immutable file: {path}")
        if path.read_bytes() != payload:
            raise AuthorityError(f"authority output already exists with different bytes: {path}")
        return _sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise AuthorityError(f"authority output parent is a symlink: {path.parent}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise AuthorityError(f"temporary authority output already exists: {temporary}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(payload)


def _write_json(path: Path, payload: Any) -> str:
    return _write_immutable(path, _canonical_bytes(payload))


def _control_path(vault: Path, name: str) -> Path:
    return vault / "control" / name


def _object_path(vault: Path, digest: str, suffix: str) -> Path:
    if not SHA256_RE.fullmatch(digest):
        raise AuthorityError(f"invalid content digest: {digest}")
    if not suffix.startswith("."):
        suffix = "." + suffix
    return vault / "objects" / "sha256" / f"{digest}{suffix}"


def _store_object(vault: Path, source: Path, digest: str, suffix: str) -> Path:
    source = _safe_existing_file(source, label="source object")
    destination = _object_path(vault, digest, suffix)
    _assert_no_symlink_ancestors(destination, root=vault)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or sha256_file(destination) != digest:
            raise AuthorityError(f"content-addressed object mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise AuthorityError(f"temporary authority object already exists: {temporary}")
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != digest:
            raise AuthorityError(f"source changed while copying: {source}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _store_json_object(vault: Path, payload: Any) -> tuple[Path, str]:
    """Store a canonical JSON object under its content digest.

    Review envelopes can contain several independent records.  Keeping each
    validated record as a canonical object means a later report can point to
    the exact record even when the client supplied one combined JSON file.
    """

    encoded = _canonical_bytes(payload)
    digest = _sha256_bytes(encoded)
    destination = _object_path(vault, digest, ".json")
    _write_immutable(destination, encoded)
    return destination, digest


def _read_json(path: Path, *, label: str) -> Any:
    path = _safe_existing_file(path, label=label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityError(f"invalid {label}: {path}") from error


def _find_control(vault: Path, names: Iterable[str], explicit: Path | None = None) -> Path:
    if explicit is not None:
        return _safe_existing_file(explicit, label="authority report")
    for name in names:
        path = _control_path(vault, name)
        if path.is_file() and not path.is_symlink():
            return path
    raise AuthorityError(
        f"authority report is missing under {vault / 'control'}: {', '.join(names)}"
    )


def _parse_assignments(
    values: list[str], *, option: str
) -> list[tuple[str, Path, int | None]]:
    assignments: list[tuple[str, Path, int | None]] = []
    for value in values:
        name, separator, remainder = value.partition("=")
        if not separator or not name or not remainder:
            raise AuthorityError(f"{option} must be NAME=PATH or NAME=PATH:COUNT")
        path_text, count_text = remainder, None
        possible_path, possible_separator, possible_count = remainder.rpartition(":")
        if possible_separator and possible_count.isdigit():
            path_text, count_text = possible_path, possible_count
        path = Path(path_text)
        assignments.append((name, path, int(count_text) if count_text else None))
    return assignments


def _discover_cohorts(root: Path) -> dict[str, Path]:
    children = {
        child.name: child
        for child in root.iterdir()
        if child.is_dir() and not child.is_symlink() and child.name in COHORT_COUNTS
    }
    return {name: children[name] for name in COHORT_COUNTS if name in children}


def _load_inventory(vault: Path, explicit: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    path = _find_control(vault, ("inventory.json",), explicit)
    payload = _read_json(path, label="inventory")
    if not isinstance(payload, dict) or payload.get("inventory_version") != INVENTORY_VERSION:
        raise AuthorityError("invalid authority inventory version")
    return payload, path, sha256_file(path)


def _load_render_manifest(
    vault: Path, explicit: Path | None = None
) -> tuple[dict[str, Any], Path, str]:
    path = _find_control(vault, ("render-manifest.json",), explicit)
    payload = _read_json(path, label="render manifest")
    if not isinstance(payload, dict) or payload.get("render_version") != RENDER_VERSION:
        raise AuthorityError("invalid authority render manifest version")
    return payload, path, sha256_file(path)


def _validate_render_manifest_documents(vault: Path, payload: dict[str, Any]) -> None:
    """Verify every manifest and page object below the external vault."""

    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise AuthorityError("render manifest has no documents list")
    seen: set[str] = set()
    for item in documents:
        if not isinstance(item, dict):
            raise AuthorityError("render manifest document is not an object")
        digest = item.get("document_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise AuthorityError("render manifest has an invalid document digest")
        if digest in seen:
            raise AuthorityError(f"duplicate render manifest document: {digest}")
        seen.add(digest)
        manifest_relpath = item.get("manifest_relpath")
        manifest_digest = item.get("manifest_sha256")
        if not isinstance(manifest_relpath, str) or not isinstance(manifest_digest, str):
            raise AuthorityError(f"render manifest has an invalid per-document identity: {digest}")
        manifest_path = vault / _assert_relative(
            manifest_relpath, label="render manifest relative path"
        )
        manifest_path = _safe_existing_file(
            manifest_path, root=vault, label="render manifest object"
        )
        if sha256_file(manifest_path) != manifest_digest:
            raise AuthorityError(f"render manifest object hash mismatch: {digest}")
        per_document = _read_json(manifest_path, label="render document manifest")
        if not isinstance(per_document, dict) or per_document.get("document_sha256") != digest:
            raise AuthorityError(f"render document identity mismatch: {digest}")
        pages = item.get("pages")
        per_document_pages = per_document.get("pages")
        if not isinstance(pages, list) or per_document_pages != pages:
            raise AuthorityError(f"render page manifest mismatch: {digest}")
        for page in pages:
            if not isinstance(page, dict):
                raise AuthorityError(f"render page entry is invalid: {digest}")
            page_relpath = page.get("relative_path")
            page_digest = page.get("sha256")
            if not isinstance(page_relpath, str) or not isinstance(page_digest, str):
                raise AuthorityError(f"render page identity is invalid: {digest}")
            page_path = vault / _assert_relative(page_relpath, label="render page path")
            page_path = _safe_existing_file(page_path, root=vault, label="render page object")
            if sha256_file(page_path) != page_digest:
                raise AuthorityError(f"render page object hash mismatch: {digest}")


def _inventory_documents(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    documents = inventory.get("documents")
    if not isinstance(documents, list):
        raise AuthorityError("inventory has no documents list")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            raise AuthorityError("inventory document is not an object")
        digest = document.get("source_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise AuthorityError("inventory contains an invalid PDF digest")
        if digest in seen:
            raise AuthorityError(f"duplicate PDF hash in inventory: {digest}")
        seen.add(digest)
        cohorts = document.get("cohorts")
        if cohorts is None:
            # Read old inventories so that an interrupted upgrade fails at the
            # membership gate rather than silently treating one label as the
            # authoritative master membership.
            cohorts = (
                [document["cohort"]]
                if document.get("cohort") is not None
                else []
            )
        if not isinstance(cohorts, list):
            raise AuthorityError("inventory document has invalid cohort memberships")
        if len(set(cohorts)) != len(cohorts):
            raise AuthorityError(f"duplicate cohort membership for PDF: {digest}")
        if any(cohort not in COHORT_COUNTS for cohort in cohorts):
            raise AuthorityError(f"unknown authority cohort membership: {cohorts}")
        document["cohorts"] = sorted(cohorts, key=lambda value: tuple(COHORT_COUNTS).index(value))
        output.append(document)
    return output


def _validate_membership(inventory: dict[str, Any], *, require_complete: bool) -> dict[str, int]:
    documents = _inventory_documents(inventory)
    counts: Counter[str] = Counter()
    for document in documents:
        cohorts = document["cohorts"]
        if require_complete and MASTER_COHORT not in cohorts:
            raise AuthorityError(
                f"authority cohort subsets must belong to {MASTER_COHORT}: "
                f"{document['source_sha256']}"
            )
        counts.update(cohorts)
    if require_complete:
        missing = set(COHORT_COUNTS) - set(counts)
        wrong = {
            name: (counts.get(name, 0), expected)
            for name, expected in COHORT_COUNTS.items()
            if counts.get(name, 0) != expected
        }
        if len(documents) != COHORT_COUNTS[MASTER_COHORT] or missing or wrong:
            raise AuthorityError(
                "insufficient authority cohort membership: "
                + json.dumps(
                    {
                        "master_unique_documents": len(documents),
                        "required_master_documents": COHORT_COUNTS[MASTER_COHORT],
                        "missing": sorted(missing),
                        "counts": wrong,
                    },
                    sort_keys=True,
                )
            )
    return {name: int(counts.get(name, 0)) for name in COHORT_COUNTS}


def _normalise_identity(raw: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(raw or {})
    nested = source.get("authority_identity") or source.get("identity")
    if isinstance(nested, dict):
        source = {**source, **nested}
    revision = source.get("release_revision", source.get("revision"))
    images = source.get("images", source.get("image_digests"))
    if images is None:
        images = {
            key: source[key]
            for key in ("api_image_digest", "frontend_image_digest", "worker_image_digest")
            if source.get(key) is not None
        }
    models = source.get("models", source.get("model_digests"))
    if models is None:
        models = {
            key: source[key]
            for key in ("model_sha256", "model_config_sha256", "adapter_config_sha256")
            if source.get(key) is not None
        }
    config = source.get("config_sha256", source.get("config"))
    corpus = source.get("corpus_sha256", source.get("corpus"))
    gold = source.get("gold_sha256", source.get("gold"))
    evaluator = source.get("evaluator_sha256", source.get("evaluator"))
    result: dict[str, Any] = {
        "release_revision": revision,
        "images": images,
        "models": models,
        "config_sha256": config,
        "corpus_sha256": corpus,
        "gold_sha256": gold,
        "evaluator_sha256": evaluator,
    }
    return {key: value for key, value in result.items() if value not in (None, {}, [])}


def _identity_complete(identity: dict[str, Any]) -> bool:
    required = {
        "release_revision",
        "images",
        "models",
        "config_sha256",
        "corpus_sha256",
        "gold_sha256",
        "evaluator_sha256",
    }
    if not required <= set(identity) or any(
        identity.get(key) in (None, {}, "") for key in required
    ):
        return False
    if not REVISION_RE.fullmatch(str(identity["release_revision"])):
        return False
    for key in ("config_sha256", "corpus_sha256", "gold_sha256", "evaluator_sha256"):
        if not SHA256_RE.fullmatch(str(identity[key])):
            return False
    for group in ("images", "models"):
        values = identity[group]
        if not isinstance(values, dict) or not values:
            return False
        for value in values.values():
            digest = str(value)
            if digest.startswith("sha256:"):
                digest = digest.removeprefix("sha256:")
            if not SHA256_RE.fullmatch(digest):
                return False
    return True


def _identity_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _canonical_digest(_normalise_identity(left)) == _canonical_digest(
        _normalise_identity(right)
    )


def _digest_or_value(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists() or path.is_symlink():
        path = _safe_existing_file(path, label=label)
        return sha256_file(path)
    if not value.strip():
        raise AuthorityError(f"{label} may not be empty")
    return value


def _identity_from_options(
    *,
    identity_path: Path | None = None,
    release_revision: str | None = None,
    images: list[str] | None = None,
    models: list[str] | None = None,
    config: str | None = None,
    corpus: str | None = None,
    gold: str | None = None,
    evaluator: str | None = None,
) -> dict[str, Any]:
    identity = _normalise_identity(
        _read_json(identity_path, label="identity") if identity_path is not None else None
    )
    if release_revision is not None:
        identity["release_revision"] = release_revision
    if images:
        parsed = _parse_identity_pairs(images, option="--image")
        identity["images"] = parsed
    if models:
        parsed = _parse_identity_pairs(models, option="--model")
        identity["models"] = parsed
    for key, value, label in (
        ("config_sha256", config, "config"),
        ("corpus_sha256", corpus, "corpus"),
        ("gold_sha256", gold, "gold"),
        ("evaluator_sha256", evaluator, "evaluator"),
    ):
        resolved = _digest_or_value(value, label=label)
        if resolved is not None:
            identity[key] = resolved
    return _normalise_identity(identity)


def _parse_identity_pairs(values: list[str], *, option: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        key, separator, digest = value.partition("=")
        if not separator or not key or not digest:
            raise AuthorityError(f"{option} must be NAME=VALUE")
        if key in output:
            raise AuthorityError(f"duplicate {option} key: {key}")
        output[key] = _digest_or_value(digest, label=option) or ""
    return dict(sorted(output.items()))


def _merge_identity(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and merged[key] not in (None, {}, [])
            and value not in (None, {}, [])
            and _canonical_digest(merged[key]) != _canonical_digest(value)
        ):
            raise AuthorityError(f"authority identity mismatch for {key}")
        merged[key] = value
    return _normalise_identity(merged)


def _load_review_files(root: Path, explicit: list[Path]) -> list[Path]:
    if explicit:
        paths = [_safe_existing_file(path, label="review annotation") for path in explicit]
    else:
        root = _safe_directory(root, label="review root")
        paths = [
            path
            for path in _iter_regular_files(root)
            if path.suffix.casefold() == ".json" and path.name not in {"manifest.json"}
        ]
    if not paths:
        raise AuthorityError("no review annotations found")
    return sorted(paths)


def _annotation_payload(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise AuthorityError("review annotation must be an object")
    metadata: dict[str, Any] = {}
    if isinstance(raw.get("annotation"), dict):
        payload = dict(raw["annotation"])
        metadata.update({key: value for key, value in raw.items() if key != "annotation"})
    else:
        payload = dict(raw)
    allowed_metadata = {
        "review_status",
        "adjudication_status",
        "adjudicated",
        "adjudicator",
        "reviewer_ids",
        "source_sha256",
        "reviewed_pages",
        "notes",
    }
    for key in tuple(payload):
        if key in allowed_metadata:
            metadata[key] = payload.pop(key)
    return payload, metadata


def _review_status(metadata: dict[str, Any]) -> str:
    status = metadata.get("adjudication_status", metadata.get("review_status"))
    if status is None and metadata.get("adjudicated") is True:
        status = "adjudicated"
    if not isinstance(status, str) or status.casefold() not in {
        "adjudicated",
        "complete",
        "approved",
        "sealed",
        "passed",
    }:
        raise AuthorityError("review is incomplete or unadjudicated")
    return status.casefold()


def _authority_payloads(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract V2 gold and V1 review payloads from supported JSON envelopes."""

    if not isinstance(raw, dict):
        raise AuthorityError("authority review file must be an object")
    gold: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    def add_values(target: list[dict[str, Any]], value: Any, *, label: str) -> None:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict):
                raise AuthorityError(f"{label} must contain JSON objects")
            target.append(item)

    # A client may submit one file per model, or one immutable envelope per
    # document.  Keep both forms equivalent at the contract boundary.
    if raw.get("gold_version") == "gold_document_v2":
        gold.append(raw)
    for key in ("gold_document", "gold"):
        if key in raw:
            add_values(gold, raw[key], label=key)
    if raw.get("review_version") == "review_record_v1":
        reviews.append(raw)
    for key in ("review_records", "reviews"):
        if key in raw:
            add_values(reviews, raw[key], label=key)
    return gold, reviews


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _rendered_page_hashes(rendered_document: dict[str, Any], digest: str) -> list[dict[str, Any]]:
    pages = rendered_document.get("pages")
    if not isinstance(pages, list) or not pages:
        raise AuthorityError(f"render manifest has no pages for reviewed PDF: {digest}")
    output: list[dict[str, Any]] = []
    for index, page in enumerate(pages, 1):
        if not isinstance(page, dict):
            raise AuthorityError(f"render manifest has an invalid page for reviewed PDF: {digest}")
        if page.get("page_number") != index or not SHA256_RE.fullmatch(str(page.get("sha256", ""))):
            raise AuthorityError(f"render manifest page identity is invalid: {digest}")
        output.append(page)
    return output


def _validated_review_objects(
    vault: Path,
    summary: dict[str, Any],
    *,
    source_sha256: str,
    rendered_document: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Dereference and validate the immutable gold/review objects used by a seal."""

    if GoldDocumentV2 is None or ReviewRecordV1 is None:
        raise AuthorityError("GoldDocumentV2/ReviewRecordV1 contracts are unavailable")
    gold_relpath = summary.get("annotation_relpath")
    gold_object_sha256 = summary.get("gold_object_sha256")
    if not isinstance(gold_relpath, str) or not SHA256_RE.fullmatch(
        str(gold_object_sha256 or "")
    ):
        raise AuthorityError(f"sealed document lacks an immutable gold object: {source_sha256}")
    gold_path = _safe_existing_file(
        vault / _assert_relative(gold_relpath, label="gold object path"),
        root=vault,
        label="gold object",
    )
    if sha256_file(gold_path) != gold_object_sha256:
        raise AuthorityError(f"gold object hash mismatch: {source_sha256}")
    try:
        gold = GoldDocumentV2.model_validate(_read_json(gold_path, label="gold object"))
    except ValidationError as error:
        raise AuthorityError(f"invalid immutable gold object: {source_sha256}") from error
    if (
        gold.source_sha256 != source_sha256
        or gold.document_id != summary.get("document_id")
        or gold.gold_sha256 != summary.get("gold_sha256")
    ):
        raise AuthorityError(f"gold object identity mismatch: {source_sha256}")

    record_paths = summary.get("review_record_relpaths")
    record_digests = summary.get("review_record_sha256s")
    if (
        not isinstance(record_paths, list)
        or not isinstance(record_digests, list)
        or len(record_paths) != 4
        or len(record_digests) != 4
        or len(set(record_paths)) != 4
        or len(set(record_digests)) != 4
    ):
        raise AuthorityError(f"review object identities are incomplete: {source_sha256}")
    records: dict[str, Any] = {}
    for relative, digest in zip(record_paths, record_digests, strict=True):
        if not isinstance(relative, str) or not SHA256_RE.fullmatch(str(digest)):
            raise AuthorityError(f"invalid review object identity: {source_sha256}")
        path = _safe_existing_file(
            vault / _assert_relative(relative, label="review object path"),
            root=vault,
            label="review object",
        )
        if sha256_file(path) != digest:
            raise AuthorityError(f"review object hash mismatch: {source_sha256}")
        try:
            record = ReviewRecordV1.model_validate(_read_json(path, label="review object"))
        except ValidationError as error:
            raise AuthorityError(f"invalid immutable review object: {source_sha256}") from error
        kind = _enum_text(record.review_kind)
        if kind in records:
            raise AuthorityError(f"duplicate immutable {kind} review: {source_sha256}")
        if (
            record.source_sha256 != source_sha256
            or record.document_id != gold.document_id
            or _enum_text(record.state) != "frozen"
            or _enum_text(record.decision) == "needs_review"
        ):
            raise AuthorityError(f"immutable review identity/state mismatch: {source_sha256}")
        records[kind] = record
    required = {"independent_a", "independent_b", "adjudicator", "red_team"}
    if set(records) != required:
        raise AuthorityError(f"immutable four-pass review set is incomplete: {source_sha256}")
    ids = [records[kind].review_id for kind in sorted(required)]
    if set(ids) != set(summary.get("review_ids", [])):
        raise AuthorityError(f"review summary IDs do not match objects: {source_sha256}")
    independent = {records["independent_a"].review_id, records["independent_b"].review_id}
    if (
        records["independent_a"].reviewer_identity
        == records["independent_b"].reviewer_identity
        or set(records["adjudicator"].independent_review_ids) != independent
    ):
        raise AuthorityError(f"immutable review independence mismatch: {source_sha256}")
    rendered_pages = _rendered_page_hashes(rendered_document, source_sha256)
    expected_page_hashes = {page["sha256"] for page in rendered_pages}
    expected_manifest = rendered_document.get("manifest_sha256")
    if any(
        set(record.image_sha256s) != expected_page_hashes
        or record.image_manifest_sha256 != expected_manifest
        for record in records.values()
    ):
        raise AuthorityError(f"immutable review image identity mismatch: {source_sha256}")
    return gold, records


def _load_machine_payload(path: Path) -> tuple[Any, str, dict[str, Any] | None]:
    if path.is_symlink():
        raise AuthorityError(f"machine output may not be a symlink: {path}")
    if path.is_file():
        payload = _read_json(path, label="machine output")
        return payload, sha256_file(path), payload if isinstance(payload, dict) else None
    if not path.is_dir():
        raise AuthorityError(f"machine output is missing: {path}")
    files = _iter_regular_files(path)
    digest_input = bytearray()
    json_files = []
    for file in files:
        relative = file.relative_to(path).as_posix()
        encoded_name = relative.encode("utf-8")
        content = file.read_bytes()
        digest_input.extend(len(encoded_name).to_bytes(8, "big"))
        digest_input.extend(encoded_name)
        digest_input.extend(len(content).to_bytes(8, "big"))
        digest_input.extend(content)
        if file.suffix.casefold() == ".json":
            json_files.append(file)
    payload: Any = {}
    for name in ("machine-output.json", "result.json", "output.json", "identity.json"):
        match = path / name
        if match in json_files:
            payload = _read_json(match, label="machine output")
            break
    return (
        payload,
        _sha256_bytes(bytes(digest_input)),
        payload if isinstance(payload, dict) else None,
    )


def _machine_identity(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return _normalise_identity(payload)


def _assert_machine_only(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    for key in ("reviewed_output", "human_corrected", "adjudicated_output"):
        if payload.get(key) not in (None, False, 0, "", []):
            raise AuthorityError(f"machine-output baseline contains {key}")
    for key in ("gold", "audited_gold", "gold_documents"):
        if key in payload:
            raise AuthorityError(f"machine output may not supply authority-controlled {key}")
    kind = payload.get("output_kind", payload.get("result_kind"))
    if kind is not None and str(kind).casefold() not in {
        "machine",
        "machine_output",
        "baseline",
        "raw",
    }:
        raise AuthorityError(f"non-machine output cannot be used as baseline: {kind}")


def _call_authority_metrics(context: dict[str, Any]) -> Any | None:
    """Call optional authority_metrics implementations without hard dependency."""

    try:
        module = importlib.import_module("gmoney.evaluation.authority_metrics")
    except ModuleNotFoundError:
        return None
    candidates = [
        getattr(module, name, None) for name in ("evaluate", "evaluate_authority", "compute", "run")
    ]
    function = next((item for item in candidates if callable(item)), None)
    if function is None:
        return None
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        kwargs = {
            name: value
            for name, value in context.items()
            if name in signature.parameters
            and signature.parameters[name].kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        ]
        if all(parameter.name in kwargs for parameter in required):
            return function(**kwargs)
    for args in (
        (context,),
        (context.get("baseline_payloads"), context.get("candidate_payloads")),
        (context.get("candidate_payloads"),),
        (context.get("baseline_payloads"), context.get("candidate_payloads"), context),
    ):
        try:
            return function(*args)
        except TypeError:
            continue
    # The optional evaluator may expose a document-level ``gold, actual`` API.
    # A baseline-only operation has no gold/actual pair; defer to the fallback
    # metrics rather than turning an otherwise valid identity check into a crash.
    return None


def _fallback_metrics(payloads: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    metrics: list[dict[str, Any]] = []
    for payload in payloads:
        if not payload:
            continue
        candidate = payload.get("metrics") or payload.get("quality")
        if isinstance(candidate, dict):
            metrics.append(candidate)
        if isinstance(payload.get("evaluation"), dict):
            metrics.append(payload["evaluation"])
    if not metrics:
        return None
    if len(metrics) == 1:
        result = dict(metrics[0])
    else:
        numeric: dict[str, list[float]] = defaultdict(list)
        for metric in metrics:
            for key, value in metric.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric[key].append(float(value))
        result = {key: sum(values) / len(values) for key, values in numeric.items()}
    if "passed" in result:
        result["passed"] = bool(result["passed"])
    return result


def _serialise_metrics(value: Any) -> Any:
    """Make optional evaluator reports safe for canonical JSON storage."""

    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return value


def _write_report_or_exit(path: Path, payload: dict[str, Any], *, passed: bool) -> None:
    _write_json(path, payload)
    if not passed:
        raise typer.Exit(code=1)


def _validate_linked_json_object(
    vault: Path, payload: dict[str, Any], *, prefix: str
) -> dict[str, Any]:
    """Load a content-addressed authority object linked by a control report."""

    relative = payload.get(f"{prefix}_relpath")
    digest = payload.get(f"{prefix}_sha256")
    if not isinstance(relative, str) or not SHA256_RE.fullmatch(str(digest or "")):
        raise AuthorityError(f"{prefix} authority object link is missing")
    path = _safe_existing_file(
        vault / _assert_relative(relative, label=f"{prefix} authority object path"),
        root=vault,
        label=f"{prefix} authority object",
    )
    if sha256_file(path) != digest:
        raise AuthorityError(f"{prefix} authority object hash mismatch")
    value = _read_json(path, label=f"{prefix} authority object")
    if not isinstance(value, dict):
        raise AuthorityError(f"{prefix} authority object must be a JSON object")
    return value


def _effective_vault(ctx: typer.Context, override: Path | None) -> Path:
    value = override or (ctx.obj or {}).get("vault") or DEFAULT_VAULT
    return _vault_path(Path(value), create=True)


@app.callback()
def main(
    ctx: typer.Context,
    vault: Annotated[
        Path, typer.Option("--vault", help="External authority vault root.")
    ] = DEFAULT_VAULT,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["vault"] = vault


@app.command("inventory")
def inventory(
    ctx: typer.Context,
    source_root: Annotated[
        list[Path] | None, typer.Option("--source-root", "--input-root")
    ] = None,
    cohort: Annotated[list[str] | None, typer.Option("--cohort")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    vault = _effective_vault(ctx, vault_override)
    assignments = _parse_assignments(cohort or [], option="--cohort")
    source_roots = list(source_root or [])
    if assignments and source_roots:
        raise typer.BadParameter("use --source-root or --cohort, not both")
    if not assignments:
        if not source_roots:
            source_roots = [vault / "sources"]
        discovered_assignments: list[tuple[str, Path, int | None]] = []
        for root in source_roots:
            discovered = _discover_cohorts(root) if root.is_dir() else {}
            if discovered:
                discovered_assignments.extend(
                    (name, path, None) for name, path in discovered.items()
                )
            else:
                discovered_assignments.append(("unassigned", root, None))
    else:
        discovered_assignments = assignments
    # A document is identified by source bytes, not by the cohort directory
    # in which it happened to be found.  The production and passing cohorts
    # are memberships of the staging master, so the same PDF is expected to
    # occur in more than one assigned root.  Repeated bytes within one archive
    # are harmless copies, too: content identity remains unambiguous and the
    # lexicographically first relative path is retained for provenance.
    documents_by_digest: dict[str, dict[str, Any]] = {}
    seen_by_root: dict[Path, set[str]] = {}
    for name, root, declared_count in discovered_assignments:
        root = _safe_directory(root, label=f"{name} source root")
        files = [path for path in _iter_regular_files(root) if path.suffix.casefold() == ".pdf"]
        if declared_count is not None and len(files) != declared_count:
            raise AuthorityError(f"{name} requires {declared_count} PDFs, found {len(files)}")
        if name not in COHORT_COUNTS and name != "unassigned":
            raise AuthorityError(f"unknown authority cohort: {name}")
        root_seen = seen_by_root.setdefault(root, set())
        for source in files:
            digest = sha256_file(source)
            if digest in root_seen:
                continue
            root_seen.add(digest)
            object_path = _store_object(vault, source, digest, ".pdf")
            relative = source.relative_to(root).as_posix()
            relative = _assert_relative(relative, label="source relative path")
            document = documents_by_digest.get(digest)
            if document is None:
                document = {
                    "source_sha256": digest,
                    "cohorts": [],
                    "object_relpath": object_path.relative_to(vault).as_posix(),
                    # Keep this legacy convenience field for consumers that
                    # only need one stable source path.  The complete mapping
                    # is source_relpaths below.
                    "source_relpath": relative,
                    "source_relpaths": {},
                    "size_bytes": source.stat().st_size,
                }
                documents_by_digest[digest] = document
            source_key = name if name != "unassigned" else "unassigned"
            previous_relative = document["source_relpaths"].get(source_key)
            if previous_relative is None or relative < previous_relative:
                document["source_relpaths"][source_key] = relative
            if name != "unassigned" and name not in document["cohorts"]:
                document["cohorts"].append(name)
                document["cohorts"].sort(key=lambda value: tuple(COHORT_COUNTS).index(value))
    documents = sorted(documents_by_digest.values(), key=lambda item: item["source_sha256"])
    for document in documents:
        document["source_relpath"] = min(
            document["source_relpaths"].items(), key=lambda item: item[0]
        )[1]
    counts = Counter(
        cohort
        for document in documents
        for cohort in document.get("cohorts", [])
    )
    master_unique_documents = sum(
        MASTER_COHORT in document.get("cohorts", []) for document in documents
    )
    membership_complete = (
        len(documents) == COHORT_COUNTS[MASTER_COHORT]
        and master_unique_documents == COHORT_COUNTS[MASTER_COHORT]
        and all(counts.get(name, 0) == expected for name, expected in COHORT_COUNTS.items())
    )
    payload = {
        "vault_version": VAULT_VERSION,
        "inventory_version": INVENTORY_VERSION,
        "documents": documents,
        "cohort_counts": {name: int(counts.get(name, 0)) for name in COHORT_COUNTS},
        "unassigned_count": sum(not document.get("cohorts") for document in documents),
        "master_unique_documents": master_unique_documents,
        "membership_gaps": {
            name: max(expected - int(counts.get(name, 0)), 0)
            for name, expected in COHORT_COUNTS.items()
        },
        "membership_complete": membership_complete,
    }
    destination = _assert_external(
        output or _control_path(vault, "inventory.json"), label="inventory output"
    )
    digest = _write_json(destination, payload)
    typer.echo(
        json.dumps(
            {
                "inventory": str(destination),
                "sha256": digest,
                "documents": len(documents),
                "cohort_counts": payload["cohort_counts"],
                "membership_gaps": payload["membership_gaps"],
                "membership_complete": membership_complete,
            }
        )
    )


@app.command("render")
def render(
    ctx: typer.Context,
    inventory_path: Annotated[Path | None, typer.Option("--inventory")] = None,
    source_sha256: Annotated[list[str] | None, typer.Option("--source-sha256", "--source")] = None,
    dpi: Annotated[int, typer.Option("--dpi")] = 300,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    if dpi != 300:
        raise typer.BadParameter("authority rendering is fixed at 300 DPI")
    vault = _effective_vault(ctx, vault_override)
    inventory_payload, _inventory_file, inventory_digest = _load_inventory(vault, inventory_path)
    documents = _inventory_documents(inventory_payload)
    requested = set(source_sha256 or [])
    if requested and not requested <= {item["source_sha256"] for item in documents}:
        raise AuthorityError("render source hash is not in inventory")
    selected = [item for item in documents if not requested or item["source_sha256"] in requested]
    rendered: list[dict[str, Any]] = []
    for document in selected:
        digest = document["source_sha256"]
        source = vault / document["object_relpath"]
        source = _safe_existing_file(source, root=vault, label="PDF object")
        if sha256_file(source) != digest:
            raise AuthorityError(f"PDF object hash mismatch: {digest}")
        render_root = vault / "renders" / digest / "300dpi"
        _assert_no_symlink_ancestors(render_root, root=vault)
        render_root.mkdir(parents=True, exist_ok=True)
        if render_root.is_symlink() or not render_root.is_dir():
            raise AuthorityError(f"render output directory is not regular: {render_root}")
        manifest_path = render_root / "manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            prior = _read_json(manifest_path, label="render document manifest")
            if not isinstance(prior, dict):
                raise AuthorityError(f"render document manifest is invalid: {digest}")
            prior_item = {
                "document_sha256": digest,
                "manifest_relpath": manifest_path.relative_to(vault).as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "pages": prior.get("pages"),
            }
            if (
                prior.get("render_version") != RENDER_VERSION
                or prior.get("document_sha256") != digest
                or prior.get("dpi") != 300
                or prior.get("colorspace") != "sRGB"
                or prior.get("renderer") != "PyMuPDF"
                or prior.get("renderer_version") != fitz.VersionBind
            ):
                raise AuthorityError(f"render document manifest identity mismatch: {digest}")
            _validate_render_manifest_documents(vault, {"documents": [prior_item]})
            rendered.append(prior_item)
            continue
        pages: list[dict[str, Any]] = []
        before_size = source.stat().st_size
        before_hash = sha256_file(source)
        try:
            pdf = fitz.open(source)
            for index, page in enumerate(pdf, 1):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(300 / 72, 300 / 72), colorspace=fitz.csRGB, alpha=False
                )
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                page_path = render_root / f"page-{index:04d}.png"
                temporary = page_path.with_name(f".{page_path.name}.tmp-{os.getpid()}")
                image.save(temporary, format="PNG", optimize=False, compress_level=9)
                page_bytes = temporary.read_bytes()
                page_digest = _sha256_bytes(page_bytes)
                if page_path.exists() or page_path.is_symlink():
                    if page_path.is_symlink() or page_path.read_bytes() != page_bytes:
                        raise AuthorityError(f"render output changed for {digest} page {index}")
                    temporary.unlink()
                else:
                    temporary.replace(page_path)
                pages.append(
                    {
                        "page_number": index,
                        "sha256": page_digest,
                        "relative_path": page_path.relative_to(vault).as_posix(),
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "dpi": 300,
                        "colorspace": "sRGB",
                    }
                )
            pdf.close()
        except AuthorityError:
            raise
        except Exception as error:
            raise AuthorityError(f"failed to render {digest}") from error
        if sha256_file(source) != before_hash or source.stat().st_size != before_size:
            raise AuthorityError(f"PDF changed while rendering: {digest}")
        manifest = {
            "render_version": RENDER_VERSION,
            "document_sha256": digest,
            "dpi": 300,
            "colorspace": "sRGB",
            "renderer": "PyMuPDF",
            "renderer_version": fitz.VersionBind,
            "pages": pages,
        }
        manifest_digest = _write_json(manifest_path, manifest)
        rendered.append(
            {
                "document_sha256": digest,
                "manifest_relpath": manifest_path.relative_to(vault).as_posix(),
                "manifest_sha256": manifest_digest,
                "pages": pages,
            }
        )
    existing: dict[str, Any] = {}
    render_manifest_path = _control_path(vault, "render-manifest.json")
    if render_manifest_path.is_file() and not render_manifest_path.is_symlink():
        existing_payload = _read_json(render_manifest_path, label="render manifest")
        if isinstance(existing_payload, dict):
            _validate_render_manifest_documents(vault, existing_payload)
            existing = {
                item["document_sha256"]: item for item in existing_payload.get("documents", [])
            }
            inventory_hashes = {item["source_sha256"] for item in documents}
            unknown_existing = set(existing) - inventory_hashes
            if unknown_existing:
                raise AuthorityError(
                    "existing render manifest contains documents outside inventory: "
                    + ", ".join(sorted(unknown_existing))
                )
    for item in rendered:
        existing[item["document_sha256"]] = item
    aggregate = {
        "vault_version": VAULT_VERSION,
        "render_version": RENDER_VERSION,
        "dpi": 300,
        "colorspace": "sRGB",
        "inventory_sha256": inventory_digest,
        "documents": [existing[key] for key in sorted(existing)],
    }
    destination = _assert_external(output or render_manifest_path, label="render manifest output")
    digest = _write_json(destination, aggregate)
    typer.echo(
        json.dumps(
            {"render_manifest": str(destination), "sha256": digest, "documents": len(rendered)}
        )
    )


@app.command("validate-review")
def validate_review(
    ctx: typer.Context,
    review_root: Annotated[
        Path | None, typer.Option("--review-root", "--gold-root", "--annotations")
    ] = None,
    annotation: Annotated[list[Path] | None, typer.Option("--annotation")] = None,
    inventory_path: Annotated[Path | None, typer.Option("--inventory")] = None,
    render_manifest_path: Annotated[Path | None, typer.Option("--render-manifest")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    vault = _effective_vault(ctx, vault_override)
    inventory, _inventory_file, inventory_digest = _load_inventory(vault, inventory_path)
    render_manifest, _render_file, render_digest = _load_render_manifest(
        vault, render_manifest_path
    )
    if render_manifest.get("inventory_sha256") != inventory_digest:
        raise AuthorityError("render manifest inventory identity mismatch")
    _validate_render_manifest_documents(vault, render_manifest)
    inventory_documents = {item["source_sha256"]: item for item in _inventory_documents(inventory)}
    render_documents = {
        item["document_sha256"]: item for item in render_manifest.get("documents", [])
    }
    root = review_root or vault / "gold"
    paths = _load_review_files(root, annotation or [])
    if GoldDocumentV2 is None or ReviewRecordV1 is None:
        raise AuthorityError("GoldDocumentV2/ReviewRecordV1 contracts are unavailable")

    gold_by_source: dict[str, tuple[Any, Path, str]] = {}
    reviews_by_source: defaultdict[str, list[tuple[Any, Path, str]]] = defaultdict(list)
    seen_review_ids: set[str] = set()
    for path in paths:
        raw = _read_json(path, label="review annotation")
        gold_payloads, review_payloads = _authority_payloads(raw)
        # GoldAnnotation (including an image_review with passes=2) is a
        # legacy schema and is intentionally not accepted as authority gold.
        if not gold_payloads and not review_payloads:
            raise AuthorityError(
                f"{path} is not a GoldDocumentV2/ReviewRecordV1 authority record"
            )
        raw_digest = sha256_file(path)
        for payload in gold_payloads:
            try:
                parsed = GoldDocumentV2.model_validate(payload)
            except ValidationError as error:
                raise AuthorityError(f"invalid GoldDocumentV2 {path}: {error}") from error
            digest = parsed.source_sha256
            if digest not in inventory_documents:
                raise AuthorityError(f"gold document identity is not in inventory: {digest}")
            if digest in gold_by_source:
                raise AuthorityError(f"duplicate GoldDocumentV2 for PDF: {digest}")
            gold_by_source[digest] = (parsed, path, raw_digest)
        for payload in review_payloads:
            try:
                parsed_review = ReviewRecordV1.model_validate(payload)
            except ValidationError as error:
                raise AuthorityError(f"invalid ReviewRecordV1 {path}: {error}") from error
            digest = parsed_review.source_sha256
            if digest not in inventory_documents:
                raise AuthorityError(f"review identity is not in inventory: {digest}")
            if parsed_review.review_id in seen_review_ids:
                raise AuthorityError(f"duplicate ReviewRecordV1: {parsed_review.review_id}")
            seen_review_ids.add(parsed_review.review_id)
            reviews_by_source[digest].append((parsed_review, path, raw_digest))

    if not gold_by_source:
        raise AuthorityError("no GoldDocumentV2 records found")
    for digest in reviews_by_source:
        if digest not in gold_by_source:
            raise AuthorityError(f"review has no matching GoldDocumentV2: {digest}")

    validated: list[dict[str, Any]] = []
    for digest, (gold, _gold_path, gold_file_digest) in sorted(gold_by_source.items()):
        rendered_document = render_documents.get(digest)
        if not rendered_document:
            raise AuthorityError(f"render manifest is missing reviewed PDF: {digest}")
        rendered_pages = _rendered_page_hashes(rendered_document, digest)
        if gold.page_count != len(rendered_pages):
            raise AuthorityError(f"gold page count does not match rendered PDF: {digest}")
        for gold_page, rendered_page in zip(gold.pages, rendered_pages, strict=True):
            if gold_page.artifact_sha256 != rendered_page["sha256"]:
                raise AuthorityError(f"gold page hash mismatch: {digest}")
            for key in ("width", "height", "dpi"):
                if getattr(gold_page, key) != rendered_page.get(key):
                    raise AuthorityError(f"gold page geometry mismatch: {digest}")
        expected_pages = {page["sha256"] for page in rendered_pages}
        expected_manifest = rendered_document.get("manifest_sha256")
        if not isinstance(expected_manifest, str) or not SHA256_RE.fullmatch(expected_manifest):
            raise AuthorityError(f"render manifest identity is invalid: {digest}")

        records = reviews_by_source.get(digest, [])
        by_kind: dict[str, tuple[Any, Path, str]] = {}
        for record, record_path, record_file_digest in records:
            if record.document_id != gold.document_id:
                raise AuthorityError(f"review document identity mismatch: {digest}")
            kind = _enum_text(record.review_kind)
            if kind in by_kind:
                raise AuthorityError(f"duplicate {kind} review for PDF: {digest}")
            if _enum_text(record.state) != "frozen":
                raise AuthorityError(f"review is incomplete or unadjudicated: {digest}")
            if _enum_text(record.decision) == "needs_review":
                raise AuthorityError(f"review is incomplete or unadjudicated: {digest}")
            if set(record.image_sha256s) != expected_pages or len(record.image_sha256s) != len(
                expected_pages
            ):
                raise AuthorityError(f"review image hashes mismatch: {digest}")
            if record.image_manifest_sha256 != expected_manifest:
                raise AuthorityError(f"review image manifest identity mismatch: {digest}")
            by_kind[kind] = (record, record_path, record_file_digest)
        required_kinds = {"independent_a", "independent_b", "adjudicator", "red_team"}
        if set(by_kind) != required_kinds:
            missing = sorted(required_kinds - set(by_kind))
            extra = sorted(set(by_kind) - required_kinds)
            raise AuthorityError(
                f"GoldDocumentV2 requires independent A, independent B, adjudicator, "
                f"and red-team records for {digest}; missing={missing}, extra={extra}"
            )
        independent_a = by_kind["independent_a"][0]
        independent_b = by_kind["independent_b"][0]
        adjudicator = by_kind["adjudicator"][0]
        if independent_a.reviewer_identity == independent_b.reviewer_identity:
            raise AuthorityError(f"independent reviews must have distinct reviewers: {digest}")
        if set(adjudicator.independent_review_ids) != {
            independent_a.review_id,
            independent_b.review_id,
        }:
            raise AuthorityError(
                f"adjudicator does not reference both independent reviews: {digest}"
            )

        gold_object, gold_object_digest = _store_json_object(
            vault, gold.model_dump(mode="json")
        )
        record_digests: list[str] = []
        record_relpaths: list[str] = []
        for kind in ("independent_a", "independent_b", "adjudicator", "red_team"):
            record = by_kind[kind][0]
            record_object, record_digest = _store_json_object(
                vault, record.model_dump(mode="json")
            )
            record_digests.append(record_digest)
            record_relpaths.append(record_object.relative_to(vault).as_posix())
        row_count = sum(
            len(table.rows)
            for page in gold.pages
            for table in page.tables
        )
        source_table_count = sum(len(page.tables) for page in gold.pages)
        validated.append(
            {
                "source_sha256": digest,
                # annotation_sha256 remains the immutable client-file digest
                # for compatibility; gold_sha256 is the V2 contract identity.
                "annotation_sha256": gold_file_digest,
                "annotation_relpath": gold_object.relative_to(vault).as_posix(),
                "gold_sha256": gold.gold_sha256,
                "gold_object_sha256": gold_object_digest,
                "review_status": "frozen",
                "reviewer": adjudicator.reviewer_identity,
                "review_method": "authority_v2",
                "review_passes": None,
                "reviewed_page_count": len(expected_pages),
                "source_table_count": source_table_count,
                "row_count": row_count,
                "document_id": gold.document_id,
                "review_ids": [by_kind[kind][0].review_id for kind in (
                    "independent_a", "independent_b", "adjudicator", "red_team"
                )],
                "review_kinds": ["independent_a", "independent_b", "adjudicator", "red_team"],
                "review_record_sha256s": record_digests,
                "review_record_relpaths": record_relpaths,
            }
        )
    validated.sort(key=lambda item: item["source_sha256"])
    report = {
        "vault_version": VAULT_VERSION,
        "review_version": REVIEW_VERSION,
        "inventory_sha256": inventory_digest,
        "render_manifest_sha256": render_digest,
        "annotations": validated,
        "passed": True,
    }
    destination = _assert_external(
        output or _control_path(vault, "review-report.json"), label="review report output"
    )
    digest = _write_json(destination, report)
    typer.echo(
        json.dumps(
            {"review_report": str(destination), "sha256": digest, "annotations": len(validated)}
        )
    )


@app.command("seal")
def seal(
    ctx: typer.Context,
    inventory_path: Annotated[Path | None, typer.Option("--inventory")] = None,
    review_report_path: Annotated[Path | None, typer.Option("--review-report")] = None,
    render_manifest_path: Annotated[Path | None, typer.Option("--render-manifest")] = None,
    identity_path: Annotated[Path | None, typer.Option("--identity")] = None,
    release_revision: Annotated[
        str | None, typer.Option("--release-revision", "--revision")
    ] = None,
    image: Annotated[list[str] | None, typer.Option("--image")] = None,
    model: Annotated[list[str] | None, typer.Option("--model")] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
    evaluator: Annotated[str | None, typer.Option("--evaluator")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    vault = _effective_vault(ctx, vault_override)
    inventory, inventory_file, inventory_digest = _load_inventory(vault, inventory_path)
    review, review_file, review_digest = _read_report(
        vault, ("review-report.json",), review_report_path
    )
    render_manifest, render_file, render_digest = _load_render_manifest(vault, render_manifest_path)
    if render_manifest.get("inventory_sha256") != inventory_digest:
        raise AuthorityError("render manifest inventory identity mismatch")
    _validate_render_manifest_documents(vault, render_manifest)
    counts = _validate_membership(inventory, require_complete=True)
    if review.get("review_version") != REVIEW_VERSION or review.get("passed") is not True:
        raise AuthorityError("review report is not passed")
    if review.get("inventory_sha256") != inventory_digest:
        raise AuthorityError("review report inventory identity mismatch")
    if review.get("render_manifest_sha256") != render_digest:
        raise AuthorityError("review report render identity mismatch")
    annotations = review.get("annotations")
    if not isinstance(annotations, list):
        raise AuthorityError("review report has no annotations list")
    reviews: dict[str, dict[str, Any]] = {}
    inventory_hashes = {item["source_sha256"] for item in _inventory_documents(inventory)}
    for item in annotations:
        if not isinstance(item, dict) or not isinstance(item.get("source_sha256"), str):
            raise AuthorityError("review report contains an invalid annotation")
        source_sha = item["source_sha256"]
        if source_sha in reviews:
            raise AuthorityError(f"duplicate review evidence for PDF: {source_sha}")
        if not SHA256_RE.fullmatch(source_sha):
            raise AuthorityError(f"review report contains an invalid source hash: {source_sha}")
        if source_sha not in inventory_hashes:
            raise AuthorityError(f"review report identity is not in inventory: {source_sha}")
        reviews[source_sha] = item
    render_documents = {
        item.get("document_sha256"): item for item in render_manifest.get("documents", [])
    }
    unknown_render = set(render_documents) - inventory_hashes
    if unknown_render:
        raise AuthorityError(
            "render manifest contains documents outside inventory: "
            + ", ".join(sorted(str(item) for item in unknown_render))
        )
    documents: list[dict[str, Any]] = []
    for item in _inventory_documents(inventory):
        source_sha = item["source_sha256"]
        if source_sha not in reviews or source_sha not in render_documents:
            raise AuthorityError(f"sealed document lacks review/render evidence: {source_sha}")
        review_item = reviews[source_sha]
        required_review_fields = (
            "annotation_sha256",
            "gold_sha256",
            "document_id",
            "review_ids",
            "review_kinds",
            "review_record_sha256s",
            "row_count",
            "source_table_count",
        )
        if any(field not in review_item for field in required_review_fields):
            raise AuthorityError(f"sealed document lacks V2 review evidence: {source_sha}")
        if review_item.get("review_status") != "frozen" or set(
            review_item.get("review_kinds", [])
        ) != {"independent_a", "independent_b", "adjudicator", "red_team"}:
            raise AuthorityError(f"sealed document review is not complete: {source_sha}")
        if not SHA256_RE.fullmatch(str(review_item["annotation_sha256"] or "")):
            raise AuthorityError(f"sealed document gold file identity is invalid: {source_sha}")
        if not SHA256_RE.fullmatch(str(review_item["gold_sha256"] or "")):
            raise AuthorityError(f"sealed document gold identity is invalid: {source_sha}")
        if not isinstance(review_item["review_ids"], list) or len(
            review_item["review_ids"]
        ) != 4:
            raise AuthorityError(f"sealed document review IDs are incomplete: {source_sha}")
        if not isinstance(review_item["review_record_sha256s"], list) or len(
            review_item["review_record_sha256s"]
        ) != 4 or any(
            not SHA256_RE.fullmatch(str(value)) for value in review_item["review_record_sha256s"]
        ):
            raise AuthorityError(
                f"sealed document review record identities are invalid: {source_sha}"
            )
        gold, _records = _validated_review_objects(
            vault,
            review_item,
            source_sha256=source_sha,
            rendered_document=render_documents[source_sha],
        )
        documents.append(
            {
                "source_sha256": source_sha,
                "cohorts": item["cohorts"],
                "annotation_sha256": review_item["annotation_sha256"],
                "gold_sha256": gold.gold_sha256,
                "gold_object_sha256": review_item["gold_object_sha256"],
                "gold_object_relpath": review_item["annotation_relpath"],
                "review_ids": review_item["review_ids"],
                "review_record_sha256s": review_item["review_record_sha256s"],
                "review_record_relpaths": review_item["review_record_relpaths"],
                "render_manifest_sha256": render_documents[source_sha]["manifest_sha256"],
                "page_hashes": [page["sha256"] for page in render_documents[source_sha]["pages"]],
                "row_count": review_item["row_count"],
                "source_table_count": review_item["source_table_count"],
            }
        )
    documents.sort(key=lambda item: (tuple(item["cohorts"]), item["source_sha256"]))
    corpus_identity = _canonical_digest(
        {
            "documents": [
                {"source_sha256": item["source_sha256"], "cohorts": item["cohorts"]}
                for item in documents
            ],
            "cohort_counts": counts,
        }
    )
    identity = _identity_from_options(
        identity_path=identity_path,
        release_revision=release_revision,
        images=image or [],
        models=model or [],
        config=config,
        corpus=corpus_identity,
        gold=review_digest,
        evaluator=evaluator,
    )
    if not _identity_complete(identity):
        raise AuthorityError(
            "authority seal identity must bind revision, images, models, config, corpus, "
            "gold, and evaluator"
        )
    payload = {
        "vault_version": VAULT_VERSION,
        "seal_version": SEAL_VERSION,
        "inventory_sha256": inventory_digest,
        "review_report_sha256": review_digest,
        "render_manifest_sha256": render_digest,
        "corpus_sha256": corpus_identity,
        "gold_sha256": review_digest,
        "cohort_counts": counts,
        "documents": documents,
        "identity": identity,
    }
    destination = _assert_external(
        output or _control_path(vault, "authority-seal.json"), label="seal output"
    )
    digest = _write_json(destination, payload)
    typer.echo(
        json.dumps({"seal": str(destination), "sha256": digest, "documents": len(documents)})
    )


def _read_report(
    vault: Path, names: tuple[str, ...], explicit: Path | None
) -> tuple[dict[str, Any], Path, str]:
    path = _find_control(vault, names, explicit)
    payload = _read_json(path, label="authority report")
    if not isinstance(payload, dict):
        raise AuthorityError("authority report must be an object")
    return payload, path, sha256_file(path)


@app.command("baseline")
def baseline(
    ctx: typer.Context,
    replay: Annotated[list[Path], typer.Option("--replay", "--machine-output")],
    identity_path: Annotated[Path | None, typer.Option("--identity")] = None,
    replay_digest: Annotated[list[str] | None, typer.Option("--replay-digest")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    vault = _effective_vault(ctx, vault_override)
    if len(replay) != 2:
        raise AuthorityError("baseline requires exactly two replay outputs")
    payloads: list[dict[str, Any] | None] = []
    digests: list[str] = []
    for path in replay:
        payload, digest, payload_dict = _load_machine_payload(path)
        _assert_machine_only(payload_dict)
        payloads.append(payload_dict)
        digests.append(digest)
        if isinstance(payload_dict, dict) and payload_dict.get("replay_digest") not in (
            None,
            digest,
        ):
            raise AuthorityError(f"machine output replay digest mismatch: {path}")
    if replay_digest and replay_digest != digests:
        raise AuthorityError("supplied replay digest does not match machine output")
    if digests[0] != digests[1]:
        raise AuthorityError("two baseline replays are not byte-identical")
    identity = _identity_from_options(identity_path=identity_path)
    payload_identities = [_machine_identity(payload) for payload in payloads]
    for candidate in payload_identities:
        if candidate and identity and not _identity_equal(candidate, identity):
            raise AuthorityError("machine output identity mismatch")
        if candidate:
            identity = _merge_identity(identity, candidate)
    if not _identity_complete(identity):
        raise AuthorityError(
            "baseline identity must bind revision, images, models, config, corpus, gold, "
            "and evaluator"
        )
    context = {
        "mode": "baseline",
        "vault": vault,
        "replay_paths": replay,
        "replay_digests": digests,
        "baseline_payloads": payloads,
        "candidate_payloads": [],
        "identity": identity,
    }
    metrics = _call_authority_metrics(context)
    if metrics is None:
        metrics = _fallback_metrics(payloads)
    metrics = _serialise_metrics(metrics)
    report = {
        "vault_version": VAULT_VERSION,
        "baseline_version": BASELINE_VERSION,
        "identity": identity,
        "replay_digests": digests,
        "replay_identical": True,
        "metrics": metrics,
        # Machine envelopes may expose diagnostics, but those values do not
        # become authority merely by appearing in a deterministic replay.
        # A sealed-gold evaluator report must set this in a later composed
        # baseline workflow before M0 can pass.
        "metrics_authoritative": False,
    }
    destination = _assert_external(
        output or _control_path(vault, "baseline.json"), label="baseline output"
    )
    digest = _write_json(destination, report)
    typer.echo(
        json.dumps({"baseline": str(destination), "sha256": digest, "replay_sha256": digests[0]})
    )


@app.command("evaluate")
def evaluate(
    ctx: typer.Context,
    baseline_path: Annotated[Path | None, typer.Option("--baseline")] = None,
    candidate: Annotated[list[Path] | None, typer.Option("--candidate", "--machine-output")] = None,
    identity_path: Annotated[Path | None, typer.Option("--identity")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    vault = _effective_vault(ctx, vault_override)
    baseline_payload, _baseline_file, baseline_digest = _read_report(
        vault, ("baseline.json",), baseline_path
    )
    if baseline_payload.get("baseline_version") != BASELINE_VERSION:
        raise AuthorityError("invalid authority baseline")
    expected_identity = _normalise_identity(baseline_payload.get("identity"))
    if not _identity_complete(expected_identity):
        raise AuthorityError("baseline identity is incomplete")
    if len(baseline_payload.get("replay_digests", [])) != 2 or not baseline_payload.get(
        "replay_identical"
    ):
        raise AuthorityError("baseline does not contain two verified replay digests")
    if not candidate:
        raise AuthorityError("evaluate requires at least one candidate machine output")
    candidate = list(candidate)
    candidate_digests: list[str] = []
    for path in candidate:
        _payload, digest, payload_dict = _load_machine_payload(path)
        _assert_machine_only(payload_dict)
        candidate_digests.append(digest)
        candidate_identity = _machine_identity(payload_dict)
        if candidate_identity and not _identity_equal(candidate_identity, expected_identity):
            raise AuthorityError("candidate machine output identity mismatch")
    override_identity = _identity_from_options(identity_path=identity_path)
    if override_identity and not _identity_equal(override_identity, expected_identity):
        raise AuthorityError("evaluation identity mismatch")
    # Candidate-controlled gold/metrics are never evaluation authority.  This
    # legacy command records the candidate identities but remains held until a
    # sealed-gold composition supplies an authoritative report (the M4 v2
    # evaluator does that directly).
    metrics = None
    passed = False
    report = {
        "vault_version": VAULT_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "baseline_sha256": baseline_digest,
        "candidate_digests": candidate_digests,
        "identity": expected_identity,
        "metrics": metrics,
        "passed": passed,
        "blocking_reasons": ["sealed_gold_evaluation_required"],
    }
    destination = _assert_external(
        output or _control_path(vault, "evaluation.json"), label="evaluation output"
    )
    _write_report_or_exit(destination, report, passed=passed)


def _validate_seal_chain(
    vault: Path,
    *,
    inventory: dict[str, Any],
    inventory_digest: str,
    review: dict[str, Any],
    review_digest: str,
    seal_payload: dict[str, Any],
) -> None:
    """Revalidate every object behind a seal rather than trusting summaries."""

    render, _render_path, render_digest = _load_render_manifest(vault)
    _validate_render_manifest_documents(vault, render)
    if (
        render.get("inventory_sha256") != inventory_digest
        or seal_payload.get("render_manifest_sha256") != render_digest
    ):
        raise AuthorityError("seal render/inventory identity mismatch")
    if (
        review.get("review_version") != REVIEW_VERSION
        or review.get("passed") is not True
        or review.get("inventory_sha256") != inventory_digest
        or review.get("render_manifest_sha256") != render_digest
        or seal_payload.get("review_report_sha256") != review_digest
    ):
        raise AuthorityError("seal review identity mismatch")
    inventory_documents = _inventory_documents(inventory)
    for document in inventory_documents:
        source = _safe_existing_file(
            vault / _assert_relative(document["object_relpath"], label="PDF object path"),
            root=vault,
            label="PDF object",
        )
        if sha256_file(source) != document["source_sha256"]:
            raise AuthorityError(f"sealed PDF object hash mismatch: {document['source_sha256']}")
    render_by_source = {
        item["document_sha256"]: item for item in render.get("documents", [])
    }
    summaries = review.get("annotations")
    if not isinstance(summaries, list):
        raise AuthorityError("seal review annotations are missing")
    review_by_source = {
        item.get("source_sha256"): item for item in summaries if isinstance(item, dict)
    }
    if len(review_by_source) != len(summaries):
        raise AuthorityError("seal review annotations contain duplicate/invalid identities")
    sealed_documents = seal_payload.get("documents")
    if not isinstance(sealed_documents, list):
        raise AuthorityError("seal documents are missing")
    seal_by_source = {
        item.get("source_sha256"): item for item in sealed_documents if isinstance(item, dict)
    }
    expected_sources = {item["source_sha256"] for item in inventory_documents}
    if (
        set(render_by_source) != expected_sources
        or set(review_by_source) != expected_sources
        or set(seal_by_source) != expected_sources
    ):
        raise AuthorityError("seal source sets do not match inventory")
    for document in inventory_documents:
        digest = document["source_sha256"]
        gold, _records = _validated_review_objects(
            vault,
            review_by_source[digest],
            source_sha256=digest,
            rendered_document=render_by_source[digest],
        )
        sealed = seal_by_source[digest]
        if (
            sealed.get("cohorts") != document["cohorts"]
            or sealed.get("gold_sha256") != gold.gold_sha256
            or sealed.get("gold_object_sha256")
            != review_by_source[digest].get("gold_object_sha256")
            or sealed.get("render_manifest_sha256")
            != render_by_source[digest].get("manifest_sha256")
        ):
            raise AuthorityError(f"sealed document summary mismatch: {digest}")
    counts = _validate_membership(inventory, require_complete=True)
    computed_corpus = _canonical_digest(
        {
            "documents": [
                {"source_sha256": item["source_sha256"], "cohorts": item["cohorts"]}
                for item in sorted(
                    inventory_documents,
                    key=lambda value: (tuple(value["cohorts"]), value["source_sha256"]),
                )
            ],
            "cohort_counts": counts,
        }
    )
    if seal_payload.get("corpus_sha256") != computed_corpus:
        raise AuthorityError("seal corpus identity does not match inventory")


@app.command("gate-m0-m4")
def gate_m0_m4(
    ctx: typer.Context,
    inventory_path: Annotated[Path | None, typer.Option("--inventory")] = None,
    review_report_path: Annotated[Path | None, typer.Option("--review-report")] = None,
    seal_path: Annotated[Path | None, typer.Option("--seal")] = None,
    baseline_path: Annotated[Path | None, typer.Option("--baseline")] = None,
    evaluation_path: Annotated[Path | None, typer.Option("--evaluation")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    milestone_report: Annotated[
        list[Path] | None, typer.Option("--milestone-report")
    ] = None,
    vault_override: Annotated[Path | None, typer.Option("--vault")] = None,
) -> None:
    vault = _effective_vault(ctx, vault_override)
    sections: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    inventory_digest: str | None = None
    review_digest: str | None = None
    seal_digest: str | None = None
    seal_payload: dict[str, Any] | None = None
    baseline_digest: str | None = None
    baseline_payload: dict[str, Any] | None = None
    baseline_authority: Any | None = None
    try:
        inventory, inventory_file, inventory_digest = _load_inventory(vault, inventory_path)
        counts = _validate_membership(inventory, require_complete=True)
        sections["m0"] = {
            "passed": True,
            "inventory_sha256": inventory_digest,
            "cohort_counts": counts,
        }
    except AuthorityError as error:
        sections["m0"] = {"passed": False, "blocking_reasons": [str(error)]}
        blockers.append("m0")
        inventory = {}
    try:
        review, review_file, review_digest = _read_report(
            vault, ("review-report.json",), review_report_path
        )
        passed = review.get("review_version") == REVIEW_VERSION and review.get("passed") is True
        sections["review"] = {"passed": passed, "sha256": review_digest}
        if not passed:
            blockers.append("review")
    except AuthorityError as error:
        sections["review"] = {"passed": False, "blocking_reasons": [str(error)]}
        blockers.append("review")
    try:
        seal_payload, seal_file, seal_digest = _read_report(
            vault, ("authority-seal.json",), seal_path
        )
        counts = (
            _validate_membership(seal_payload, require_complete=True)
            if "documents" in seal_payload
            else None
        )
        if inventory_digest is None or review_digest is None:
            raise AuthorityError("seal prerequisites are missing")
        _validate_seal_chain(
            vault,
            inventory=inventory,
            inventory_digest=inventory_digest,
            review=review,
            review_digest=review_digest,
            seal_payload=seal_payload,
        )
        seal_identity = _normalise_identity(seal_payload.get("identity"))
        seal_bound = (
            seal_payload.get("inventory_sha256") == inventory_digest
            and seal_payload.get("review_report_sha256") == review_digest
            and _identity_complete(seal_identity)
            and seal_identity.get("corpus_sha256") == seal_payload.get("corpus_sha256")
            and seal_identity.get("gold_sha256") == review_digest
        )
        sections["m2"] = {
            "passed": (
                seal_payload.get("seal_version") == SEAL_VERSION
                and counts is not None
                and seal_bound
            ),
            "sha256": seal_digest,
        }
        if not sections["m2"]["passed"]:
            blockers.append("m2")
    except AuthorityError as error:
        sections["m2"] = {"passed": False, "blocking_reasons": [str(error)]}
        blockers.append("m2")
    try:
        baseline_payload, baseline_file, baseline_digest = _read_report(
            vault, ("baseline.json",), baseline_path
        )
        if BaselineManifestV1 is None:
            raise AuthorityError("BaselineManifestV1 contract is unavailable")
        try:
            baseline_authority = BaselineManifestV1.model_validate(
                _validate_linked_json_object(vault, baseline_payload, prefix="authority_report")
            )
        except ValidationError as error:
            raise AuthorityError("invalid linked baseline authority report") from error
        replay_digests = baseline_payload.get("replay_digests", [])
        baseline_passed = (
            baseline_payload.get("baseline_version") == BASELINE_VERSION
            and baseline_payload.get("replay_identical") is True
            and len(replay_digests) == 2
            and len(set(replay_digests)) == 1
            and all(SHA256_RE.fullmatch(str(value)) for value in replay_digests)
            and _identity_complete(_normalise_identity(baseline_payload.get("identity")))
            and isinstance(baseline_payload.get("metrics"), dict)
            and baseline_payload.get("metrics_authoritative") is True
            and seal_payload is not None
            and _identity_equal(
                _normalise_identity(baseline_payload.get("identity")),
                _normalise_identity(seal_payload.get("identity")),
            )
            and baseline_authority.source_manifest_sha256
            == seal_payload.get("corpus_sha256")
            and baseline_authority.gold_manifest_sha256
            == seal_payload.get("gold_sha256")
            and baseline_authority.evaluator_manifest_sha256
            == _normalise_identity(seal_payload.get("identity")).get("evaluator_sha256")
            and {item.source_sha256 for item in baseline_authority.documents}
            == {
                item["source_sha256"]
                for item in seal_payload.get("documents", [])
                if "passing36" in item.get("cohorts", [])
            }
        )
        sections["m0_baseline"] = {"passed": baseline_passed, "sha256": baseline_digest}
        if not baseline_passed:
            blockers.append("m0_baseline")
    except AuthorityError as error:
        sections["m0_baseline"] = {"passed": False, "blocking_reasons": [str(error)]}
        blockers.append("m0_baseline")
    try:
        evaluation, evaluation_file, evaluation_digest = _read_report(
            vault, ("evaluation.json",), evaluation_path
        )
        evaluation_authority = _validate_linked_json_object(
            vault, evaluation, prefix="authority_report"
        )
        evaluation_passed = (
            evaluation.get("evaluation_version") == EVALUATION_VERSION
            and evaluation.get("passed") is True
            and baseline_digest is not None
            and evaluation.get("baseline_sha256") == baseline_digest
            and bool(evaluation.get("candidate_digests"))
            and isinstance(evaluation.get("metrics"), dict)
            and evaluation_authority.get("passed") is True
            and evaluation_authority.get("metrics") == evaluation.get("metrics")
            and baseline_payload is not None
            and _identity_equal(
                _normalise_identity(evaluation.get("identity")),
                _normalise_identity(baseline_payload.get("identity")),
            )
        )
        sections["evaluation"] = {"passed": evaluation_passed, "sha256": evaluation_digest}
        if not evaluation_passed:
            blockers.append("evaluation")
        typed_reports: dict[str, Any] = {}
        if M0M4GateReportV1 is not None:
            for path in milestone_report or []:
                try:
                    parsed = M0M4GateReportV1.model_validate(
                        _read_json(path, label="milestone gate report")
                    )
                except ValidationError as error:
                    raise AuthorityError(f"invalid milestone gate report: {path}") from error
                name = _enum_text(parsed.milestone).lower()
                if name in typed_reports:
                    raise AuthorityError(f"duplicate milestone gate report: {name}")
                typed_reports[name] = parsed
        for name in ("m1", "m3", "m4"):
            parsed = typed_reports.get(name)
            passed_report = (
                parsed is not None
                and _enum_text(parsed.decision) == "promote"
                and seal_payload is not None
                and baseline_payload is not None
                and baseline_authority is not None
                and parsed.source_manifest_sha256 == seal_payload.get("corpus_sha256")
                and parsed.gold_manifest_sha256 == seal_payload.get("gold_sha256")
                and parsed.evaluator_manifest_sha256
                == _normalise_identity(seal_payload.get("identity")).get("evaluator_sha256")
                and parsed.baseline_manifest_sha256 == baseline_authority.manifest_sha256
                and parsed.candidate_revision == evaluation.get("candidate_revision")
                and parsed.candidate_configuration_sha256
                == evaluation.get("candidate_configuration_sha256")
                and bool(parsed.metrics)
                and bool(parsed.evidence_sha256s)
            )
            sections[name] = {"passed": passed_report}
        for name in ("m1", "m3", "m4"):
            if not sections[name]["passed"]:
                blockers.append(name)
    except AuthorityError as error:
        sections["evaluation"] = {"passed": False, "blocking_reasons": [str(error)]}
        for name in ("evaluation", "m1", "m3", "m4"):
            sections.setdefault(name, {"passed": False})
            blockers.append(name)
    report = {
        "vault_version": VAULT_VERSION,
        "gate_version": GATE_VERSION,
        "m0_m4_passed": not blockers,
        "blocking_milestones": sorted(set(blockers)),
        "sections": sections,
    }
    destination = _assert_external(
        output or _control_path(vault, "gate-m0-m4.json"), label="gate output"
    )
    _write_report_or_exit(destination, report, passed=not blockers)


__all__ = [
    "AuthorityError",
    "COHORT_COUNTS",
    "DEFAULT_VAULT",
    "app",
    "baseline",
    "evaluate",
    "gate_m0_m4",
    "inventory",
    "render",
    "seal",
    "validate_review",
]


if __name__ == "__main__":
    app()
