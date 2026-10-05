"""Compare table readers on a directory of bills: ``gmoney-compare``.

For each PDF and each table-reader mode the offline extractor, validation, and the
reconciliation gate are run, and ``report.json`` / ``report.md`` summarise per bill and
mode: pages, rows by role, reconciliation status with every failed check, reader
disagreements, seconds per page, TeleOCR/Gemini calls, and estimated cost per page.

Bills are named by an alias and a source-hash prefix only.  File names, hospital names,
patient header fields, and row descriptions are never written to the report.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

import typer

from gmoney.extraction.reader_pipeline import READER_MODES
from gmoney.extraction.reconciliation import reconcile
from gmoney.inference.paddle import TeleOcrAdapter
from gmoney.settings import get_settings

app = typer.Typer(add_completion=False, help=__doc__)
REPORT_VERSION = "table_reader_comparison_v1"
CONSENSUS_COUNTS = (
    "rows_compared",
    "rows_agreed",
    "amount_disagreements",
    "amount_resolved_by_total",
    "description_disagreements",
    "only_teleocr",
    "only_gemini",
)


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(0)


def _adapter_calls(result: dict[str, Any], model_name: str) -> dict[str, int]:
    calls = cache_hits = 0
    for diagnostic in result.get("diagnostics") or []:
        for item in diagnostic.get("adapter_inputs") or []:
            if item.get("adapter_name") != model_name:
                continue
            if item.get("cache_hit"):
                cache_hits += 1
            else:
                calls += 1
    return {"calls": calls, "cache_hits": cache_hits}


def summarize_run(
    result: dict[str, Any],
    *,
    mode: str,
    validation_status: str,
    seconds: float,
    inr_per_usd: float,
    gpu_cost_inr_per_hour: float = 0.0,
) -> dict[str, Any]:
    """Patient-free metrics for one bill under one reader mode."""
    pages = int(result.get("pages") or 0) or 1
    rows = result.get("rows") or []
    reconciliation = reconcile(result)
    consensus = Counter()
    ungrounded = 0
    guard_reads = 0
    for diagnostic in result.get("diagnostics") or []:
        counts = (diagnostic.get("reader_consensus") or {}).get("counts") or {}
        consensus.update({key: int(counts.get(key) or 0) for key in CONSENSUS_COUNTS})
        ungrounded += len(diagnostic.get("reader_ungrounded_rows") or [])
        guard_reads += bool(diagnostic.get("reader_full_page_guard"))
    usage = (result.get("provider_usage") or {}).get("aggregate") or {}
    gemini_cost_usd = _decimal(usage.get("gemini_measured_cost_usd"))
    gemini_inr_per_page = gemini_cost_usd * Decimal(str(inr_per_usd)) / pages
    seconds_per_page = seconds / pages
    gpu_inr_per_page = Decimal(str(gpu_cost_inr_per_hour)) * Decimal(str(seconds_per_page)) / 3600
    return {
        "mode": mode,
        "status": "ok",
        "pages": pages,
        "rows_by_role": dict(sorted(Counter(str(row.get("role")) for row in rows).items())),
        "rows_pending_review": sum(row.get("review_disposition") == "pending" for row in rows),
        "validation_status": validation_status,
        "reconciliation_status": reconciliation["status"],
        "rows_total": reconciliation["rows_total"],
        "failed_checks": [
            {
                key: check.get(key)
                for key in ("id", "kind", "page", "section", "label", "expected", "actual")
            }
            | {"difference": check.get("difference")}
            for check in reconciliation["checks"]
            if check["outcome"] == "fail" and check["blocking"]
        ],
        "rounded_checks": sum(check["outcome"] == "rounded" for check in reconciliation["checks"]),
        "row_arithmetic_mismatches": sum(
            check["kind"] == "row_arithmetic" and check["outcome"] == "fail"
            for check in reconciliation["checks"]
        ),
        "reader_disagreements": dict(consensus),
        "reader_ungrounded_rows": ungrounded,
        "full_page_guard_reads": guard_reads,
        "seconds": round(seconds, 2),
        "seconds_per_page": round(seconds_per_page, 2),
        "teleocr": _adapter_calls(result, TeleOcrAdapter.spec.model_name),
        "gemini_calls": int(usage.get("gemini_calls") or 0),
        "gemini_cost_usd": format(gemini_cost_usd, "f"),
        "estimated_inr_per_page": {
            "gemini": format(gemini_inr_per_page.quantize(Decimal("0.0001")), "f"),
            "gpu": format(gpu_inr_per_page.quantize(Decimal("0.0001")), "f"),
            "total": format(
                (gemini_inr_per_page + gpu_inr_per_page).quantize(Decimal("0.0001")), "f"
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    modes = report["modes"]
    lines = [
        "# Table reader comparison",
        "",
        f"Report `{report['report_version']}`; {len(report['bills'])} bill(s); "
        f"modes: {', '.join(modes)}. Bills are identified by alias and source-hash prefix only.",
        "",
        "## Summary",
        "",
        "| Bill | Pages | "
        + " | ".join(f"{mode} status | {mode} s/page | {mode} ₹/page" for mode in modes)
        + " |",
        "|---|---:|" + "---|---:|---:|" * len(modes),
    ]
    for bill in report["bills"]:
        cells = [bill["bill"], str(bill.get("pages") or "")]
        for mode in modes:
            run = bill["runs"].get(mode) or {}
            cells.extend(
                [
                    run.get("reconciliation_status") or run.get("status", "missing"),
                    str(run.get("seconds_per_page", "")),
                    str((run.get("estimated_inr_per_page") or {}).get("total", "")),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        ["", "## Totals by mode", "", "| Mode | Verified | Flagged | Unprovable | Errors |"]
    )
    lines.append("|---|---:|---:|---:|---:|")
    for mode in modes:
        statuses = Counter(
            (bill["runs"].get(mode) or {}).get("reconciliation_status")
            or (bill["runs"].get(mode) or {}).get("status")
            for bill in report["bills"]
        )
        lines.append(
            f"| {mode} | {statuses['verified']} | {statuses['flagged']} | "
            f"{statuses['unprovable']} | {statuses['error']} |"
        )
    for bill in report["bills"]:
        lines.extend(["", f"## {bill['bill']} (sha256 {bill['source_sha256_prefix']})", ""])
        for mode in modes:
            run = bill["runs"].get(mode) or {}
            lines.append(f"### {mode}")
            lines.append("")
            if run.get("status") == "error":
                lines.extend([f"- error: `{run.get('error')}`", ""])
                continue
            disagreements = run.get("reader_disagreements") or {}
            lines.extend(
                [
                    f"- pages: {run['pages']}; rows by role: "
                    + ", ".join(f"{role} {count}" for role, count in run["rows_by_role"].items()),
                    f"- validation: {run['validation_status']}; reconciliation: "
                    f"**{run['reconciliation_status']}** (rows total {run['rows_total']}; "
                    f"rounded checks {run['rounded_checks']}; row-arithmetic mismatches "
                    f"{run['row_arithmetic_mismatches']}, report-only)",
                    f"- seconds/page: {run['seconds_per_page']}; TeleOCR calls "
                    f"{run['teleocr']['calls']} (cache hits {run['teleocr']['cache_hits']}); "
                    f"Gemini calls {run['gemini_calls']} (USD {run['gemini_cost_usd']}); "
                    f"₹/page {run['estimated_inr_per_page']}",
                    f"- pending-review rows: {run['rows_pending_review']}; ungrounded reader rows: "
                    f"{run['reader_ungrounded_rows']}; full-page guard reads: "
                    f"{run['full_page_guard_reads']}",
                ]
            )
            if any(disagreements.values()):
                lines.append(
                    "- reader agreement: "
                    + ", ".join(f"{key} {value}" for key, value in disagreements.items())
                )
            if run["failed_checks"]:
                lines.extend(
                    [
                        "",
                        "| Check | Page | Section | Label | Expected | Actual | Difference |",
                        "|---|---:|---|---|---:|---:|---:|",
                    ]
                )
                for check in run["failed_checks"]:
                    lines.append(
                        "| "
                        + " | ".join(
                            str(check.get(key) if check.get(key) is not None else "")
                            for key in (
                                "id",
                                "page",
                                "section",
                                "label",
                                "expected",
                                "actual",
                                "difference",
                            )
                        )
                        + " |"
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _default_extractor(mode: str, options: dict[str, Any]) -> Any:
    from gmoney.extraction.offline import OfflineExtractor

    return OfflineExtractor(options["vl_url"], table_reader=mode, **options["extractor"])


def _validate(source: Path, result: dict[str, Any], artifact_root: Path) -> str:
    from gmoney.extraction.validation import validate_extraction_result

    return validate_extraction_result(source, result, artifact_root).status.value


def compare_bills(
    sources: list[Path],
    modes: list[str],
    output: Path,
    *,
    extractor_factory: Callable[[str, dict[str, Any]], Any] = _default_extractor,
    validator: Callable[[Path, dict[str, Any], Path], str] = _validate,
    options: dict[str, Any] | None = None,
    inr_per_usd: float = 88.0,
    gpu_cost_inr_per_hour: float = 0.0,
    echo: Callable[[str], None] = lambda _message: None,
) -> dict[str, Any]:
    options = options or {"vl_url": "http://127.0.0.1:8111", "extractor": {}}
    bills: list[dict[str, Any]] = []
    extractors: dict[str, Any] = {}
    for index, source in enumerate(sources, start=1):
        alias = f"bill-{index:02d}"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        bill: dict[str, Any] = {
            "bill": alias,
            "source_sha256_prefix": digest[:12],
            "pages": None,
            "runs": {},
        }
        for mode in modes:
            echo(f"{alias} {mode}: extracting")
            artifact_root = output / "runs" / alias / mode / "artifacts"
            started = time.perf_counter()
            try:
                if mode not in extractors:
                    extractors[mode] = extractor_factory(mode, options)
                result = extractors[mode].extract(source, artifact_root)
                result["source_sha256"] = digest
                result["source_name"] = f"{alias}.pdf"
                validation_status = validator(source, result, artifact_root)
                seconds = time.perf_counter() - started
                run = summarize_run(
                    result,
                    mode=mode,
                    validation_status=validation_status,
                    seconds=seconds,
                    inr_per_usd=inr_per_usd,
                    gpu_cost_inr_per_hour=gpu_cost_inr_per_hour,
                )
                bill["pages"] = run["pages"]
            except Exception as error:  # one failed run must not hide the others
                run = {
                    "mode": mode,
                    "status": "error",
                    "error": f"{type(error).__name__}",
                    "seconds": round(time.perf_counter() - started, 2),
                }
            bill["runs"][mode] = run
            echo(
                f"{alias} {mode}: {run.get('reconciliation_status') or run['status']} "
                f"({run.get('seconds_per_page', run.get('seconds'))} s/page)"
            )
        bills.append(bill)
    report = {"report_version": REPORT_VERSION, "modes": modes, "bills": bills}
    output.mkdir(parents=True, exist_ok=True)
    # Alias -> file name stays on the host; never send this file back with the report.
    (output / "aliases.local.json").write_text(
        json.dumps(
            {f"bill-{index:02d}": source.name for index, source in enumerate(sources, start=1)},
            indent=2,
        )
        + "\n"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(render_markdown(report))
    return report


@app.command()
def main(
    source_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    modes: Annotated[str, typer.Option(help="Comma-separated reader modes.")] = "heuristic",
    vl_url: str = "http://127.0.0.1:8111",
    paddle_device: str = "cpu",
    vl_device: str = "cpu",
    gpu_cost_inr_per_hour: Annotated[
        float, typer.Option(help="GPU host price, for the per-page cost estimate.")
    ] = 0.0,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    selected = [mode.strip() for mode in modes.split(",") if mode.strip()]
    unknown = sorted(set(selected) - set(READER_MODES))
    if unknown or not selected:
        raise typer.BadParameter(f"unknown modes: {', '.join(unknown) or '(none)'}")
    sources = sorted(path for path in source_dir.iterdir() if path.suffix.lower() == ".pdf")
    if limit:
        sources = sources[:limit]
    if not sources:
        raise typer.BadParameter("no PDF files found")
    compare_bills(
        sources,
        selected,
        output,
        options={
            "vl_url": vl_url,
            "extractor": {"paddle_device": paddle_device, "vl_device": vl_device},
        },
        inr_per_usd=get_settings().inr_per_usd,
        gpu_cost_inr_per_hour=gpu_cost_inr_per_hour,
        echo=typer.echo,
    )
    typer.echo(f"wrote {output / 'report.md'} and {output / 'report.json'}")


if __name__ == "__main__":
    app()
