"""Report serialization and summary helpers for validation runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from hc_validate import ValidationReport


def to_jsonable(value: Any) -> Any:
    """Convert common Python/PyTorch objects into JSON-serializable values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() <= 32:
            return {
                "type": "Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "values": value.detach().cpu().tolist(),
            }
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return repr(value)


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return to_jsonable(report.to_dict())


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _failure_preview(report: ValidationReport, limit: int = 10) -> list[str]:
    return list(report.failures[:limit])


def summarize_reports(reports: dict[str, ValidationReport], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    checks = []
    for name, report in reports.items():
        checks.append(
            {
                "name": name,
                "ok": bool(report.ok),
                "num_failures": len(report.failures),
                "num_warnings": len(report.warnings),
                "failure_preview": _failure_preview(report),
            }
        )
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "num_checks": len(checks),
        "num_failed_checks": len(failed),
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
        "metadata": to_jsonable(metadata or {}),
    }


def write_markdown_summary(path: str | Path, summary: dict[str, Any], reports: dict[str, ValidationReport]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# HyperCloning validation summary")
    lines.append("")
    lines.append(f"Overall result: **{'PASS' if summary.get('ok') else 'FAIL'}**")
    lines.append("")
    lines.append(f"Checks: {summary.get('num_checks', 0)}")
    lines.append(f"Failed checks: {summary.get('num_failed_checks', 0)}")
    lines.append("")
    metadata = summary.get("metadata", {})
    if metadata:
        lines.append("## Metadata")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.append("## Check results")
    lines.append("")
    lines.append("| Check | Result | Failures | Warnings |")
    lines.append("| --- | --- | ---: | ---: |")
    for check in summary.get("checks", []):
        result = "PASS" if check.get("ok") else "FAIL"
        lines.append(
            f"| `{check.get('name')}` | {result} | {check.get('num_failures', 0)} | {check.get('num_warnings', 0)} |"
        )
    lines.append("")
    for name, report in reports.items():
        if not report.failures and not report.warnings:
            continue
        lines.append(f"## `{name}` details")
        lines.append("")
        if report.failures:
            lines.append("### Failures")
            lines.append("")
            for failure in report.failures[:50]:
                lines.append(f"- {failure}")
            if len(report.failures) > 50:
                lines.append(f"- ... {len(report.failures) - 50} more failures")
            lines.append("")
        if report.warnings:
            lines.append("### Warnings")
            lines.append("")
            for warning in report.warnings[:50]:
                lines.append(f"- {warning}")
            if len(report.warnings) > 50:
                lines.append(f"- ... {len(report.warnings) - 50} more warnings")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report_bundle(
    output_dir: str | Path,
    reports: dict[str, ValidationReport],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write individual reports plus summary files into ``output_dir``."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_reports(reports, metadata=metadata)
    for name, report in reports.items():
        write_json(output_dir / f"{name}_report.json", report_to_dict(report))
    write_json(output_dir / "summary.json", summary)
    write_markdown_summary(output_dir / "summary.md", summary, reports)
    write_json(output_dir / "all_reports.json", {name: report_to_dict(report) for name, report in reports.items()})
    return summary
