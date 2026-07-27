"""Write per-student point-loss breakdowns from evaluation results.

Reads ``out/results/<alias>/results.json`` and writes ``out/results/<alias>/report.txt``
listing each category where points were lost, with points lost and a brief reason.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

AIDE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = AIDE_DIR / "out" / "results"

# Ignore floating-point noise when deciding whether points were lost.
_POINTS_EPS = 1e-9


def _format_points(value: float) -> str:
    """Format points without unnecessary trailing zeros."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _first_sentence(text: str) -> str:
    """Keep the first complete sentence for a brief but intact debrief."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    for i, ch in enumerate(cleaned):
        if ch in ".!?" and i + 1 < len(cleaned) and cleaned[i + 1] == " ":
            return cleaned[: i + 1]
        if ch in ".!?" and i + 1 == len(cleaned):
            return cleaned
    return cleaned


def _loss_bullets(category: dict[str, Any]) -> list[str]:
    """Build brief skimmable bullets for unmet / undetermined criteria."""
    bullets: list[str] = []
    for leaf in category.get("leaf_results") or []:
        verdict = (leaf.get("verdict") or "").lower()
        if verdict not in {"not_met", "undetermined"}:
            continue

        criterion = " ".join(
            (leaf.get("criterion") or "Unnamed criterion").split()
        ).strip()
        why = _first_sentence(
            leaf.get("verdict_reasoning") or leaf.get("evidence") or ""
        )
        if not why:
            aggregation = category.get("aggregation") or {}
            why = _first_sentence(aggregation.get("reasoning") or "")

        if verdict == "undetermined":
            bullet = f"Could not verify: {criterion}"
            if why:
                bullet += f" — {why}"
        else:
            bullet = criterion
            if why:
                bullet += f" — {why}"
        bullets.append(bullet)

    if bullets:
        return bullets

    aggregation = category.get("aggregation") or {}
    fallback = _first_sentence(aggregation.get("reasoning") or "")
    return [fallback or "No scoring reasoning available."]


def point_losses_for_result(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return categories where the student lost points.

    Each item has ``category``, ``points_lost``, and ``bullets``.
    """
    losses: list[dict[str, Any]] = []
    for category in result:
        weight = float(category.get("weight") or 0.0)
        earned = float(category.get("category_points") or 0.0)
        lost = weight - earned
        if lost <= _POINTS_EPS:
            continue
        losses.append(
            {
                "category": category.get("category") or "Unknown category",
                "points_lost": lost,
                "weight": weight,
                "category_points": earned,
                "bullets": _loss_bullets(category),
            }
        )
    return losses


def format_point_loss_report(alias: str, losses: list[dict[str, Any]]) -> str:
    """Format a skimmable bullet-point report for one student."""
    lines = [f"Point losses for {alias}", ""]
    if not losses:
        lines.append("No points lost.")
        lines.append("")
        return "\n".join(lines)

    total_lost = sum(item["points_lost"] for item in losses)
    lines.append(f"Total points lost: {_format_points(total_lost)}")
    lines.append("")

    for item in losses:
        lines.append(
            f"{item['category']} (-{_format_points(item['points_lost'])} points)"
        )
        for bullet in item["bullets"]:
            lines.append(f"  - {bullet}")
        lines.append("")

    return "\n".join(lines)


def write_point_loss_analysis(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> dict[str, Path]:
    """Read class results and write ``report.txt`` into each student results dir.

    Returns a mapping from student alias to the written report path
    (``out/results/<alias>/report.txt``).
    """
    results_dir = Path(results_dir)

    written: dict[str, Path] = {}
    for student_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        result_file = student_dir / "results.json"
        if not result_file.is_file():
            continue

        alias = student_dir.name
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, list):
            continue

        losses = point_losses_for_result(result)
        report_path = student_dir / "report.txt"
        report_path.write_text(format_point_loss_report(alias, losses), encoding="utf-8")
        written[alias] = report_path

    return written


if __name__ == "__main__":
    os.chdir(AIDE_DIR)
    paths = write_point_loss_analysis()
    print(f"Wrote {len(paths)} point-loss report(s) under {DEFAULT_RESULTS_DIR}")
    for alias, path in paths.items():
        print(f"  {alias} -> {path}")
