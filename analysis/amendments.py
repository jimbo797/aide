"""Apply human point amendments in ``out/results/<alias>/results.json``.

Criterion amendments (``leaf_results[].amendment``: ``met`` or ``not_met``)
re-run category scoring instructions. Category amendments
(``aggregation.amendment``: replacement points) override the category total.
The new student score is the sum of (possibly amended) category points.

See ``AMENDMENTS.md`` for how to edit results files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

AIDE_DIR = Path(__file__).resolve().parent.parent
if str(AIDE_DIR) not in sys.path:
    sys.path.insert(0, str(AIDE_DIR))

from src.eval.eval import aggregate_category_results  # noqa: E402
from src.rubric.rubric_types import Rubric, RubricCategory  # noqa: E402

DEFAULT_RESULTS_DIR = AIDE_DIR / "out" / "results"
DEFAULT_RUBRIC = AIDE_DIR / "rubrics" / "gsu-summer-forecast.json"

LEAF_AMENDMENT_VALUES = {"met", "not_met"}


def _format_points(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _leaf_amendment(leaf: dict[str, Any]) -> str | None:
    raw = leaf.get("amendment")
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value not in LEAF_AMENDMENT_VALUES:
        raise ValueError(
            f"Invalid leaf amendment {raw!r}; expected 'met' or 'not_met'."
        )
    return value


def _category_amendment(aggregation: dict[str, Any]) -> float | None:
    if "amendment" not in aggregation:
        return None
    raw = aggregation["amendment"]
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid aggregation amendment {raw!r}; expected a number of points."
        ) from exc


def _leaves_for_rescoring(leaves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy leaves, substituting human criterion amendments as the verdict."""
    prepared: list[dict[str, Any]] = []
    for leaf in leaves:
        row = dict(leaf)
        amendment = _leaf_amendment(leaf)
        if amendment is not None:
            row["verdict"] = amendment
            row["evidence"] = (
                f"Human amendment: treat this criterion as {amendment}. "
                "Ignore the automated evaluation."
            )
        prepared.append(row)
    return prepared


def _category_has_leaf_amendments(leaves: list[dict[str, Any]]) -> bool:
    return any(_leaf_amendment(leaf) is not None for leaf in leaves)


def _rubric_category_by_description(rubric: Rubric) -> dict[str, RubricCategory]:
    return {category.description: category for category in rubric.categories}


def _preserve_original_points(category: dict[str, Any]) -> None:
    if "original_category_points" not in category:
        category["original_category_points"] = float(category.get("category_points") or 0.0)


def _preserve_original_aggregation(aggregation: dict[str, Any]) -> None:
    if "original_score" not in aggregation:
        aggregation["original_score"] = aggregation.get("score")
    if "original_reasoning" not in aggregation and "reasoning" in aggregation:
        aggregation["original_reasoning"] = aggregation.get("reasoning")


def apply_amendments_to_result(
    result: list[dict[str, Any]],
    rubric: Rubric,
    *,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], float, bool]:
    """Apply amendments in place. Returns (result, total score, whether anything changed)."""
    by_description = _rubric_category_by_description(rubric)
    changed = False
    total = 0.0

    for category in result:
        name = category.get("category") or ""
        leaves = category.get("leaf_results") or []
        aggregation = category.setdefault("aggregation", {})
        weight = float(category.get("weight") or aggregation.get("weight") or 0.0)

        category_override = _category_amendment(aggregation)
        has_leaf_amendments = _category_has_leaf_amendments(leaves)

        if category_override is not None:
            _preserve_original_points(category)
            category["category_points"] = category_override
            changed = True
        elif has_leaf_amendments:
            rubric_category = by_description.get(name)
            if rubric_category is None:
                raise KeyError(
                    f"Category {name!r} has criterion amendments but is not in the rubric."
                )
            _preserve_original_points(category)
            _preserve_original_aggregation(aggregation)
            new_aggregation = aggregate_category_results(
                _leaves_for_rescoring(leaves),
                rubric_category,
                model=model,
            )
            aggregation["score"] = new_aggregation["score"]
            aggregation["reasoning"] = new_aggregation["reasoning"]
            category["category_points"] = float(new_aggregation["score"]) * weight
            changed = True

        total += float(category.get("category_points") or 0.0)

    return result, total, changed


def apply_amendments(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    rubric_path: Path | str = DEFAULT_RUBRIC,
    *,
    model: str | None = None,
    dry_run: bool = False,
) -> list[tuple[str, float, bool]]:
    """Scan student result dirs, apply amendments, rewrite JSON and class_results.csv.

    Returns a list of ``(alias, score, amended)``.
    """
    results_dir = Path(results_dir)
    rubric = Rubric.model_validate_json(Path(rubric_path).read_text(encoding="utf-8"))

    class_rows: list[tuple[str, float, bool]] = []
    for student_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        result_file = student_dir / "results.json"
        if not result_file.is_file():
            continue

        alias = student_dir.name
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(result, list):
            continue

        updated, score, changed = apply_amendments_to_result(
            result, rubric, model=model
        )
        class_rows.append((alias, score, changed))

        if changed and not dry_run:
            result_file.write_text(
                json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if class_rows and not dry_run:
        csv_path = results_dir / "class_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["alias", "score"])
            for alias, score, _changed in class_rows:
                writer.writerow([alias, score])

    return class_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply human amendments in results.json and recompute scores."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory of per-student result folders (default: out/results).",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=DEFAULT_RUBRIC,
        help="Rubric JSON used to re-run scoring instructions.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model for re-aggregating categories with criterion amendments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scores without writing results.json or class_results.csv.",
    )
    args = parser.parse_args()

    os.chdir(AIDE_DIR)
    load_dotenv(AIDE_DIR / ".env")

    rows = apply_amendments(
        args.results_dir,
        args.rubric,
        model=args.model,
        dry_run=args.dry_run,
    )
    prefix = "Dry run: " if args.dry_run else ""
    print(f"{prefix}Processed {len(rows)} student result(s) under {args.results_dir}")
    for alias, score, changed in rows:
        flag = "amended" if changed else "unchanged"
        print(f"  {alias}: {_format_points(score)} ({flag})")


if __name__ == "__main__":
    main()
