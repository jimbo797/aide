"""
Normalize heterogeneous rubric files into one structure consumed by ratas-rubric.py.

Supported inputs:
  - Normalized JSON (schema below) — canonical hand-editable format
  - CSV — columns for title, weight, body (see weighted_rubric.load_weighted_criteria_csv)
  - TXT / MD — heuristic blocks (see weighted_rubric.load_weighted_criteria_txt)

Output is always a NormalizedRubric with non-negative category weights and ordered categories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from weighted_rubric import (
    WeightedCriterion,
    load_weighted_criteria_csv,
    load_weighted_criteria_txt,
    merge_consecutive_same_weight,
)


class NormalizedCategory(BaseModel):
    """One top-level rubric category after normalization."""

    name: str = Field(description="Short title for this category (rubric line description).")
    weight: float = Field(ge=0, description="Points or weight stake for this category (non-negative).")
    body: str = Field(description="Full rubric text: requirements, sub-bullets, clarifications.")


class NormalizedRubric(BaseModel):
    """
    Canonical rubric shape for RATA skill-tree generation.

    JSON on disk example::

        {
          "schema_version": "1.0",
          "title": "Optional assignment name",
          "source_format": "csv",
          "categories": [
            { "name": "Content", "weight": 0.2, "body": "..." }
          ]
        }
    """

    schema_version: str = "1.0"
    title: str | None = None
    source_format: str | None = Field(
        default=None,
        description="Hint: csv | txt | normalized_json — informational only.",
    )
    categories: list[NormalizedCategory] = Field(min_length=1)

    def to_weighted_criteria(self) -> list[WeightedCriterion]:
        """Convert to the list type used by skill extraction / attach_weights."""
        return [
            WeightedCriterion(name=c.name, weight=c.weight, body=c.body)
            for c in self.categories
        ]

    @classmethod
    def from_weighted_criteria(
        cls,
        rows: list[WeightedCriterion],
        *,
        title: str | None = None,
        source_format: str | None = None,
    ) -> NormalizedRubric:
        return cls(
            source_format=source_format,
            title=title,
            categories=[
                NormalizedCategory(
                    name=r.name,
                    weight=abs(float(r.weight)),
                    body=r.body,
                )
                for r in rows
            ],
        )


def save_normalized_rubric(path: str | Path, rubric: NormalizedRubric) -> None:
    """Write normalized rubric JSON (UTF-8, pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        rubric.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )


def load_normalized_rubric(path: str | Path) -> NormalizedRubric:
    """Load a normalized rubric JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return NormalizedRubric.model_validate(data)


def _is_normalized_rubric_dict(data: Any) -> bool:
    """Distinguish normalized rubric JSON from RKT skill trees (type: root) and other JSON."""
    if not isinstance(data, dict) or data.get("type") == "root":
        return False
    cats = data.get("categories")
    if not isinstance(cats, list) or not cats:
        return False
    first = cats[0]
    return (
        isinstance(first, dict)
        and "name" in first
        and "weight" in first
        and "body" in first
    )


def _sniff_normalized_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    return _is_normalized_rubric_dict(data)


InputFormat = Literal["auto", "csv", "txt", "normalized"]


def normalize_rubric_file(
    path: str | Path,
    *,
    format: InputFormat = "auto",
    merge_consecutive_same_weight: bool = False,
    csv_body_column: str | None = None,
    csv_name_column: str = "Criteria",
    csv_weight_column: str = "Weight",
) -> NormalizedRubric:
    """
    Load any supported rubric file and return a NormalizedRubric.

    * format **auto**: ``.json`` with ``schema_version`` / ``categories`` → normalized;
      ``.csv`` → CSV loader; ``.txt`` / ``.md`` → TXT loader.
    * format **normalized**: require normalized JSON (validated).
    * format **csv** / **txt**: force that loader regardless of extension.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    fmt = format
    if fmt == "auto":
        if _sniff_normalized_json(path):
            fmt = "normalized"
        elif path.suffix.lower() == ".csv":
            fmt = "csv"
        elif path.suffix.lower() in (".txt", ".md"):
            fmt = "txt"
        elif path.suffix.lower() == ".json":
            raise ValueError(
                f"{path}: JSON is not a normalized rubric (expected schema_version 1.0 and "
                "a non-empty categories array). Use .csv/.txt source, or a file produced with "
                "--write-normalized."
            )
        else:
            raise ValueError(
                f"Cannot infer rubric format for {path.suffix}; use --format csv|txt|normalized"
            )

    if fmt == "normalized":
        nr = load_normalized_rubric(path)
        if merge_consecutive_same_weight:
            merged = merge_consecutive_same_weight(nr.to_weighted_criteria())
            return NormalizedRubric.from_weighted_criteria(
                merged,
                title=nr.title,
                source_format=nr.source_format or "normalized_json",
            )
        return nr

    if fmt == "csv":
        rows = load_weighted_criteria_csv(
            path,
            name_column=csv_name_column,
            weight_column=csv_weight_column,
            body_column=csv_body_column,
        )
    elif fmt == "txt":
        rows = load_weighted_criteria_txt(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if merge_consecutive_same_weight:
        rows = merge_consecutive_same_weight(rows)

    return NormalizedRubric.from_weighted_criteria(
        rows,
        title=None,
        source_format=fmt,
    )
