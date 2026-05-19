"""Evaluate a full submission against a rubric and aggregate per category."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from eval.eval_leaf import _default_model, _structured_completion, eval_leaf
from rubric.rubric_types import Rubric, RubricCategory


class CategoryAggregation(BaseModel):
    score: float = Field(
        ge=0,
        le=1,
        description=(
            "Fraction of this category's credit earned (0.0 = none, 1.0 = full category credit)."
        ),
    )
    reasoning: str = Field(
        description="How the scoring instructions were applied to the leaf verdicts."
    )


def _leaf_results_for_prompt(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "leaf_id": r.get("leaf_id"),
                "criterion": r.get("criterion"),
                "verdict": r.get("verdict"),
                "evidence": r.get("evidence"),
            }
        )
    return rows


def aggregate_category_results(
    results: list[dict[str, Any]],
    category: RubricCategory,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Use an OpenAI call to combine leaf verdicts into one category score.

    Returns a dict with ``score`` in [0, 1] (category credit fraction) and ``reasoning``.
    """
    if not results:
        return {
            "category": category.description,
            "weight": category.weight,
            "score": 0.0,
            "reasoning": "No leaf results to aggregate.",
        }

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_name = model or _default_model()
    client = OpenAI()
    leaves_blob = json.dumps(_leaf_results_for_prompt(results), ensure_ascii=False, indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a fair grader aggregating atomic criterion verdicts into one "
                "category score. Follow the scoring instructions exactly. "
                "Each leaf verdict is one of: met, not_met, undetermined."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Category: {category.description}\n"
                f"Category weight (percent of total grade): {category.weight}\n\n"
                f"Scoring instructions:\n{category.scoring_instructions}\n\n"
                f"Leaf results:\n{leaves_blob}\n\n"
                "Compute the category score as a fraction from 0.0 to 1.0 of the credit "
                "available for this category, following the scoring instructions. "
                "Explain your reasoning."
            ),
        },
    ]

    agg = _structured_completion(
        client,
        model=model_name,
        messages=messages,
        response_model=CategoryAggregation,
    )
    assert isinstance(agg, CategoryAggregation)

    return {
        "category": category.description,
        "weight": category.weight,
        "score": agg.score,
        "reasoning": agg.reasoning,
    }


def eval_submission(
    submission_alias: str,
    rubric: Rubric,
    *,
    preprocess_dir: Path | str | None = None,
    model: str | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluate every leaf, aggregate each category via OpenAI, and return total score.

    Total score assumes category ``weight`` values are percentage points (e.g. 25 = 25%)
    and each category ``score`` is in [0, 1].
    """
    final_score = 0.0
    final_results: list[dict[str, Any]] = []

    for category in rubric.categories:
        category_results: list[dict[str, Any]] = []
        for leaf in category.criteria:
            result = eval_leaf(
                leaf,
                submission_alias,
                preprocess_dir=preprocess_dir,
                model=model,
            )
            category_results.append(result)

        aggregation = aggregate_category_results(
            category_results, category, model=model
        )
        category_points = aggregation["score"] * category.weight
        final_score += category_points

        final_results.append(
            {
                "category": category.description,
                "weight": category.weight,
                "leaf_results": category_results,
                "aggregation": aggregation,
                "category_points": category_points,
            }
        )

    return final_score, final_results
