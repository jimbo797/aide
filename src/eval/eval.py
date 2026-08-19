"""Evaluate a full submission against a rubric and aggregate per category."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from src.eval.eval_leaf import _default_model, _structured_completion, eval_leaf
from src.rubric.rubric_types import Rubric, RubricCategory, RubricCriteria
from src.util.token_usage import track_token_usage

DEFAULT_MAX_LEAF_CONCURRENCY = 4


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


async def _eval_leaves_concurrent(
    submission_alias: str,
    rubric: Rubric,
    *,
    preprocess_dir: Path | str | None,
    submissions_dir: Path | str | None,
    model: str | None,
    max_leaf_concurrency: int,
    max_loop_iters: int | None = None,
) -> list[tuple[int, int, dict[str, Any]]]:
    """Run all leaf evals concurrently; returns (category_index, leaf_index, result)."""
    sem = asyncio.Semaphore(max(1, max_leaf_concurrency))

    async def run_leaf(cat_i: int, leaf_i: int, leaf: RubricCriteria) -> tuple[int, int, dict[str, Any]]:
        async with sem:
            result = await asyncio.to_thread(
                eval_leaf,
                leaf,
                submission_alias,
                preprocess_dir=preprocess_dir,
                submissions_dir=submissions_dir,
                model=model,
                max_evidence_iterations=max_loop_iters,
            )
            result["id"] = leaf_i
            return cat_i, leaf_i, result

    return list(
        await asyncio.gather(
            *[
                run_leaf(cat_i, leaf_i, leaf)
                for cat_i, category in enumerate(rubric.categories)
                for leaf_i, leaf in enumerate(category.criteria)
            ]
        )
    )


def eval_submission(
    submission_alias: str,
    rubric: Rubric,
    *,
    preprocess_dir: Path | str | None = None,
    submissions_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    model: str | None = None,
    max_leaf_concurrency: int = DEFAULT_MAX_LEAF_CONCURRENCY,
    max_loop_iters: int | None = None,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """
    Evaluate every leaf, aggregate each category via OpenAI, and return total score.

    Leaf evaluations run concurrently via ``asyncio.gather`` (bounded by
    ``max_leaf_concurrency``). Category aggregation still runs sequentially after
    all leaves in that category finish.

    Total score assumes category ``weight`` values are percentage points (e.g. 25 = 25%)
    and each category ``score`` is in [0, 1].

    Returns ``(final_score, final_results, token_costs)``. When ``output_dir`` is set,
    full results are written to ``output_dir/<submission_alias>/results.json``.
    """
    with track_token_usage(submission_alias) as usage:
        leaf_rows = asyncio.run(
            _eval_leaves_concurrent(
                submission_alias,
                rubric,
                preprocess_dir=preprocess_dir,
                submissions_dir=submissions_dir,
                model=model,
                max_leaf_concurrency=max_leaf_concurrency,
                max_loop_iters=max_loop_iters,
            )
        )

        by_category: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for cat_i, leaf_i, result in leaf_rows:
            by_category.setdefault(cat_i, []).append((leaf_i, result))

        final_score = 0.0
        final_results: list[dict[str, Any]] = []

        for cat_i, category in enumerate(rubric.categories):
            category_results = [
                result for _, result in sorted(by_category.get(cat_i, []), key=lambda x: x[0])
            ]
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

        token_costs = usage.to_dict()

        if output_dir:
            output_dir = Path(output_dir)
            student_out = output_dir / submission_alias
            student_out.mkdir(parents=True, exist_ok=True)
            with open(student_out / "results.json", "w") as f:
                json.dump(final_results, f, indent=2)

    return final_score, final_results, token_costs
