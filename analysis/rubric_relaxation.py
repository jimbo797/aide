"""Suggest reasonable rubric leniency tweaks from class evaluation results.

Uses the same rubric + ``out/results/<alias>/results.json`` outputs produced by
``evaluate_class`` / ``eval_submission``. Flags criteria that fail often in a
way that looks stricter than the learning goal requires, without watering
criteria down into trivial checks.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

AIDE_DIR = Path(__file__).resolve().parent.parent
if str(AIDE_DIR) not in sys.path:
    sys.path.insert(0, str(AIDE_DIR))

from src.eval.eval_leaf import _default_model, _structured_completion  # noqa: E402
from src.rubric.rubric_types import Rubric  # noqa: E402

DEFAULT_RESULTS_DIR = AIDE_DIR / "out" / "results"

ChangeTarget = Literal["criterion", "scoring_instructions", "both", "none"]


class CriterionStats(BaseModel):
    category: str
    category_weight: float
    criterion: str
    n_students: int
    met: int
    not_met: int
    undetermined: int
    fail_rate: float
    sample_failures: list[dict[str, str]] = Field(default_factory=list)


class CategoryStats(BaseModel):
    category: str
    weight: float
    scoring_instructions: str
    criteria: list[CriterionStats]
    mean_fail_rate: float


class LeniencySuggestion(BaseModel):
    category: str
    target: ChangeTarget = Field(
        description=(
            "What to revise: criterion text, category scoring_instructions, both, "
            "or none if the bar is appropriately strict."
        )
    )
    criterion: str | None = Field(
        default=None,
        description="Original criterion text when the suggestion concerns a leaf criterion.",
    )
    fail_rate: float | None = Field(
        default=None,
        description="Observed not_met rate among matched submissions, if applicable.",
    )
    overly_strict: bool = Field(
        description=(
            "True if evidence shows the bar is harsher than the intended learning goal."
        )
    )
    issue: str = Field(
        description="What makes the current wording or scoring too strict (1-2 sentences)."
    )
    suggested_criterion: str | None = Field(
        default=None,
        description="Revised criterion text, if recommending a criterion change.",
    )
    suggested_scoring_instructions: str | None = Field(
        default=None,
        description="Revised scoring instructions, if recommending a scoring change.",
    )
    preserved_spirit: str = Field(
        description="The skill/intent that must still be required after the change."
    )
    rationale: str = Field(
        description="Why this change is fair and not trivializing the criterion."
    )


class RubricLeniencyReport(BaseModel):
    summary: str
    suggestions: list[LeniencySuggestion]
    category_stats: list[CategoryStats]


class _LLMLeniencyResponse(BaseModel):
    summary: str
    suggestions: list[LeniencySuggestion]


def _criterion_index(rubric: Rubric) -> dict[str, tuple[str, float, str]]:
    """Map criterion description -> (category, weight, scoring_instructions)."""
    index: dict[str, tuple[str, float, str]] = {}
    for category in rubric.categories:
        for leaf in category.criteria:
            index[leaf.description] = (
                category.description,
                category.weight,
                category.scoring_instructions,
            )
    return index


def _result_overlap(result: list[dict[str, Any]], rubric_criteria: set[str]) -> float:
    found: set[str] = set()
    for category in result:
        for leaf in category.get("leaf_results", []):
            crit = leaf.get("criterion")
            if crit:
                found.add(crit)
    if not rubric_criteria:
        return 0.0
    return len(found & rubric_criteria) / len(rubric_criteria)


def load_matching_results(
    results_dir: Path | str,
    rubric: Rubric,
    *,
    min_overlap: float = 0.9,
    aliases: set[str] | list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load student result JSONs whose criteria largely match ``rubric``.

    ``min_overlap`` defaults to 0.9 so mixed ``out/results`` folders (e.g. both
    carloan and forecast runs) only keep submissions evaluated on this rubric.
    Pass ``aliases`` to restrict further (e.g. from a class_results CSV).
    """
    results_dir = Path(results_dir)
    alias_filter = set(aliases) if aliases is not None else None
    rubric_criteria = {leaf.description for cat in rubric.categories for leaf in cat.criteria}
    matched: dict[str, list[dict[str, Any]]] = {}

    for student_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if alias_filter is not None and student_dir.name not in alias_filter:
            continue
        path = student_dir / "results.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        if _result_overlap(data, rubric_criteria) < min_overlap:
            continue
        matched[student_dir.name] = data

    return matched


def compute_rubric_stats(
    rubric: Rubric,
    results_by_alias: dict[str, list[dict[str, Any]]],
    *,
    max_samples_per_criterion: int = 4,
) -> list[CategoryStats]:
    """Aggregate met/not_met rates and sample failure evidence per criterion."""
    rubric_index = _criterion_index(rubric)
    buckets: dict[str, dict[str, Any]] = {}

    for criterion, (category, weight, scoring) in rubric_index.items():
        buckets[criterion] = {
            "category": category,
            "weight": weight,
            "scoring_instructions": scoring,
            "met": 0,
            "not_met": 0,
            "undetermined": 0,
            "samples": [],
        }

    for alias, result in results_by_alias.items():
        for category in result:
            for leaf in category.get("leaf_results", []):
                criterion = leaf.get("criterion")
                if criterion not in buckets:
                    continue
                verdict = leaf.get("verdict") or "undetermined"
                if verdict not in ("met", "not_met", "undetermined"):
                    verdict = "undetermined"
                buckets[criterion][verdict] += 1

                if (
                    verdict == "not_met"
                    and len(buckets[criterion]["samples"]) < max_samples_per_criterion
                ):
                    buckets[criterion]["samples"].append(
                        {
                            "alias": leaf.get("submission_alias") or alias,
                            "evidence": (leaf.get("evidence") or "")[:500],
                            "verdict_reasoning": (leaf.get("verdict_reasoning") or "")[:500],
                        }
                    )

    by_category: dict[str, list[CriterionStats]] = defaultdict(list)
    category_meta: dict[str, tuple[float, str]] = {}

    for criterion, (category, weight, scoring) in rubric_index.items():
        b = buckets[criterion]
        n = b["met"] + b["not_met"] + b["undetermined"]
        fail_rate = (b["not_met"] / n) if n else 0.0
        category_meta[category] = (weight, scoring)
        by_category[category].append(
            CriterionStats(
                category=category,
                category_weight=weight,
                criterion=criterion,
                n_students=n,
                met=b["met"],
                not_met=b["not_met"],
                undetermined=b["undetermined"],
                fail_rate=fail_rate,
                sample_failures=b["samples"],
            )
        )

    stats: list[CategoryStats] = []
    for category in [c.description for c in rubric.categories]:
        criteria = by_category.get(category, [])
        weight, scoring = category_meta.get(category, (0.0, ""))
        mean_fail = (
            sum(c.fail_rate for c in criteria) / len(criteria) if criteria else 0.0
        )
        stats.append(
            CategoryStats(
                category=category,
                weight=weight,
                scoring_instructions=scoring,
                criteria=criteria,
                mean_fail_rate=mean_fail,
            )
        )
    return stats


def _stats_payload_for_prompt(stats: list[CategoryStats]) -> list[dict[str, Any]]:
    """Compact JSON for the model: omit empty sample lists where possible."""
    payload: list[dict[str, Any]] = []
    for cat in stats:
        payload.append(
            {
                "category": cat.category,
                "weight": cat.weight,
                "scoring_instructions": cat.scoring_instructions,
                "mean_fail_rate": round(cat.mean_fail_rate, 3),
                "criteria": [
                    {
                        "criterion": c.criterion,
                        "n_students": c.n_students,
                        "met": c.met,
                        "not_met": c.not_met,
                        "undetermined": c.undetermined,
                        "fail_rate": round(c.fail_rate, 3),
                        "sample_failures": c.sample_failures,
                    }
                    for c in cat.criteria
                ],
            }
        )
    return payload


def suggest_rubric_leniency(
    rubric: Rubric,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    model: str | None = None,
    min_overlap: float = 0.9,
    aliases: set[str] | list[str] | None = None,
    max_samples_per_criterion: int = 4,
    focus_fail_rate: float = 0.4,
) -> RubricLeniencyReport:
    """Analyze class eval results and suggest reasonable rubric leniency fixes.

    Parameters
    ----------
    rubric:
        The ``Rubric`` used by ``evaluate_class``.
    results_dir:
        Directory of per-student ``<alias>/results.json`` eval outputs (default:
        ``out/results``).
    model:
        OpenAI model name; defaults to the same env / leaf-eval default.
    min_overlap:
        Minimum fraction of rubric criteria that must appear in a result file
        for that student to be included (filters mixed-assignment result dirs).
    aliases:
        Optional student alias allow-list (e.g. from a class results CSV).
    max_samples_per_criterion:
        Cap on failure evidence quotes sent to the model per criterion.
    focus_fail_rate:
        Soft hint: criteria failing above this rate deserve extra scrutiny.
        The model may still flag lower rates when wording is needlessly brittle.

    Returns
    -------
    RubricLeniencyReport
        Stats plus concrete suggestions that stay aligned with the spirit of
        each criterion (not trivializing).
    """
    results_by_alias = load_matching_results(
        results_dir, rubric, min_overlap=min_overlap, aliases=aliases
    )
    if not results_by_alias:
        raise ValueError(
            f"No result JSON files in {results_dir!s} matched the given rubric "
            f"(min_overlap={min_overlap})."
        )

    category_stats = compute_rubric_stats(
        rubric,
        results_by_alias,
        max_samples_per_criterion=max_samples_per_criterion,
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_name = model or _default_model()
    client = OpenAI()
    stats_blob = json.dumps(
        _stats_payload_for_prompt(category_stats), ensure_ascii=False, indent=2
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You revise grading rubrics to be fairly lenient when evaluation "
                "data shows criteria are too strict, while preserving the intended "
                "learning goals.\n\n"
                "Principles:\n"
                "- Stick to the spirit of each criterion: what skill or behavior "
                "the instructor meant to reward.\n"
                "- Soften brittle wording (exact cell refs when nearby is fine, "
                "exact phrasings, all-or-nothing lists, 'sophisticated' bars, "
                "cosmetic labeling) when students demonstrate the underlying skill.\n"
                "- Do NOT make criteria trivial. Do not remove the core requirement "
                "(e.g. still require demonstrating IPMT, narration, charts, etc.).\n"
                "- High fail rate alone is not enough to change a criterion if "
                "students simply omitted the skill; say overly_strict=false and "
                "target=none, and explain.\n"
                "- Weight-0 categories or instructions that say 'always award 0' "
                "are intentional non-scoring items; usually leave them alone.\n"
                "- Prefer the smallest wording/scoring change that would fairly "
                "award more points to students who showed substantial effort on "
                "the skill.\n"
                f"- Pay special attention to criteria with fail_rate >= "
                f"{focus_fail_rate}, and to harsh all-or-nothing scoring_instructions.\n"
                "- Only emit suggestions worth acting on; for criteria that should "
                "stay strict, include a short entry with target=none and "
                "overly_strict=false when fail_rate is high, so the caller knows "
                "you reviewed them."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Number of matched student submissions: {len(results_by_alias)}\n"
                f"Aliases: {', '.join(sorted(results_by_alias))}\n\n"
                "Per-category evaluation stats (verdicts + sample not_met evidence):\n"
                f"{stats_blob}\n\n"
                "Return a short overall summary and a list of suggestions. For each "
                "suggestion, if revising a criterion give suggested_criterion; if "
                "revising scoring give suggested_scoring_instructions; set target "
                "accordingly."
            ),
        },
    ]

    raw = _structured_completion(
        client,
        model=model_name,
        messages=messages,
        response_model=_LLMLeniencyResponse,
    )
    assert isinstance(raw, _LLMLeniencyResponse)

    return RubricLeniencyReport(
        summary=raw.summary,
        suggestions=raw.suggestions,
        category_stats=category_stats,
    )


def analyze_rubric_leniency(
    rubric: Rubric,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    model: str | None = None,
    output_path: Path | str | None = None,
    **kwargs: Any,
) -> RubricLeniencyReport:
    """Run leniency analysis and optionally write the full report JSON to disk.

    Accepts the same keyword args as ``suggest_rubric_leniency`` (``aliases``,
    ``min_overlap``, ``focus_fail_rate``, etc.).
    """
    report = suggest_rubric_leniency(
        rubric, results_dir, model=model, **kwargs
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    # Example: carloan rubric against out/results (skips forecast-only JSONs).
    json_data = (AIDE_DIR / "rubrics" / "gsu-spring-carloan-manual.json").read_text(
        encoding="utf-8"
    )
    rubric = Rubric.model_validate_json(json_data)

    report = analyze_rubric_leniency(
        rubric,
        results_dir=DEFAULT_RESULTS_DIR,
        model=os.environ.get("OPENAI_EVAL_AGENT_MODEL", "gpt-5.5"),
        output_path=DEFAULT_RESULTS_DIR / "carloan-rubric-leniency.json",
    )

    print(report.summary)
    print()
    for s in report.suggestions:
        flag = "STRICT" if s.overly_strict else "ok"
        print(f"[{flag}] {s.category}" + (f" — {s.criterion[:80]}..." if s.criterion and len(s.criterion) > 80 else (f" — {s.criterion}" if s.criterion else "")))
        print(f"  target={s.target}  fail_rate={s.fail_rate}")
        print(f"  issue: {s.issue}")
        if s.suggested_criterion:
            print(f"  suggested criterion: {s.suggested_criterion}")
        if s.suggested_scoring_instructions:
            print(f"  suggested scoring: {s.suggested_scoring_instructions}")
        print(f"  spirit: {s.preserved_spirit}")
        print()
