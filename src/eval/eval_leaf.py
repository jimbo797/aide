"""Per-leaf rubric evaluation agent against preprocessed student submissions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from src.eval.eval_leaf_tools import (
    SubmissionContext,
    TOOL_NAMES,
    ToolName,
    available_tools,
    dispatch_tool,
    tool_schemas_for_context,
)
from src.rubric.rubric_types import RubricCriteria
from src.util.openai import OpenAIClient
from src.util.token_usage import record_chat_usage
from src.util import log

AIDE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESS_DIR = AIDE_DIR / "preprocess-test"
DEFAULT_SUBMISSIONS_DIR = AIDE_DIR / "in"
DEFAULT_MAX_EVIDENCE_ITERATIONS = 3

LeafVerdict = Literal["met", "not_met", "undetermined"]


class PlannedToolCall(BaseModel):
    tool: ToolName
    rationale: str = Field(
        description="One sentence on why this tool helps judge the criterion."
    )
    offset: int | None = Field(
        default=None, description="For read_transcript, read_excel, read_txt, or read_source."
    )
    max_chars: int | None = Field(
        default=None, description="For read_transcript, read_excel, read_txt, or read_source."
    )
    query: str | None = Field(default=None, description="For search_transcript.")
    max_hits: int | None = Field(default=None, description="For search_transcript.")
    frame_indices: list[int] | None = Field(
        default=None, description="For read_frame_summaries."
    )
    include_formulas: bool | None = Field(default=None, description="For read_excel.")
    include_charts: bool | None = Field(default=None, description="For read_excel.")
    include_chart_values: bool | None = Field(default=None, description="For read_excel.")
    source: str | None = Field(
        default=None,
        description="For read_source: filename under sources/ (reference material, not student work).",
    )

    def to_arguments(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if self.offset is not None:
            args["offset"] = self.offset
        if self.max_chars is not None:
            args["max_chars"] = self.max_chars
        if self.query is not None:
            args["query"] = self.query
        if self.max_hits is not None:
            args["max_hits"] = self.max_hits
        if self.frame_indices is not None:
            args["frame_indices"] = self.frame_indices
        if self.include_formulas is not None:
            args["include_formulas"] = self.include_formulas
        if self.include_charts is not None:
            args["include_charts"] = self.include_charts
        if self.include_chart_values is not None:
            args["include_chart_values"] = self.include_chart_values
        if self.source is not None:
            args["source"] = self.source
        return args


class ToolPlan(BaseModel):
    reasoning: str = Field(
        description="Brief plan: what evidence you need and which tools will provide it."
    )
    calls: list[PlannedToolCall] = Field(
        description="Ordered tool invocations to run before evaluating (may be empty if nothing is available)."
    )


class EvalPlan(BaseModel):
    reasoning: str = Field(
        description=(
            "How the tool results relate to the criterion; what is present, missing, or ambiguous."
        )
    )
    sufficient_to_judge: bool = Field(
        description="True if there is enough evidence to choose met or not_met (not undetermined)."
    )


class LeafVerdictResult(BaseModel):
    verdict: LeafVerdict
    evidence: str | None = Field(
        default=None,
        description="Short quote or paraphrase supporting met/not_met; optional for undetermined.",
    )
    reasoning: str = Field(description="One or two sentences tying evidence to the verdict.")


def _default_model() -> str:
    return os.environ.get("OPENAI_EVAL_LEAF_MODEL") or os.environ.get(
        "OPENAI_EVAL_AGENT_MODEL", "gpt-4o-mini"
    )


def _default_max_evidence_iterations() -> int:
    raw = (os.environ.get("OPENAI_EVAL_LEAF_MAX_ITERS") or "").strip()
    if not raw:
        return DEFAULT_MAX_EVIDENCE_ITERATIONS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_EVIDENCE_ITERATIONS


def _structured_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
) -> BaseModel:
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
        )
        record_chat_usage(resp, model=model)
        parsed = resp.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except Exception:
        pass

    schema_hint = json.dumps(response_model.model_json_schema(), indent=2)
    json_messages = messages + [
        {
            "role": "user",
            "content": (
                f"Respond with JSON only, matching this schema:\n{schema_hint}"
            ),
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=json_messages,
        response_format={"type": "json_object"},
    )
    record_chat_usage(resp, model=model)
    raw = resp.choices[0].message.content or "{}"
    return response_model.model_validate_json(raw)


def _plan_tools(
    client: OpenAI,
    *,
    model: str,
    leaf: RubricCriteria,
    ctx: SubmissionContext,
    iteration: int = 1,
    max_iterations: int = DEFAULT_MAX_EVIDENCE_ITERATIONS,
    prior_tool_results: list[dict[str, Any]] | None = None,
    prior_eval_plan: EvalPlan | None = None,
) -> ToolPlan:
    tools = available_tools(ctx)
    tool_docs = tool_schemas_for_context(ctx)
    if tools:
        tool_list = "\n".join(
            f"- {s['function']['name']}: {s['function']['description']}"
            for s in tool_docs
        )
    else:
        tool_list = (
            "(none — no preprocessed transcript, frame summaries, metadata, excel, "
            "text content, or reference sources found for this submission)"
        )

    prior_blob = ""
    if prior_tool_results or prior_eval_plan is not None:
        prior_payload: dict[str, Any] = {
            "prior_tool_results": prior_tool_results or [],
        }
        if prior_eval_plan is not None:
            prior_payload["prior_evaluation_notes"] = prior_eval_plan.model_dump()
        prior_blob = (
            "\n\nEvidence gathered so far (do not repeat identical tool calls; "
            "only plan additional calls that fill gaps):\n"
            f"{json.dumps(prior_payload, ensure_ascii=False, indent=2)}\n"
        )

    follow_up_hint = ""
    if iteration > 1:
        follow_up_hint = (
            "This is a follow-up evidence round. Prefer deeper reads of promising "
            "items already listed (e.g. read_frame_summaries after list_frame_summaries). "
            "If nothing useful remains, return an empty calls list.\n\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are planning evidence gathering for a single rubric criterion. "
                "Choose only tools from the available list. Prefer targeted searches over "
                "reading entire transcripts when the criterion mentions specific concepts. "
                "Evidence gathering may take multiple plan→execute rounds; plan only the "
                "calls needed for this round. "
                "list_sources and read_source access assignment reference materials under "
                "sources/ — these are NOT part of the student's submission; use them only "
                "as external context (e.g. citations or claimed facts), never as evidence "
                "of what the student wrote."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Criterion:\n{leaf.description}\n\n"
                f"Submission alias: {ctx.alias}\n"
                f"Preprocess directory: {ctx.preprocess_dir}\n"
                f"Evidence round: {iteration} of {max_iterations}\n\n"
                f"{follow_up_hint}"
                f"Available tools:\n{tool_list}"
                f"{prior_blob}\n"
                "Plan which tools to call and with what arguments. "
                "Do not evaluate yet — only plan evidence collection."
            ),
        },
    ]
    plan = _structured_completion(client, model=model, messages=messages, response_model=ToolPlan)

    valid = set(tools)
    filtered: list[PlannedToolCall] = []
    for call in plan.calls:
        if call.tool not in TOOL_NAMES:
            continue
        if call.tool not in valid:
            continue
        filtered.append(call)
    if len(filtered) > 12:
        filtered = filtered[:12]
    return ToolPlan(reasoning=plan.reasoning, calls=filtered)


def _execute_tool_plan(
    ctx: SubmissionContext, plan: ToolPlan
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in plan.calls:
        args = call.to_arguments()
        result = dispatch_tool(ctx, call.tool, args)
        results.append(
            {
                "tool": call.tool,
                "arguments": args,
                "rationale": call.rationale,
                "result": result,
            }
        )
    return results


def _plan_evaluation(
    client: OpenAI,
    *,
    model: str,
    leaf: RubricCriteria,
    tool_plan: ToolPlan,
    tool_results: list[dict[str, Any]],
    iteration: int = 1,
    max_iterations: int = DEFAULT_MAX_EVIDENCE_ITERATIONS,
) -> EvalPlan:
    evidence_blob = json.dumps(
        {
            "latest_tool_plan": tool_plan.model_dump(),
            "all_tool_results": tool_results,
        },
        ensure_ascii=False,
        indent=2,
    )
    more_rounds = iteration < max_iterations
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fair grader preparing to judge one atomic rubric criterion. "
                "Review the evidence from tools and explain how it bears on the criterion. "
                "Do not output a final verdict yet. "
                "Set sufficient_to_judge to true only if the evidence is enough to decide "
                "met or not_met. If key details are still missing (for example only frame "
                "previews were listed and full frame annotations were not read), set "
                "sufficient_to_judge to false and name what should be gathered next. "
                "Reference sources (list_sources / read_source) are not student work; "
                "do not credit the student for content that appears only in those sources."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Criterion:\n{leaf.description}\n\n"
                f"Evidence round: {iteration} of {max_iterations}\n"
                f"Another evidence round available after this: {more_rounds}\n\n"
                f"Evidence from tools:\n{evidence_blob}\n\n"
                "Explain how you would evaluate the student against this criterion "
                "based on the evidence. Say whether you have enough information to decide."
            ),
        },
    ]
    return _structured_completion(
        client, model=model, messages=messages, response_model=EvalPlan
    )


def _determine_verdict(
    client: OpenAI,
    *,
    model: str,
    leaf: RubricCriteria,
    eval_plan: EvalPlan,
    tool_results: list[dict[str, Any]],
) -> LeafVerdictResult:
    evidence_blob = json.dumps(
        {"eval_plan": eval_plan.model_dump(), "tool_results": tool_results},
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fair grader. Decide whether ONE atomic criterion is met, not met, "
                "or undetermined for this student submission.\n"
                "- met: clearly satisfied\n"
                "- not_met: enough information to judge and the criterion is not satisfied\n"
                "- undetermined: submission silent on this requirement or insufficient evidence\n"
                "Reference sources from list_sources/read_source are assignment materials, "
                "not the student's work — never treat them as evidence that the student met "
                "a criterion."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Criterion:\n{leaf.description}\n\n"
                f"Prior evaluation plan:\n{eval_plan.reasoning}\n"
                f"Sufficient to judge (prior step): {eval_plan.sufficient_to_judge}\n\n"
                f"Evidence:\n{evidence_blob}\n\n"
                "Output your final verdict."
            ),
        },
    ]
    return _structured_completion(
        client, model=model, messages=messages, response_model=LeafVerdictResult
    )

# Agent that evaluates a single leaf against a submission
def eval_leaf(
    leaf: RubricCriteria,
    submission_alias: str,
    *,
    preprocess_dir: Path | str | None = None,
    submissions_dir: Path | str | None = None,
    model: str | None = None,
    max_evidence_iterations: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate one rubric leaf against a preprocessed student submission.

    Iterates plan→execute evidence rounds (each with its own tool plan and
    execution), then judges sufficiency after each round. Stops early when
    evidence is sufficient or a follow-up plans no further tool calls.

    Phases per evidence round:
    1. Plan which evidence tools to use for this criterion.
    2. Execute those tools against transcript / frame-summary / excel / text
       artifacts, and optionally assignment reference sources.
    3. Assess whether the accumulated evidence is enough to judge.

    After the loop:
    4. Output met | not_met | undetermined with evidence.
    """

    pdir = Path(preprocess_dir) if preprocess_dir is not None else DEFAULT_PREPROCESS_DIR
    if not pdir.is_absolute():
        pdir = (AIDE_DIR / pdir).resolve()

    sdir = Path(submissions_dir) if submissions_dir is not None else DEFAULT_SUBMISSIONS_DIR
    if not sdir.is_absolute():
        sdir = (AIDE_DIR / sdir).resolve()

    ctx = SubmissionContext.load(submission_alias, pdir, submissions_dir=sdir)
    model_name = model or _default_model()
    client = OpenAIClient().client
    max_iters = (
        max(1, max_evidence_iterations)
        if max_evidence_iterations is not None
        else _default_max_evidence_iterations()
    )

    iterations: list[dict[str, Any]] = []
    all_tool_results: list[dict[str, Any]] = []
    tool_plan = ToolPlan(reasoning="No tools planned.", calls=[])
    eval_plan = EvalPlan(
        reasoning="No evaluation planned yet.",
        sufficient_to_judge=False,
    )

    if not available_tools(ctx):
        eval_plan = EvalPlan(
            reasoning=(
                "No preprocessed transcript, frame summaries, metadata, excel, text content, "
                "or reference sources were found for this submission; cannot gather evidence."
            ),
            sufficient_to_judge=False,
        )
        verdict = LeafVerdictResult(
            verdict="undetermined",
            evidence=None,
            reasoning="Missing preprocess artifacts for this submission alias.",
        )
    else:
        for iteration in range(1, max_iters + 1):
            tool_plan = _plan_tools(
                client,
                model=model_name,
                leaf=leaf,
                ctx=ctx,
                iteration=iteration,
                max_iterations=max_iters,
                prior_tool_results=all_tool_results or None,
                prior_eval_plan=eval_plan if iteration > 1 else None,
            )

            if iteration > 1 and not tool_plan.calls:
                iterations.append(
                    {
                        "iteration": iteration,
                        "tool_plan": tool_plan.model_dump(),
                        "tool_results": [],
                        "eval_plan": eval_plan.model_dump(),
                        "stopped_reason": "no_additional_tools_planned",
                    }
                )
                break

            round_results = _execute_tool_plan(ctx, tool_plan)
            all_tool_results.extend(round_results)

            eval_plan = _plan_evaluation(
                client,
                model=model_name,
                leaf=leaf,
                tool_plan=tool_plan,
                tool_results=all_tool_results,
                iteration=iteration,
                max_iterations=max_iters,
            )

            round_record: dict[str, Any] = {
                "iteration": iteration,
                "tool_plan": tool_plan.model_dump(),
                "tool_results": round_results,
                "eval_plan": eval_plan.model_dump(),
            }
            if eval_plan.sufficient_to_judge:
                round_record["stopped_reason"] = "sufficient_to_judge"
                iterations.append(round_record)
                break

            if not tool_plan.calls:
                round_record["stopped_reason"] = "no_tools_planned"
                iterations.append(round_record)
                break

            if iteration == max_iters:
                round_record["stopped_reason"] = "max_iterations"
            iterations.append(round_record)

        verdict = _determine_verdict(
            client,
            model=model_name,
            leaf=leaf,
            eval_plan=eval_plan,
            tool_results=all_tool_results,
        )

    return {
        # "leaf_id": leaf.id,
        "criterion": leaf.description,
        "submission_alias": submission_alias,
        "preprocess_dir": str(pdir),
        "verdict": verdict.verdict,
        "evidence": verdict.evidence,
        # Backward-compatible flat fields: last plan + all accumulated results.
        "tool_plan": tool_plan.model_dump(),
        "tool_results": all_tool_results,
        "eval_plan": eval_plan.model_dump(),
        "iterations": iterations,
        "verdict_reasoning": verdict.reasoning,
    }
