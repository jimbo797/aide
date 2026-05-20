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
from src.util import log

AIDE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESS_DIR = AIDE_DIR / "preprocess-test"

LeafVerdict = Literal["met", "not_met", "undetermined"]


class PlannedToolCall(BaseModel):
    tool: ToolName
    rationale: str = Field(
        description="One sentence on why this tool helps judge the criterion."
    )
    offset: int | None = Field(default=None, description="For read_transcript.")
    max_chars: int | None = Field(default=None, description="For read_transcript.")
    query: str | None = Field(default=None, description="For search_transcript.")
    max_hits: int | None = Field(default=None, description="For search_transcript.")
    frame_indices: list[int] | None = Field(
        default=None, description="For read_frame_summaries."
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
    raw = resp.choices[0].message.content or "{}"
    return response_model.model_validate_json(raw)


def _plan_tools(
    client: OpenAI,
    *,
    model: str,
    leaf: RubricCriteria,
    ctx: SubmissionContext,
) -> ToolPlan:
    tools = available_tools(ctx)
    tool_docs = tool_schemas_for_context(ctx)
    if tools:
        tool_list = "\n".join(
            f"- {s['function']['name']}: {s['function']['description']}"
            for s in tool_docs
        )
    else:
        tool_list = "(none — no preprocessed transcript or frame summaries found for this submission)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are planning evidence gathering for a single rubric criterion. "
                "Choose only tools from the available list. Prefer targeted searches over "
                "reading entire transcripts when the criterion mentions specific concepts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Criterion:\n{leaf.description}\n\n"
                f"Submission alias: {ctx.alias}\n"
                f"Preprocess directory: {ctx.preprocess_dir}\n\n"
                f"Available tools:\n{tool_list}\n\n"
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
) -> EvalPlan:
    evidence_blob = json.dumps(
        {"tool_plan": tool_plan.model_dump(), "tool_results": tool_results},
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fair grader preparing to judge one atomic rubric criterion. "
                "Review the evidence from tools and explain how it bears on the criterion. "
                "Do not output a final verdict yet."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Criterion:\n{leaf.description}\n\n"
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
                "- undetermined: submission silent on this requirement or insufficient evidence"
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
    model: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate one rubric leaf against a preprocessed student submission.

    Phases:
    1. Plan which evidence tools to use for this criterion.
    2. Execute those tools against transcript / frame-summary artifacts.
    3. Plan how to judge the criterion from the tool results.
    4. Output met | not_met | undetermined with evidence.
    """

    pdir = Path(preprocess_dir) if preprocess_dir is not None else DEFAULT_PREPROCESS_DIR
    if not pdir.is_absolute():
        pdir = (AIDE_DIR / pdir).resolve()

    ctx = SubmissionContext.load(submission_alias, pdir)
    model_name = model or _default_model()
    client = OpenAIClient().client

    # log(submission_alias, "Planning tools")
    tool_plan = _plan_tools(client, model=model_name, leaf=leaf, ctx=ctx)

    # log(submission_alias, "Executing tools")
    tool_results = _execute_tool_plan(ctx, tool_plan)

    if not available_tools(ctx):
        eval_plan = EvalPlan(
            reasoning=(
                "No preprocessed transcript or frame summaries were found for this submission; "
                "cannot gather evidence."
            ),
            sufficient_to_judge=False,
        )
        verdict = LeafVerdictResult(
            verdict="undetermined",
            evidence=None,
            reasoning="Missing preprocess artifacts for this submission alias.",
        )
    else:
        # log(submission_alias, "Planning evaluation")
        eval_plan = _plan_evaluation(
            client,
            model=model_name,
            leaf=leaf,
            tool_plan=tool_plan,
            tool_results=tool_results,
        )
        # log(submission_alias, "Evaluating")
        verdict = _determine_verdict(
            client,
            model=model_name,
            leaf=leaf,
            eval_plan=eval_plan,
            tool_results=tool_results,
        )

    return {
        # "leaf_id": leaf.id,
        "criterion": leaf.description,
        "submission_alias": submission_alias,
        "preprocess_dir": str(pdir),
        "verdict": verdict.verdict,
        "evidence": verdict.evidence,
        "tool_plan": tool_plan.model_dump(),
        "tool_results": tool_results,
        "eval_plan": eval_plan.model_dump(),
        "verdict_reasoning": verdict.reasoning,
    }
