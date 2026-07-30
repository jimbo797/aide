"""Accumulate OpenAI token usage from API responses during an evaluation run."""

from __future__ import annotations

import json
import threading
import requests
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

# USD per 1M tokens, keyed by model name (see aide/model_pricing.json).
_PRICING_PATH = Path(__file__).resolve().parents[2] / "model_pricing.json"


@lru_cache(maxsize=1)
def load_model_pricing() -> dict[str, dict[str, float]]:
    with open(_PRICING_PATH) as f:
        return json.load(f)
    # res = requests.get("https://www.llm-prices.com/current-v1.json", timeout=100)
    # res.raise_for_status()
    # return {m["id"]: m for m in res.json()["prices"]}


def tokens_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    model: str,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    """Cost in USD from token counts. Pricing rates are USD per 1M tokens."""
    rates = (pricing or load_model_pricing()).get(model)
    if not rates:
        return 0.0
    return (
        prompt_tokens * float(rates["input"])
        + completion_tokens * float(rates["output"])
    ) / 1_000_000


def _with_costs(by_model: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], float]:
    pricing = load_model_pricing()
    enriched: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    for model, bucket in by_model.items():
        prompt = int(bucket.get("prompt_tokens", 0) or 0)
        completion = int(bucket.get("completion_tokens", 0) or 0)
        cost = tokens_cost_usd(prompt, completion, model=model, pricing=pricing)
        total_cost += cost
        enriched[model] = {**bucket, "cost_usd": round(cost, 6)}
    return enriched, round(total_cost, 6)


class TokenUsageTracker:
    def __init__(self, *, alias: str | None = None) -> None:
        self.alias = alias
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        self.by_model: dict[str, dict[str, int]] = {}

    def record(self, usage: Any, *, model: str) -> None:
        if usage is None:
            return
        # Chat: prompt/completion; audio transcriptions: input/output tokens.
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0)
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0)
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
            self.api_calls += 1
            bucket = self.by_model.setdefault(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0,
                },
            )
            bucket["prompt_tokens"] += prompt
            bucket["completion_tokens"] += completion
            bucket["total_tokens"] += total
            bucket["api_calls"] += 1

    def to_dict(self) -> dict[str, Any]:
        by_model, cost_usd = _with_costs(self.by_model)
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "cost_usd": cost_usd,
            "by_model": by_model,
        }


_tracker: ContextVar[TokenUsageTracker | None] = ContextVar(
    "token_usage_tracker", default=None
)


def record_chat_usage(resp: Any, *, model: str) -> None:
    tracker = _tracker.get()
    if tracker is None:
        return
    tracker.record(getattr(resp, "usage", None), model=model)


@contextmanager
def track_token_usage(alias: str | None = None) -> Iterator[TokenUsageTracker]:
    """Accumulate usage for the current context. Nested calls reuse the outer tracker."""
    existing = _tracker.get()
    if existing is not None:
        yield existing
        return
    tracker = TokenUsageTracker(alias=alias)
    token = _tracker.set(tracker)
    try:
        yield tracker
    finally:
        _tracker.reset(token)


def student_costs_dict(
    alias: str,
    time_elapsed: timedelta | float,
    token_costs: dict[str, Any],
) -> dict[str, Any]:
    seconds = (
        time_elapsed.total_seconds()
        if isinstance(time_elapsed, timedelta)
        else float(time_elapsed)
    )
    return {
        "submission_alias": alias,
        "time_elapsed": seconds,
        "token_costs": token_costs,
    }


def aggregate_costs(per_student: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-student costs dicts into a class-level summary (no student list)."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    api_calls = 0
    by_model: dict[str, dict[str, int]] = {}
    total_time = 0.0

    for student in per_student:
        total_time += float(student.get("time_elapsed", 0) or 0)
        costs = student.get("token_costs") or {}
        prompt_tokens += int(costs.get("prompt_tokens", 0) or 0)
        completion_tokens += int(costs.get("completion_tokens", 0) or 0)
        total_tokens += int(costs.get("total_tokens", 0) or 0)
        api_calls += int(costs.get("api_calls", 0) or 0)
        for model, bucket in (costs.get("by_model") or {}).items():
            agg = by_model.setdefault(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0,
                },
            )
            agg["prompt_tokens"] += int(bucket.get("prompt_tokens", 0) or 0)
            agg["completion_tokens"] += int(bucket.get("completion_tokens", 0) or 0)
            agg["total_tokens"] += int(bucket.get("total_tokens", 0) or 0)
            agg["api_calls"] += int(bucket.get("api_calls", 0) or 0)

    by_model_with_costs, cost_usd = _with_costs(by_model)
    n = len(per_student)
    return {
        "total_time_elapsed": total_time,
        "average_time_elapsed": (total_time / n) if n else 0.0,
        "token_costs": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "api_calls": api_calls,
            "cost_usd": cost_usd,
            "by_model": by_model_with_costs,
        },
    }
