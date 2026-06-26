"""Tools for per-leaf evaluation against preprocessed student submission artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ToolName = Literal[
    "read_transcript",
    "search_transcript",
    "list_frame_summaries",
    "read_frame_summaries",
    "read_metadata",
    "read_excel",
]

TOOL_NAMES: tuple[ToolName, ...] = (
    "read_transcript",
    "search_transcript",
    "list_frame_summaries",
    "read_frame_summaries",
    "read_metadata",
    "read_excel",
)


@dataclass
class SubmissionContext:
    alias: str
    preprocess_dir: Path
    transcript: str | None
    frame_summaries: list[dict[str, Any]] | None
    metadata: dict[str, Any] | None
    excel: dict[str, Any] | None

    @classmethod
    def load(cls, alias: str, preprocess_dir: Path) -> SubmissionContext:
        base = preprocess_dir / alias
        transcript_path = base / "transcript.txt"
        summaries_path = base / "frames_summary.json"
        metadata_path = base / "metadata.json"
        excel_path = base / "excel.json"

        transcript: str | None = None
        if transcript_path.is_file():
            transcript = transcript_path.read_text(encoding="utf-8")

        frame_summaries: list[dict[str, Any]] | None = None
        if summaries_path.is_file():
            raw = json.loads(summaries_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                frame_summaries = [x for x in raw if isinstance(x, dict)]

        metadata: dict[str, Any] | None = None
        if metadata_path.is_file():
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                metadata = raw

        excel: dict[str, Any] | None = None
        if excel_path.is_file():
            raw = json.loads(excel_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                excel = raw

        return cls(
            alias=alias,
            preprocess_dir=base,
            transcript=transcript,
            frame_summaries=frame_summaries,
            metadata=metadata,
            excel=excel,
        )


def available_tools(ctx: SubmissionContext) -> list[ToolName]:
    names: list[ToolName] = []
    if ctx.transcript is not None:
        names.extend(["read_transcript", "search_transcript"])
    if ctx.frame_summaries is not None:
        names.extend(["list_frame_summaries", "read_frame_summaries"])
    if ctx.metadata is not None:
        names.extend(["read_metadata"])
    if ctx.excel is not None:
        names.extend(["read_excel"])
    return names


def tool_schemas_for_context(ctx: SubmissionContext) -> list[dict[str, Any]]:
    """OpenAI function schemas for tools that have data for this submission."""
    schemas: list[dict[str, Any]] = []
    if ctx.transcript is not None:
        schemas.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_transcript",
                        "description": (
                            "Read a slice of the student's video transcript (UTF-8 text by character offset)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "offset": {
                                    "type": "integer",
                                    "description": "0-based character offset (default 0).",
                                    "default": 0,
                                },
                                "max_chars": {
                                    "type": "integer",
                                    "description": "Maximum characters to return (default 8000).",
                                    "default": 8000,
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_transcript",
                        "description": (
                            "Case-insensitive substring search in the transcript. "
                            "Returns snippets with short surrounding context."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "max_hits": {"type": "integer", "default": 15},
                            },
                            "required": ["query"],
                        },
                    },
                },
            ]
        )
    if ctx.frame_summaries is not None:
        schemas.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_frame_summaries",
                        "description": (
                            "List all preprocessed vision summaries of key video frames "
                            "(index, timestamp, short preview). Use before fetching full frame detail."
                        ),
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_frame_summaries",
                        "description": (
                            "Fetch full structured vision annotations for up to 10 frames by 0-based index."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "frame_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "0-based frame indices (max 10 per call).",
                                },
                            },
                            "required": ["frame_indices"],
                        },
                    },
                },
            ]
        )
    if ctx.metadata is not None:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "read_metadata",
                    "description": (
                        "Read structural video metadata "
                        "(duration, resolution, codec, file size, bit rate, etc.)."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    if ctx.excel is not None:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "read_excel",
                    "description": (
                        "Read preprocessed Excel sheet content: sheet name, formula cells, "
                        "chart metadata, and a slice of cell values as CSV text (by character offset). "
                        "When include_charts is true, each chart includes type, title, legend, position, "
                        "and per-series data refs plus trendline settings (type, polynomial order, "
                        "forecast periods, whether R²/equation are displayed, and cached label text)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "offset": {
                                "type": "integer",
                                "description": "0-based character offset into sheet_values_csv (default 0).",
                                "default": 0,
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum CSV characters to return (default 8000).",
                                "default": 8000,
                            },
                            "include_formulas": {
                                "type": "boolean",
                                "description": "Include formula_cells in the response (default true).",
                                "default": True,
                            },
                            "include_charts": {
                                "type": "boolean",
                                "description": (
                                    "Include chart metadata from preprocessing (default true), "
                                    "including trendline type, forecast periods, and R²/equation display settings."
                                ),
                                "default": True,
                            },
                            "include_chart_values": {
                                "type": "boolean",
                                "description": (
                                    "Include cached numeric values for each chart series (default false; "
                                    "cell values are usually available in sheet_values_csv)."
                                ),
                                "default": False,
                            },
                        },
                    },
                },
            }
        )
    return schemas


def _format_charts_for_tool(
    charts: list[Any], *, include_values: bool
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        series_out: list[dict[str, Any]] = []
        for series in chart.get("series") or []:
            if not isinstance(series, dict):
                continue
            entry = {k: v for k, v in series.items() if k != "values"}
            if include_values and "values" in series:
                entry["values"] = series["values"]
            series_out.append(entry)
        out.append(
            {
                "type": chart.get("type"),
                "title": chart.get("title"),
                "legend": chart.get("legend"),
                "anchor": chart.get("anchor"),
                "series": series_out,
            }
        )
    return out


def dispatch_tool(ctx: SubmissionContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "read_transcript":
        text = ctx.transcript
        if text is None:
            return {"ok": False, "error": "No transcript available for this submission."}
        offset = int(args.get("offset") or 0)
        max_chars = int(args.get("max_chars") or 8000)
        if offset < 0 or offset > len(text):
            offset = 0
        chunk = text[offset : offset + max_chars]
        return {
            "ok": True,
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(text),
            "truncated": offset + len(chunk) < len(text),
            "text": chunk,
        }

    if name == "search_transcript":
        text = ctx.transcript
        if text is None:
            return {"ok": False, "error": "No transcript available for this submission."}
        q = (args.get("query") or "").strip()
        if len(q) < 2:
            return {"ok": False, "error": "query must be at least 2 characters"}
        max_hits = min(int(args.get("max_hits") or 15), 50)
        low = text.casefold()
        qq = q.casefold()
        hits: list[dict[str, Any]] = []
        start = 0
        while len(hits) < max_hits:
            i = low.find(qq, start)
            if i < 0:
                break
            a = max(0, i - 100)
            b = min(len(text), i + len(q) + 100)
            snippet = text[a:b].replace("\n", " ")
            hits.append({"index": i, "snippet": snippet})
            start = i + 1
        return {"ok": True, "query": q, "hit_count": len(hits), "hits": hits}

    if name == "list_frame_summaries":
        loaded = ctx.frame_summaries
        if loaded is None:
            return {"ok": False, "error": "No frame summaries available for this submission."}
        frames_out: list[dict[str, Any]] = []
        for i, item in enumerate(loaded):
            ann = item.get("annotation") or {}
            so = str(ann.get("scene_overview") or "")
            if len(so) > 280:
                so = so[:277] + "..."
            frames_out.append(
                {
                    "index": i,
                    "time_seconds": item.get("time_seconds"),
                    "frame_path": item.get("frame_path"),
                    "preview": so,
                }
            )
        return {"ok": True, "total": len(loaded), "frames": frames_out}

    if name == "read_frame_summaries":
        loaded = ctx.frame_summaries
        if loaded is None:
            return {"ok": False, "error": "No frame summaries available for this submission."}
        raw_indices = args.get("frame_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            return {"ok": False, "error": "frame_indices must be a non-empty array of integers"}
        parsed: list[int] = []
        for x in raw_indices:
            try:
                parsed.append(int(x))
            except (TypeError, ValueError):
                return {"ok": False, "error": f"Invalid frame index: {x!r}"}
        seen: set[int] = set()
        uniq: list[int] = []
        for ix in parsed:
            if ix in seen:
                continue
            seen.add(ix)
            uniq.append(ix)
            if len(uniq) >= 10:
                break
        details: list[dict[str, Any]] = []
        for ix in uniq:
            if ix < 0 or ix >= len(loaded):
                details.append({"index": ix, "ok": False, "error": "index out of range"})
                continue
            item = loaded[ix]
            details.append(
                {
                    "index": ix,
                    "ok": True,
                    "time_seconds": item.get("time_seconds"),
                    "frame_path": item.get("frame_path"),
                    "annotation": item.get("annotation"),
                }
            )
        return {"ok": True, "frames": details}

    if name == "read_metadata":
        loaded = ctx.metadata
        if loaded is None:
            return {"ok": False, "error": "No metadata available for this submission."}
        return {"ok": True, "metadata": loaded}

    if name == "read_excel":
        loaded = ctx.excel
        if loaded is None:
            return {"ok": False, "error": "No excel content available for this submission."}
        csv_text = str(loaded.get("sheet_values_csv") or "")
        offset = int(args.get("offset") or 0)
        max_chars = int(args.get("max_chars") or 8000)
        include_formulas = args.get("include_formulas", True)
        include_charts = args.get("include_charts", True)
        include_chart_values = args.get("include_chart_values", False)
        if offset < 0 or offset > len(csv_text):
            offset = 0
        chunk = csv_text[offset : offset + max_chars]
        result: dict[str, Any] = {
            "ok": True,
            "sheet_name": loaded.get("sheet_name"),
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(csv_text),
            "truncated": offset + len(chunk) < len(csv_text),
            "sheet_values_csv": chunk,
        }
        if include_formulas:
            formulas: list[dict[str, str]] = []
            for item in loaded.get("formula_cells") or []:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    formulas.append({"cell": str(item[0]), "formula": str(item[1])})
            result["formula_cells"] = formulas
        if include_charts:
            raw_charts = loaded.get("charts") or []
            if isinstance(raw_charts, list):
                result["charts"] = _format_charts_for_tool(
                    raw_charts, include_values=bool(include_chart_values)
                )
            else:
                result["charts"] = []
        return result

    return {"ok": False, "error": f"Unknown tool: {name}"}
