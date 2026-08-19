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
    "read_txt",
    "list_sources",
    "read_source",
]

TOOL_NAMES: tuple[ToolName, ...] = (
    "read_transcript",
    "search_transcript",
    "list_frame_summaries",
    "read_frame_summaries",
    "read_metadata",
    "read_excel",
    "read_txt",
    "list_sources",
    "read_source",
)

# Reference materials under ``<submissions_dir>/<alias>/sources/`` — not student work.
SOURCES_DIRNAME = "sources"


@dataclass
class SubmissionContext:
    alias: str
    preprocess_dir: Path
    transcript: dict[str, str]
    frame_summaries: dict[str, list[dict[str, Any]]]
    metadata: dict[str, dict[str, Any]]
    excel: dict[str, dict[str, Any]]
    txt: dict[str, str]
    sources: dict[str, str]

    @classmethod
    def load(
        cls,
        alias: str,
        preprocess_dir: Path,
        submissions_dir: Path | None = None,
    ) -> SubmissionContext:
        base = preprocess_dir / alias
        transcript: dict[str, str] = {}
        frame_summaries: dict[str, list[dict[str, Any]]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        excel: dict[str, dict[str, Any]] = {}
        txt: dict[str, str] = {}
        sources: dict[str, str] = {}

        def artifact_name(path: Path) -> str:
            relative_parent = path.parent.relative_to(base)
            return str(relative_parent) if relative_parent.parts else "submission"

        if base.is_dir():
            for path in sorted(base.rglob("transcript.txt")):
                transcript[artifact_name(path)] = path.read_text(encoding="utf-8")

            for path in sorted(base.rglob("frames_summary.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    frame_summaries[artifact_name(path)] = [
                        item for item in raw if isinstance(item, dict)
                    ]

            for path in sorted(base.rglob("metadata.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    metadata[artifact_name(path)] = raw

            for path in sorted(base.rglob("excel.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    excel[artifact_name(path)] = raw

            for path in sorted(base.rglob("content.txt")):
                txt[artifact_name(path)] = path.read_text(encoding="utf-8")

        if submissions_dir is not None:
            sources_dir = submissions_dir / alias / SOURCES_DIRNAME
            if sources_dir.is_dir():
                for path in sorted(sources_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    name = str(path.relative_to(sources_dir))
                    try:
                        sources[name] = path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        continue

        return cls(
            alias=alias,
            preprocess_dir=base,
            transcript=transcript,
            frame_summaries=frame_summaries,
            metadata=metadata,
            excel=excel,
            txt=txt,
            sources=sources,
        )


def available_tools(ctx: SubmissionContext) -> list[ToolName]:
    names: list[ToolName] = []
    if ctx.transcript:
        names.extend(["read_transcript", "search_transcript"])
    if ctx.frame_summaries:
        names.extend(["list_frame_summaries", "read_frame_summaries"])
    if ctx.metadata:
        names.extend(["read_metadata"])
    if ctx.excel:
        names.extend(["read_excel"])
    if ctx.txt:
        names.extend(["read_txt"])
    if ctx.sources:
        names.extend(["list_sources", "read_source"])
    return names


def _artifact_property(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": sorted(artifacts),
        "description": (
            "Artifact directory to read. May be omitted when only one is available."
        ),
    }


def _source_property(sources: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": sorted(sources),
        "description": (
            "Reference source filename under sources/ (NOT student work). "
            "May be omitted when only one source is available."
        ),
    }


def tool_schemas_for_context(ctx: SubmissionContext) -> list[dict[str, Any]]:
    """OpenAI function schemas for tools that have data for this submission."""
    schemas: list[dict[str, Any]] = []
    if ctx.transcript:
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
                                "artifact": _artifact_property(ctx.transcript),
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
                            "required": (
                                ["artifact"] if len(ctx.transcript) > 1 else []
                            ),
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
                                "artifact": _artifact_property(ctx.transcript),
                                "query": {"type": "string"},
                                "max_hits": {"type": "integer", "default": 15},
                            },
                            "required": (
                                ["query", "artifact"]
                                if len(ctx.transcript) > 1
                                else ["query"]
                            ),
                        },
                    },
                },
            ]
        )
    if ctx.frame_summaries:
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
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "artifact": _artifact_property(ctx.frame_summaries)
                            },
                            "required": (
                                ["artifact"] if len(ctx.frame_summaries) > 1 else []
                            ),
                        },
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
                                "artifact": _artifact_property(ctx.frame_summaries),
                                "frame_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "0-based frame indices (max 10 per call).",
                                },
                            },
                            "required": (
                                ["frame_indices", "artifact"]
                                if len(ctx.frame_summaries) > 1
                                else ["frame_indices"]
                            ),
                        },
                    },
                },
            ]
        )
    if ctx.metadata:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "read_metadata",
                    "description": (
                        "Read structural video metadata "
                        "(duration, resolution, codec, file size, bit rate, etc.)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "artifact": _artifact_property(ctx.metadata)
                        },
                        "required": ["artifact"] if len(ctx.metadata) > 1 else [],
                    },
                },
            }
        )
    if ctx.excel:
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
                            "artifact": _artifact_property(ctx.excel),
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
                        "required": ["artifact"] if len(ctx.excel) > 1 else [],
                    },
                },
            }
        )
    if ctx.txt:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "read_txt",
                    "description": (
                        "Read a slice of a student's submitted text file "
                        "(UTF-8 text by character offset)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "artifact": _artifact_property(ctx.txt),
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
                        "required": ["artifact"] if len(ctx.txt) > 1 else [],
                    },
                },
            }
        )
    if ctx.sources:
        schemas.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "list_sources",
                        "description": (
                            "List assignment reference source documents under sources/ "
                            "(filename, size, short preview). These are NOT part of the "
                            "student's submission — use them only as external context "
                            "(e.g. to check citations or compare claimed facts)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_source",
                        "description": (
                            "Read a slice of an assignment reference source document "
                            "(UTF-8 text by character offset). Content is NOT student work; "
                            "do not treat it as evidence of what the student submitted."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source": _source_property(ctx.sources),
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
                            "required": ["source"] if len(ctx.sources) > 1 else [],
                        },
                    },
                },
            ]
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


def _select_artifact(
    artifacts: dict[str, Any],
    args: dict[str, Any],
    artifact_type: str,
) -> tuple[str | None, Any | None, dict[str, Any] | None]:
    requested = args.get("artifact")
    if requested is None:
        if len(artifacts) == 1:
            name = next(iter(artifacts))
            return name, artifacts[name], None
        return None, None, {
            "ok": False,
            "error": f"artifact is required when multiple {artifact_type} artifacts exist",
            "available_artifacts": sorted(artifacts),
        }

    name = str(requested)
    if name not in artifacts:
        return None, None, {
            "ok": False,
            "error": f"Unknown {artifact_type} artifact: {name}",
            "available_artifacts": sorted(artifacts),
        }
    return name, artifacts[name], None


def dispatch_tool(ctx: SubmissionContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "read_transcript":
        if not ctx.transcript:
            return {"ok": False, "error": "No transcript available for this submission."}
        artifact, text, error = _select_artifact(
            ctx.transcript, args, "transcript"
        )
        if error is not None:
            return error
        assert isinstance(text, str)
        offset = int(args.get("offset") or 0)
        max_chars = int(args.get("max_chars") or 8000)
        if offset < 0 or offset > len(text):
            offset = 0
        chunk = text[offset : offset + max_chars]
        return {
            "ok": True,
            "artifact": artifact,
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(text),
            "truncated": offset + len(chunk) < len(text),
            "text": chunk,
        }

    if name == "search_transcript":
        if not ctx.transcript:
            return {"ok": False, "error": "No transcript available for this submission."}
        artifact, text, error = _select_artifact(
            ctx.transcript, args, "transcript"
        )
        if error is not None:
            return error
        assert isinstance(text, str)
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
        return {
            "ok": True,
            "artifact": artifact,
            "query": q,
            "hit_count": len(hits),
            "hits": hits,
        }

    if name == "list_frame_summaries":
        if not ctx.frame_summaries:
            return {"ok": False, "error": "No frame summaries available for this submission."}
        artifact, loaded, error = _select_artifact(
            ctx.frame_summaries, args, "frame summary"
        )
        if error is not None:
            return error
        assert isinstance(loaded, list)
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
        return {
            "ok": True,
            "artifact": artifact,
            "total": len(loaded),
            "frames": frames_out,
        }

    if name == "read_frame_summaries":
        if not ctx.frame_summaries:
            return {"ok": False, "error": "No frame summaries available for this submission."}
        artifact, loaded, error = _select_artifact(
            ctx.frame_summaries, args, "frame summary"
        )
        if error is not None:
            return error
        assert isinstance(loaded, list)
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
        return {"ok": True, "artifact": artifact, "frames": details}

    if name == "read_metadata":
        if not ctx.metadata:
            return {"ok": False, "error": "No metadata available for this submission."}
        artifact, loaded, error = _select_artifact(
            ctx.metadata, args, "metadata"
        )
        if error is not None:
            return error
        return {"ok": True, "artifact": artifact, "metadata": loaded}

    if name == "read_excel":
        if not ctx.excel:
            return {"ok": False, "error": "No excel content available for this submission."}
        artifact, loaded, error = _select_artifact(ctx.excel, args, "Excel")
        if error is not None:
            return error
        assert isinstance(loaded, dict)
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
            "artifact": artifact,
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

    if name == "read_txt":
        if not ctx.txt:
            return {"ok": False, "error": "No text file available for this submission."}
        artifact, text, error = _select_artifact(ctx.txt, args, "text")
        if error is not None:
            return error
        assert isinstance(text, str)
        offset = int(args.get("offset") or 0)
        max_chars = int(args.get("max_chars") or 8000)
        if offset < 0 or offset > len(text):
            offset = 0
        chunk = text[offset : offset + max_chars]
        return {
            "ok": True,
            "artifact": artifact,
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(text),
            "truncated": offset + len(chunk) < len(text),
            "text": chunk,
        }

    if name == "list_sources":
        if not ctx.sources:
            return {
                "ok": False,
                "error": "No reference sources available for this submission alias.",
            }
        items: list[dict[str, Any]] = []
        for source_name, text in sorted(ctx.sources.items()):
            preview = text[:280].replace("\n", " ")
            if len(text) > 280:
                preview = preview[:277] + "..."
            items.append(
                {
                    "source": source_name,
                    "total_chars": len(text),
                    "preview": preview,
                    "is_student_submission": False,
                }
            )
        return {
            "ok": True,
            "note": (
                "These are assignment reference materials, not part of the "
                "student's submission."
            ),
            "total": len(items),
            "sources": items,
        }

    if name == "read_source":
        if not ctx.sources:
            return {
                "ok": False,
                "error": "No reference sources available for this submission alias.",
            }
        # Reuse artifact selection keyed as ``source`` rather than ``artifact``.
        select_args = dict(args)
        if "source" in select_args and "artifact" not in select_args:
            select_args["artifact"] = select_args["source"]
        source_name, text, error = _select_artifact(
            ctx.sources, select_args, "source"
        )
        if error is not None:
            # Normalize error key name for callers.
            if "available_artifacts" in error:
                error = {
                    **error,
                    "available_sources": error.pop("available_artifacts"),
                }
            return error
        assert isinstance(text, str)
        offset = int(args.get("offset") or 0)
        max_chars = int(args.get("max_chars") or 8000)
        if offset < 0 or offset > len(text):
            offset = 0
        chunk = text[offset : offset + max_chars]
        return {
            "ok": True,
            "source": source_name,
            "is_student_submission": False,
            "note": (
                "This text is an assignment reference source, not student work."
            ),
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(text),
            "truncated": offset + len(chunk) < len(text),
            "text": chunk,
        }

    return {"ok": False, "error": f"Unknown tool: {name}"}
