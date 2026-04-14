#!/usr/bin/env python3
"""
Per-leaf agentic evaluation against a pre-built RKT skill tree.

1. Build the RKT elsewhere (e.g. ``python3 ratas-rubric.py rubrics/foo.csv``) so you have
   a skill-tree JSON (type: root).
2. Load that tree once, flatten to atomic leaves.
3. For each student submission, for each leaf, run a small tool loop: the model may call
   read_submission / search_submission, optionally get_video_metadata (when a local video path
   or YouTube URL is configured for the run), then must call submit_leaf_verdict for that leaf.

For programmatic tree construction from CSV/TXT (without the agent), use
``materialize_rubric_tree`` from this module — same pipeline as ``ratas-rubric.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from request import RubricLeafRef, flatten_rubric_leaves, rubric_skill_extract, rubric_skill_tree_construct
from rubric_normalize import InputFormat, normalize_rubric_file
from rkt_io import load_skill_tree
from type import LeafVerdict, RKTRoot
from weighted_rubric import (
    attach_weights_from_rows,
    format_categories_for_tree,
    format_for_skill_extract,
    normalize_rkt_positive_weights,
)
from video_metadata import (
    ffprobe_available,
    looks_like_youtube_url,
    probe_video_metadata_for_tool,
    yt_dlp_available,
)

load_dotenv()


def _sniff_rkt_skill_tree_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    return isinstance(data, dict) and data.get("type") == "root" and isinstance(
        data.get("rows"), list
    )


def load_rkt_json(path: str | Path) -> RKTRoot:
    """Load a pre-generated RKT skill tree JSON. Use ``ratas-rubric.py`` to create it."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if not _sniff_rkt_skill_tree_json(p):
        raise ValueError(
            f"{p}: not an RKT skill tree (expected JSON with type 'root' and 'rows'). "
            "Generate one with: python3 ratas-rubric.py <rubric.csv|txt|normalized.json>"
        )
    return load_skill_tree(p)


def materialize_rubric_tree(
    rubric_path: str | Path,
    *,
    input_format: InputFormat = "auto",
    merge_consecutive_same_weight: bool = False,
    csv_body_column: str | None = None,
    csv_name_column: str = "Criteria",
    csv_weight_column: str = "Weight",
    no_merge_weights: bool = False,
    no_sync_descriptions: bool = False,
) -> RKTRoot:
    """
    Turn a rubric source file into an RKTRoot (for use **outside** the per-leaf agent).

    - If the file is already RKT JSON (type: root), loads it.
    - Otherwise normalizes CSV/TXT/normalized rubric JSON and runs skill extract + tree
      construction (same as ``ratas-rubric.py``).
    """
    path = Path(rubric_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sniff_rkt_skill_tree_json(path):
        return load_skill_tree(path)

    normalized = normalize_rubric_file(
        path,
        format=input_format,
        merge_consecutive_same_weight=merge_consecutive_same_weight,
        csv_body_column=csv_body_column,
        csv_name_column=csv_name_column,
        csv_weight_column=csv_weight_column,
    )
    rows = normalized.to_weighted_criteria()
    if not rows:
        raise ValueError(f"No categories in normalized rubric: {path}")

    rubric_str = format_for_skill_extract(rows)
    categories_block = format_categories_for_tree(rows)
    skills = rubric_skill_extract(rubric_str)
    skill_tree = rubric_skill_tree_construct(skills, categories_block=categories_block)
    if not no_merge_weights:
        skill_tree = attach_weights_from_rows(
            skill_tree,
            rows,
            sync_descriptions=not no_sync_descriptions,
        )
    return normalize_rkt_positive_weights(skill_tree)


def _leaf_row(L: RubricLeafRef, verdict: LeafVerdict, evidence: str | None) -> dict[str, Any]:
    return {
        "leaf_id": L.leaf_id,
        "category": L.category,
        "category_weight": L.category_weight,
        "basic_group": L.basic_group,
        "criterion": L.rule_text,
        "leaf_weight": L.leaf_weight,
        "verdict": verdict,
        "evidence": evidence,
    }


@dataclass
class YouTubeMediaConfig:
    """Fixed YouTube URL and optional yt-dlp auth (same ideas as ``youtube_transcribe.py``)."""

    url: str
    cookiefile: Path | None = None
    cookiesfrombrowser: tuple[str, ...] | None = None
    youtube_player_client: str | None = None
    youtube_player_fallback: bool = True


@dataclass
class PerLeafAgentState:
    """One leaf evaluation: submission text + fixed leaf + slot for the final tool result."""

    submission_text: str
    leaf: RubricLeafRef
    verdict_result: dict[str, Any] | None = None
    #: Local file (mp4, mov, …) for ``get_video_metadata`` via ffprobe.
    video_path: Path | None = None
    #: YouTube URL for ``get_video_metadata`` via yt-dlp (no download). Mutually exclusive with ``video_path``.
    youtube: YouTubeMediaConfig | None = None
    #: Shared across leaves so ``get_video_metadata`` only hits ffprobe / YouTube once per run.
    metadata_cache: dict[str, Any] | None = None


def _per_leaf_tool_schemas(*, include_video_metadata: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_submission",
                "description": "Read a slice of the student submission as UTF-8 text (by character offset).",
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
                "name": "search_submission",
                "description": (
                    "Case-insensitive substring search in the submission. "
                    "Returns up to max_hits snippets with short context."
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
    if include_video_metadata:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "get_video_metadata",
                    "description": (
                        "Read technical metadata for the student's video without downloading it. "
                        "For a local file (configured path): ffprobe — duration, resolution, codecs, bitrate. "
                        "For a YouTube URL (configured for this run): yt-dlp extract_info — duration, title, "
                        "video id, resolution/fps when available. No arguments; source is fixed for this run. "
                        "Repeated calls return cached metadata for the same run."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "submit_leaf_verdict",
                "description": (
                    "Submit your final judgment for THIS leaf only (the criterion in the system prompt). "
                    "Call exactly once when you are ready to finish this leaf."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["met", "not_met", "undetermined"],
                            "description": "met = clearly satisfied; not_met = clear enough to judge and not satisfied; "
                            "undetermined = silent or insufficient information in the submission.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Brief quote or paraphrase for met/not_met; optional for undetermined.",
                        },
                    },
                    "required": ["verdict"],
                },
            },
        }
    )
    return tools


def _dispatch_per_leaf_tool(state: PerLeafAgentState, name: str, args: dict[str, Any]) -> Any:
    if name == "read_submission":
        text = state.submission_text
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

    if name == "search_submission":
        q = (args.get("query") or "").strip()
        if len(q) < 2:
            return {"ok": False, "error": "query must be at least 2 characters"}
        max_hits = min(int(args.get("max_hits") or 15), 50)
        text = state.submission_text
        low = text.casefold()
        qq = q.casefold()
        hits = []
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

    if name == "get_video_metadata":
        yt = state.youtube
        vpath = state.video_path
        if not yt and not vpath:
            return {
                "ok": False,
                "error": "No video source configured (neither --video nor --youtube-url).",
            }
        if yt and vpath:
            return {"ok": False, "error": "Internal error: both local video and YouTube configured."}

        cache_key = f"youtube:{yt.url}" if yt else f"file:{vpath.resolve() if vpath else ''}"
        cache = state.metadata_cache
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        if yt:
            result = probe_video_metadata_for_tool(
                local_path=None,
                youtube_url=yt.url,
                cookiefile=yt.cookiefile,
                cookiesfrombrowser=yt.cookiesfrombrowser,
                youtube_player_client=yt.youtube_player_client,
                youtube_player_fallback=yt.youtube_player_fallback,
            )
        else:
            result = probe_video_metadata_for_tool(local_path=vpath, youtube_url=None)

        if cache is not None and cache_key:
            cache[cache_key] = result
        return result

    if name == "submit_leaf_verdict":
        verdict = args.get("verdict")
        if verdict not in ("met", "not_met", "undetermined"):
            return {"ok": False, "error": "verdict must be met | not_met | undetermined"}
        ev = args.get("evidence")
        if ev is not None and not isinstance(ev, str):
            ev = None
        if ev == "":
            ev = None
        state.verdict_result = {"verdict": verdict, "evidence": ev}
        return {"ok": True, "recorded": True, "leaf_id": state.leaf.leaf_id}

    return {"ok": False, "error": f"Unknown tool: {name}"}


def _single_leaf_system_prompt(L: RubricLeafRef, *, has_video_metadata_tool: bool) -> str:
    basic = (L.basic_group or "").strip()
    basic_line = f'Basic rule group: "{basic}"\n' if basic else ""
    cw = L.category_weight
    lw = L.leaf_weight
    wparts = []
    if cw is not None:
        wparts.append(f"category weight: {cw}")
    if lw is not None:
        wparts.append(f"leaf weight: {lw}")
    wline = ("Weights: " + "; ".join(wparts) + "\n") if wparts else ""
    video_line = ""
    if has_video_metadata_tool:
        video_line = (
            "- You may call get_video_metadata to read duration and technical details of the student's "
            "video (local file or YouTube URL configured for this run; results are cached if you call again).\n"
        )
    return f"""You are a fair grader. Evaluate exactly ONE atomic criterion (one rubric leaf).

leaf_id (for your logs only): {L.leaf_id}
Category: {L.category!r}
{basic_line}{wline}Atomic criterion to judge:
---
{L.rule_text}
---

Instructions:
- Inspect the student submission via read_submission and search_submission. Do not assume you already have the full text in chat.
{video_line}- When you can decide, call submit_leaf_verdict once with verdict met, not_met, or undetermined.
- Use undetermined if the submission does not address this requirement or there is not enough information (e.g. no transcript and no usable metadata).
- For met or not_met, include a short evidence string (quote or paraphrase, or duration facts from metadata) when possible.
- You must call submit_leaf_verdict to complete this task."""


def run_single_leaf_agent_loop(
    state: PerLeafAgentState,
    *,
    model: str | None = None,
    max_turns: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Tool loop for one leaf. Returns (merged_row_dict, turn_transcript).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_name = model or os.environ.get("OPENAI_EVAL_AGENT_MODEL", "gpt-4o-mini")
    client = OpenAI()
    has_video = state.video_path is not None or state.youtube is not None
    tools = _per_leaf_tool_schemas(include_video_metadata=has_video)
    L = state.leaf

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _single_leaf_system_prompt(L, has_video_metadata_tool=has_video)},
        {
            "role": "user",
            "content": (
                "Evaluate this single criterion against the student submission using the tools. "
                "Finish by calling submit_leaf_verdict."
            ),
        },
    ]
    transcript: list[dict[str, Any]] = []

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        transcript.append(
            {
                "turn": turn,
                "assistant": msg.model_dump()
                if hasattr(msg, "model_dump")
                else {"content": msg.content, "tool_calls": getattr(msg, "tool_calls", None)},
            }
        )

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                raw = tc.function.arguments or "{}"
                args = json.loads(raw) if isinstance(raw, str) else {}
            except json.JSONDecodeError:
                args = {}
                result: Any = {"ok": False, "error": "Invalid JSON in tool arguments"}
            else:
                result = _dispatch_per_leaf_tool(state, fname, args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        if state.verdict_result is not None:
            vr = state.verdict_result
            return (
                _leaf_row(L, vr["verdict"], vr.get("evidence")),
                transcript,
            )

    # No submit_leaf_verdict
    return (
        _leaf_row(L, "undetermined", "Model did not call submit_leaf_verdict in time."),
        transcript,
    )


def evaluate_submission_per_leaf_agents(
    rubric: RKTRoot,
    submission_text: str,
    *,
    model: str | None = None,
    max_turns_per_leaf: int = 16,
    video_path: str | Path | None = None,
    youtube: YouTubeMediaConfig | None = None,
    on_leaf_done: Callable[[dict[str, Any], int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    For each leaf in ``rubric``, run ``run_single_leaf_agent_loop``.

    ``on_leaf_done`` is optional ``callback(row, index, total)`` for progress (e.g. print).

    Returns (merged_rows, transcripts_by_leaf) where transcripts_by_leaf is a list of
    dicts ``{{"leaf_id": int, "turns": [...]}}`` in leaf order.

    If ``video_path`` is set, ``get_video_metadata`` uses ffprobe on that file.
    If ``youtube`` is set (``YouTubeMediaConfig``), the same tool uses yt-dlp metadata only (no download).
    At most one of ``video_path`` and ``youtube`` should be set. The model does not choose the URL/path.

    Metadata is cached in-memory for the whole submission run so each leaf does not re-fetch YouTube.
    """
    if video_path and youtube:
        raise ValueError("Pass at most one of video_path and youtube")

    leaves = flatten_rubric_leaves(rubric)
    merged: list[dict[str, Any]] = []
    all_transcripts: list[dict[str, Any]] = []
    total = len(leaves)
    vpath = Path(video_path).resolve() if video_path else None
    metadata_cache: dict[str, Any] = {}

    for i, L in enumerate(leaves):
        state = PerLeafAgentState(
            submission_text=submission_text,
            leaf=L,
            video_path=vpath,
            youtube=youtube,
            metadata_cache=metadata_cache,
        )
        row, turns = run_single_leaf_agent_loop(state, model=model, max_turns=max_turns_per_leaf)
        merged.append(row)
        all_transcripts.append({"leaf_id": L.leaf_id, "turns": turns})
        if on_leaf_done:
            on_leaf_done(row, i, total)

    return merged, all_transcripts


def main() -> None:
    aide_dir = Path(__file__).resolve().parent
    default_tree = aide_dir / "rubrics" / "gsu-sumprod.json"
    default_response = aide_dir / "sample-responses" / "gsu-student-response.txt"

    parser = argparse.ArgumentParser(
        description=(
            "Per-leaf agentic grading: load a pre-built RKT JSON, then one tool-using agent per atomic leaf."
        )
    )
    parser.add_argument(
        "rkt_json",
        nargs="?",
        default=str(default_tree),
        help="Path to RKT skill tree JSON (from ratas-rubric.py; type: root)",
    )
    parser.add_argument(
        "submission",
        nargs="?",
        default=str(default_response),
        help="Path to student submission text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write merged leaf assessments as JSON",
    )
    parser.add_argument(
        "--transcript",
        metavar="PATH",
        help="Write per-leaf tool transcripts as JSON",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_EVAL_AGENT_MODEL", "gpt-4o-mini"),
        help="Chat model (default: env OPENAI_EVAL_AGENT_MODEL or gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-turns-per-leaf",
        type=int,
        default=16,
        help="Max assistant steps per leaf (each step may include multiple tool calls)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-leaf progress on stderr",
    )
    parser.add_argument(
        "--video",
        metavar="PATH",
        help=(
            "Local video file (mp4, mov, …). Enables get_video_metadata via ffprobe. "
            "Do not combine with --youtube-url."
        ),
    )
    parser.add_argument(
        "--youtube-url",
        metavar="URL",
        help=(
            "Student's YouTube video link. Enables get_video_metadata via yt-dlp without downloading "
            "the video (duration, title, id, resolution when available). Requires: pip install yt-dlp. "
            "Do not combine with --video."
        ),
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        metavar="FILE",
        help="With --youtube-url: Netscape cookies.txt for yt-dlp (unlisted / signed-in only)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="With --youtube-url: load cookies from browser (e.g. chrome, firefox, safari)",
    )
    parser.add_argument(
        "--browser-profile",
        type=str,
        default=None,
        metavar="NAME",
        help="Browser profile name with --cookies-from-browser",
    )
    parser.add_argument(
        "--youtube-player-client",
        type=str,
        default=None,
        metavar="NAME",
        help="Force one yt-dlp YouTube client (e.g. android); disables multi-client fallback",
    )
    parser.add_argument(
        "--no-youtube-player-fallback",
        action="store_true",
        help="With --youtube-url: only yt-dlp's default YouTube client (no automatic retries)",
    )
    args = parser.parse_args()

    if args.video and args.youtube_url:
        print("Use only one of --video and --youtube-url", file=sys.stderr)
        sys.exit(1)
    if args.cookies and args.cookies_from_browser:
        print("Use only one of --cookies and --cookies-from-browser", file=sys.stderr)
        sys.exit(1)
    if (args.cookies or args.cookies_from_browser) and not args.youtube_url:
        print("--cookies / --cookies-from-browser require --youtube-url", file=sys.stderr)
        sys.exit(1)

    tree_path = Path(args.rkt_json)
    sub_path = Path(args.submission)
    if not tree_path.is_file():
        print(f"RKT file not found: {tree_path}", file=sys.stderr)
        sys.exit(1)
    if not sub_path.is_file():
        print(f"Submission not found: {sub_path}", file=sys.stderr)
        sys.exit(1)

    try:
        rubric = load_rkt_json(tree_path)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    submission_text = sub_path.read_text(encoding="utf-8")
    n_leaves = len(flatten_rubric_leaves(rubric))

    video_arg: Path | None = None
    youtube_cfg: YouTubeMediaConfig | None = None

    if args.video:
        video_arg = Path(args.video)
        if not video_arg.is_file():
            print(f"Video file not found: {video_arg}", file=sys.stderr)
            sys.exit(1)
        if not ffprobe_available() and not args.quiet:
            print(
                "Warning: ffprobe not on PATH; get_video_metadata will fail until ffmpeg is installed.",
                file=sys.stderr,
            )

    if args.youtube_url:
        url = args.youtube_url.strip()
        if not looks_like_youtube_url(url):
            print(
                f"Not a recognized YouTube URL: {url!r} (expected https://youtube.com/... or youtu.be/...)",
                file=sys.stderr,
            )
            sys.exit(1)
        if not yt_dlp_available() and not args.quiet:
            print(
                "Warning: yt-dlp not installed; get_video_metadata will fail. pip install yt-dlp",
                file=sys.stderr,
            )
        cookiesfrombrowser: tuple[str, ...] | None = None
        if args.cookies_from_browser:
            b = args.cookies_from_browser.strip()
            cookiesfrombrowser = (b, args.browser_profile) if args.browser_profile else (b,)
        youtube_cfg = YouTubeMediaConfig(
            url=url,
            cookiefile=args.cookies,
            cookiesfrombrowser=cookiesfrombrowser,
            youtube_player_client=args.youtube_player_client,
            youtube_player_fallback=not args.no_youtube_player_fallback,
        )

    def progress(row: dict[str, Any], index: int, total: int) -> None:
        if args.quiet:
            return
        vid = row["verdict"]
        print(f"[leaf {index + 1}/{total}] L{row['leaf_id']} -> {vid}", file=sys.stderr)

    merged, transcripts = evaluate_submission_per_leaf_agents(
        rubric,
        submission_text,
        model=args.model,
        max_turns_per_leaf=args.max_turns_per_leaf,
        video_path=video_arg,
        youtube=youtube_cfg,
        on_leaf_done=progress,
    )

    if args.transcript:
        p = Path(args.transcript)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.quiet:
            print(f"Wrote transcript {p}", file=sys.stderr)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.quiet:
            print(f"Wrote {out}", file=sys.stderr)

    if not args.quiet:
        print(f"Evaluated {n_leaves} leaves.", file=sys.stderr)

    for row in merged:
        vid = row["verdict"]
        mark = {"met": "✓", "not_met": "✗", "undetermined": "?"}[vid]
        crit = (row["criterion"] or "")[:72]
        if len(row.get("criterion") or "") > 72:
            crit += "..."
        cat = (row["category"] or "")[:40]
        print(f"{mark} [{vid:14}] L{row['leaf_id']:2} | {cat} | {crit}")
        if row.get("evidence"):
            print(f"         evidence: {str(row['evidence'])[:120]}")


if __name__ == "__main__":
    main()
