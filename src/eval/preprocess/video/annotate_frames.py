#!/usr/bin/env python3
"""
Send extracted keyframe images (e.g. from ``video_frame_extract.py``) to an OpenAI vision
model and produce detailed, structured annotations — including verbatim on-screen text and
spreadsheet/table content where legible.

Images are sent as ``data:image/...;base64,...`` URIs (``image_url.url``), with
``detail: "high"`` so small UI text is easier to read.

Requires: pip install openai pydantic | optional: python-dotenv, pillow

Environment: ``OPENAI_API_KEY`` (loaded from ``aide/.env`` when present); optional ``OPENAI_VIDEO_FRAME_MODEL`` (default ``gpt-4o``). Optional ``OPENAI_VIDEO_FRAME_MAX_TOKENS`` caps completion size (default ``16384``; set ``0`` to omit).

Optional: ``pip install pillow`` and ``--max-long-edge`` to shrink very large JPEGs before upload.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List
from openai import OpenAI
from src.util.openai import OpenAIClient
from pydantic import BaseModel, Field, ValidationError

from .extract_frames import ExtractedFrame

MIME_BY_SUFFIX: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

DEFAULT_MODEL = os.environ.get("OPENAI_VIDEO_FRAME_MODEL", "gpt-4o")

SYSTEM_PROMPT = """You are an expert technical analyst documenting still frames from student \
submission videos (tutorials, screen recordings, webcam segments).

Your job is exhaustive, literal visual documentation — not interpretation of whether answers \
are correct.

Rules:
1. Transcribe every readable string (window titles, menus, spreadsheet cells, browser chrome, \
URLs, button labels, watermarks). Preserve spelling and casing; use quotation marks for UI text.
2. For spreadsheets, tables, or grids: state the application if identifiable (Excel, Google \
Sheets, etc.), then describe structure (headers, row/column indices if visible), and reproduce \
cell values you can read in a clear grid or row-wise form. Merge ranges if shown.
3. For code editors or terminals: reproduce readable code or command lines in fenced code blocks \
only when you are confident; otherwise paraphrase and mark uncertainty.
4. For math: use plain text or Unicode; describe notation if ambiguous.
5. Describe layout spatially when it helps (e.g. “left pane: file tree; right: editor”).
6. If text is blurry or cropped, say exactly what is uncertain. Never invent values you cannot see.
7. Respond with a single JSON object matching the schema described in the user message — no markdown \
outside the JSON."""


class FrameVisionAnnotation(BaseModel):
    """Structured annotation for one frame (parsed from model JSON)."""

    scene_overview: str = Field(
        ...,
        description="2–5 sentences: overall scene, purpose, dominant elements.",
    )
    shot_type: str = Field(
        ...,
        description="e.g. webcam_only, screen_capture, split_webcam_and_screen, whiteboard, document_camera, mixed, other",
    )
    persons_and_actions: str = Field(
        default="",
        description="People, hands, pointers, gaze — or empty if none.",
    )
    screen_ui_and_applications: str = Field(
        default="",
        description="OS, app names, window titles, menus, toolbars, dialogs, browser tabs/URLs if legible.",
    )
    verbatim_visible_text: str = Field(
        ...,
        description="All readable text in sensible reading order; quote UI strings; group by region if helpful.",
    )
    tabular_and_sheet_content: str = Field(
        default="",
        description="Spreadsheets/tables: headers, row/column labels, and cell values read from the image.",
    )
    math_code_formulas_and_diagrams: str = Field(
        default="",
        description="Equations, code, flowcharts, plots — transcribe or describe precisely.",
    )
    spatial_layout_notes: str = Field(
        default="",
        description="Where major elements sit (e.g. top banner, bottom taskbar, split panes).",
    )
    illegible_or_uncertain: str = Field(
        default="",
        description="Cropped, blurred, or occluded content; guesses you did NOT make.",
    )


USER_JSON_SCHEMA_HINT = """Return one JSON object with exactly these string fields (all required; use empty string \"\" if nothing applies except scene_overview and shot_type and verbatim_visible_text):
- scene_overview
- shot_type
- persons_and_actions
- screen_ui_and_applications
- verbatim_visible_text
- tabular_and_sheet_content
- math_code_formulas_and_diagrams
- spatial_layout_notes
- illegible_or_uncertain"""


def _mime_for_path(path: Path) -> str:
    return MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")


def _maybe_downscale_image_bytes(
    raw: bytes, src_mime: str, max_long_edge: int
) -> tuple[bytes, str]:
    """Return (bytes, mime). Re-encodes as JPEG when Pillow resizes or recompresses."""
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return raw, src_mime

    im = Image.open(BytesIO(raw))
    im = im.convert("RGB")
    w, h = im.size
    m = max(w, h)
    out_mime = "image/jpeg"
    if m <= max_long_edge:
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), out_mime
    scale = max_long_edge / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:
        resample = Image.LANCZOS
    im = im.resize((nw, nh), resample)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), out_mime


def image_file_to_data_uri(path: Path, *, max_long_edge: int | None = None) -> str:
    """Read an image file and return a data URI suitable for OpenAI ``image_url.url``."""
    raw = path.read_bytes()
    mime = _mime_for_path(path)
    if max_long_edge and max_long_edge > 0:
        try:
            raw, mime = _maybe_downscale_image_bytes(raw, mime, max_long_edge)
        except Exception:
            raw = path.read_bytes()
            mime = _mime_for_path(path)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _vision_max_output_tokens() -> int | None:
    raw = (os.environ.get("OPENAI_VIDEO_FRAME_MAX_TOKENS") or "16384").strip()
    if not raw or raw == "0":
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return 16384


def _degraded_frame_annotation(reason: str, *, image_path: Path) -> FrameVisionAnnotation:
    """Valid object when the model returns truncated or non-JSON output."""
    safe = reason.replace("\n", " ").strip()
    if len(safe) > 1200:
        safe = safe[:1200] + "…"
    return FrameVisionAnnotation(
        scene_overview=(
            f"Fallback annotation: vision output was invalid or truncated for `{image_path.name}`."
        ),
        shot_type="other",
        verbatim_visible_text="",
        illegible_or_uncertain=safe,
    )


def _vision_json_completion(
    client: OpenAI,
    *,
    model: str,
    data_uri: str,
    user_text: str,
) -> tuple[str, str | None]:
    """Return (message_content, finish_reason_or_none)."""
    max_out = _vision_max_output_tokens()
    kwargs: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
    }
    if max_out is not None:
        kwargs["max_tokens"] = max_out
    try:
        resp = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("max_tokens", None)
        if max_out is not None:
            kwargs["max_completion_tokens"] = max_out
        resp = client.chat.completions.create(**kwargs)
    choice0 = resp.choices[0]
    content = choice0.message.content
    if not content:
        raise RuntimeError("Empty model response")
    fr = getattr(choice0, "finish_reason", None)
    return content, fr


def annotate_frame(
    image_path: Path,
    time_seconds: float | None,
    model: str,
    max_long_edge: int | None = None,
) -> FrameVisionAnnotation:
    """Call the vision model once for ``image_path``; return parsed structured annotation."""

    client = OpenAIClient().client

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    data_uri = image_file_to_data_uri(image_path, max_long_edge=max_long_edge)
    t_note = (
        f"This frame occurs at approximately **{time_seconds:.3f}s** in the video timeline."
        if time_seconds is not None
        else "Video timestamp for this frame is unknown."
    )
    user_text = f"""{t_note}

File name (for disambiguation only — do not invent content from it): `{image_path.name}`

{USER_JSON_SCHEMA_HINT}

Be as thorough as possible on verbatim text and spreadsheet/table cells. If the frame \
contains a huge spreadsheet, prioritize headers, visible row/column labels, and a \
representative sample of cells — **complete, valid JSON is more important than listing \
every cell**, so keep each string field bounded and never truncate inside a JSON string."""

    compact_suffix = (
        "\n\nOutput a **single valid JSON object** with the same field names as above. "
        "If the scene is text-heavy, cap each string field at roughly 3000 characters "
        "(summarize or sample); never leave strings unclosed — finish the JSON object."
    )

    content, finish_reason = _vision_json_completion(
        client, model=model, data_uri=data_uri, user_text=user_text
    )
    if finish_reason == "length":
        print(
            f"[warn] {image_path.name}: completion stopped at max length; retrying compact…",
            flush=True,
        )
        content, finish_reason = _vision_json_completion(
            client, model=model, data_uri=data_uri, user_text=user_text + compact_suffix
        )

    try:
        return FrameVisionAnnotation.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError, ValueError) as e1:
        print(
            f"[warn] {image_path.name}: JSON parse failed ({e1!s}); retrying compact…",
            flush=True,
        )
        content2, _fr2 = _vision_json_completion(
            client, model=model, data_uri=data_uri, user_text=user_text + compact_suffix
        )
        try:
            return FrameVisionAnnotation.model_validate_json(content2)
        except (ValidationError, json.JSONDecodeError, ValueError) as e2:
            print(
                f"[warn] {image_path.name}: compact retry failed ({e2!s}); using degraded annotation.",
                flush=True,
            )
            return _degraded_frame_annotation(f"{e1!s} | retry: {e2!s}", image_path=image_path)


def annotate_frames(
    frames: list[ExtractedFrame],
    summary_path: Path,
    ) -> Path:
    """Summarize each frame and write a JSON array (same shape as ``video_frame_summarize.py``)."""
    results: list[dict[str, Any]] = []
    for frame in frames:
        annotation = annotate_frame(frame.path, frame.time_seconds, model=DEFAULT_MODEL, max_long_edge=1280)
        results.append({
            "frame_path": str(frame.path.resolve()),
            "time_seconds": frame.time_seconds,
            # "model": DEFAULT_MODEL,
            "annotation": annotation.model_dump(),
        })
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    pass
