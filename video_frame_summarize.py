#!/usr/bin/env python3
"""
Send extracted keyframe images (e.g. from ``video_frame_extract.py``) to an OpenAI vision
model and produce detailed, structured annotations — including verbatim on-screen text and
spreadsheet/table content where legible.

Images are sent as ``data:image/...;base64,...`` URIs (``image_url.url``), with
``detail: "high"`` so small UI text is easier to read.

Requires: pip install openai python-dotenv pydantic
Environment: ``OPENAI_API_KEY``; optional ``OPENAI_VIDEO_FRAME_MODEL`` (default ``gpt-4o``).

Optional: ``pip install pillow`` and ``--max-long-edge`` to shrink very large JPEGs before upload.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

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


def summarize_frame(
    client: OpenAI,
    *,
    image_path: Path,
    time_seconds: float | None,
    model: str,
    max_long_edge: int | None = None,
) -> FrameVisionAnnotation:
    """Call the vision model once for ``image_path``; return parsed structured annotation."""
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

Be as thorough as possible on verbatim text and spreadsheet/table cells."""

    # json_schema + strict requires additionalProperties: false on all objects in schema;
    # Pydantic v2 model_json_schema may need refinement — use json_object mode for robustness.
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
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
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty model response")
    return FrameVisionAnnotation.model_validate_json(content)


@dataclass(frozen=True)
class FrameJob:
    path: Path
    time_seconds: float | None


def iter_frames_from_summary_json(path: Path) -> Iterator[tuple[str, FrameJob]]:
    """
    Load JSON like ``video_frame_extract.py`` stdout: a list of objects with ``label`` and ``frames``.
    Yields (label, FrameJob) in list order.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at top level: {path}")
    for entry in data:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "submission")
        frames = entry.get("frames") or []
        if not isinstance(frames, list):
            continue
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            p = fr.get("path")
            if not p:
                continue
            ts = fr.get("time_seconds")
            tsf: float | None
            try:
                tsf = float(ts) if ts is not None else None
            except (TypeError, ValueError):
                tsf = None
            yield label, FrameJob(path=Path(p), time_seconds=tsf)


def iter_frames_from_dir(frames_dir: Path, *, patterns: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.webp")) -> Iterator[FrameJob]:
    """Sorted image paths under ``frames_dir`` (shallow glob)."""
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(frames_dir.glob(pat))
    paths = sorted({p.resolve() for p in paths if p.is_file()}, key=lambda p: p.name)
    for p in paths:
        yield FrameJob(path=p, time_seconds=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate extracted video frames with an OpenAI vision model (base64 data URIs)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--summary-json",
        type=Path,
        metavar="PATH",
        help="JSON array from video_frame_extract.py (uses path + time_seconds per frame)",
    )
    src.add_argument(
        "--frames-dir",
        type=Path,
        metavar="DIR",
        help="Directory of frame images (*.jpg, *.png, …)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video_frame_summaries.json"),
        help="Write results JSON here (default: ./video_frame_summaries.json)",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Vision-capable model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Pause between API calls (rate limiting)",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=None,
        metavar="PX",
        help="If Pillow is installed, downscale so max(width,height) <= this before base64",
    )
    parser.add_argument(
        "--label-filter",
        type=str,
        default=None,
        metavar="SUBSTR",
        help="With --summary-json: only process entries whose label contains this substring",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List frames that would be processed; do not call the API",
    )
    args = parser.parse_args()

    jobs: list[tuple[str, FrameJob]] = []
    if args.summary_json:
        for label, job in iter_frames_from_summary_json(args.summary_json):
            if args.label_filter and args.label_filter not in label:
                continue
            jobs.append((label, job))
    else:
        d = args.frames_dir
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        for job in iter_frames_from_dir(d):
            jobs.append(("frames_dir", job))

    if not jobs:
        raise SystemExit("No frames to process.")

    if args.dry_run:
        print(json.dumps([{"label": L, "path": str(j.path), "time_seconds": j.time_seconds} for L, j in jobs], indent=2))
        return

    client = OpenAI()
    results: list[dict[str, Any]] = []
    for i, (label, job) in enumerate(jobs):
        print(f"[{i + 1}/{len(jobs)}] {label} :: {job.path.name}", flush=True)
        ann = summarize_frame(
            client,
            image_path=job.path,
            time_seconds=job.time_seconds,
            model=args.model,
            max_long_edge=args.max_long_edge,
        )
        results.append(
            {
                "label": label,
                "frame_path": str(job.path.resolve()),
                "time_seconds": job.time_seconds,
                "model": args.model,
                "annotation": ann.model_dump(),
            }
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} annotations to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
