from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

_PTS_TIME_RE = re.compile(r"pts_time\s*:\s*([0-9.]+)")

@dataclass(frozen=True)
class ExtractedFrame:
    """One keyframe saved to disk."""
    path: Path
    time_seconds: float


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _parse_showinfo_pts_times(stderr_text: str) -> list[float]:
    times: list[float] = []
    for line in stderr_text.splitlines():
        m = _PTS_TIME_RE.search(line)
        if m:
            try:
                times.append(float(m.group(1)))
            except ValueError:
                continue
    return times


def extact_important_frames(
    video_path: Path, 
    frames_dir: Path,
    *,
    scene_threshold: float = 0.32,
    include_t0: bool = True,
    min_gap_seconds: float = 0.35,
    scale_max_width: int | None = 1280,
) -> list[ExtractedFrame]:
    """
    Save JPEGs for the first frame and each frame whose ``scene`` score vs. the previous
    output frame exceeds ``scene_threshold`` (ffmpeg convention: roughly 0.25–0.45 for cuts).

    Drops saved frames that are within ``min_gap_seconds`` of a previously kept frame
    (keeps the earlier one) to limit near-duplicate bursts from gradual transitions.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    pattern = str(frames_dir / f"{stem}_scene_%06d.jpg")

    # select: first frame OR scene cut; showinfo on stderr gives pts_time per emitted frame
    th = scene_threshold
    if include_t0:
        select_expr = f"gte(scene\\,{th})+eq(n\\,0)"
    else:
        select_expr = f"gte(scene\\,{th})"

    vf_parts = [f"select='{select_expr}'", "showinfo"]
    if scale_max_width and scale_max_width > 0:
        vf_parts.append(f"scale={scale_max_width}:-1:flags=lanczos")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-y",
        "-i",
        str(video_path.resolve()),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        pattern,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    err = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode != 0:
        tail = err.strip()[-800:]
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {tail}")

    written = sorted(frames_dir.glob(f"{stem}_scene_*.jpg"))
    pts_times = _parse_showinfo_pts_times(err)
    if len(pts_times) < len(written):
        # Rare stderr timing; keep frame order, approximate time by index
        pts_times = pts_times + [float("nan")] * (len(written) - len(pts_times))
    elif len(pts_times) > len(written):
        pts_times = pts_times[: len(written)]

    raw = [
        ExtractedFrame(
            path=p,
            time_seconds=float(i) if (t != t) else t,  # NaN → index-order placeholder
        )
        for i, (p, t) in enumerate(zip(written, pts_times))
    ]
    raw.sort(key=lambda x: x.time_seconds)

    if min_gap_seconds <= 0:
        return raw

    kept: list[ExtractedFrame] = []
    for frame in raw:
        if not kept or frame.time_seconds - kept[-1].time_seconds >= min_gap_seconds:
            kept.append(frame)
    return kept

# TODO: Implement this and make a tool for it. Also requires transcription with timestamps 
# to be done first.
def extract_frame_at_time(video_path: Path, time: float, frames_dir: Path) -> None:
    pass
