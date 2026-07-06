from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def _build_vf(*parts: str) -> str:
    return ",".join(p for p in parts if p)


def _dedupe_by_time(
    frames: list[ExtractedFrame],
    min_gap_seconds: float,
) -> list[ExtractedFrame]:
    sorted_frames = sorted(frames, key=lambda x: x.time_seconds)
    if min_gap_seconds <= 0:
        return sorted_frames

    kept: list[ExtractedFrame] = []
    for frame in sorted_frames:
        if not kept or frame.time_seconds - kept[-1].time_seconds >= min_gap_seconds:
            kept.append(frame)
    return kept


def _ffmpeg_extract(
    video_path: Path,
    vf: str,
    output_pattern: str,
) -> list[ExtractedFrame]:
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
        output_pattern,
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

    pattern_path = Path(output_pattern)
    written = sorted(pattern_path.parent.glob(pattern_path.name.replace("%06d", "*")))
    pts_times = _parse_showinfo_pts_times(err)
    if len(pts_times) < len(written):
        pts_times = pts_times + [float("nan")] * (len(written) - len(pts_times))
    elif len(pts_times) > len(written):
        pts_times = pts_times[: len(written)]

    return [
        ExtractedFrame(
            path=p,
            time_seconds=float(i) if (t != t) else t,
        )
        for i, (p, t) in enumerate(zip(written, pts_times))
    ]


def _extract_scene_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    scene_threshold: float,
    include_t0: bool,
    scale_max_width: int | None,
) -> list[ExtractedFrame]:
    stem = video_path.stem
    pattern = str(frames_dir / f"{stem}_scene_%06d.jpg")

    if include_t0:
        select_expr = f"gte(scene\\,{scene_threshold})+eq(n\\,0)"
    else:
        select_expr = f"gte(scene\\,{scene_threshold})"

    scale = (
        f"scale={scale_max_width}:-1:flags=lanczos"
        if scale_max_width and scale_max_width > 0
        else ""
    )
    vf = _build_vf(f"select='{select_expr}'", "showinfo", scale)
    return _ffmpeg_extract(video_path, vf, pattern)


def _extract_interval_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    sample_interval_seconds: float,
    scale_max_width: int | None,
) -> list[ExtractedFrame]:
    stem = video_path.stem
    pattern = str(frames_dir / f"{stem}_interval_%06d.jpg")

    scale = (
        f"scale={scale_max_width}:-1:flags=lanczos"
        if scale_max_width and scale_max_width > 0
        else ""
    )
    vf = _build_vf(f"fps={1.0 / sample_interval_seconds}", "showinfo", scale)
    return _ffmpeg_extract(video_path, vf, pattern)


def extact_important_frames(
    video_path: Path, 
    frames_dir: Path,
    *,
    scene_threshold: float = 0.32,
    include_t0: bool = True,
    min_gap_seconds: float = 0.35,
    scale_max_width: int | None = 1280,
    sample_interval_seconds: float | None = 30.0,
) -> list[ExtractedFrame]:
    """
    Save JPEGs for scene changes and, optionally, at fixed time intervals.

    Scene frames: the first frame and each frame whose ``scene`` score vs. the previous
    output frame exceeds ``scene_threshold`` (ffmpeg convention: roughly 0.25–0.45 for cuts).

    Interval frames: one frame every ``sample_interval_seconds`` (e.g. 30 → t=0, 30, 60…).
    Set ``sample_interval_seconds`` to ``None`` to skip interval sampling.

    Drops saved frames that are within ``min_gap_seconds`` of a previously kept frame
    (keeps the earlier one) to limit near-duplicate bursts from gradual transitions.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if sample_interval_seconds is not None and sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = _extract_scene_frames(
        video_path,
        frames_dir,
        scene_threshold=scene_threshold,
        include_t0=include_t0,
        scale_max_width=scale_max_width,
    )

    if sample_interval_seconds is not None:
        frames.extend(
            _extract_interval_frames(
                video_path,
                frames_dir,
                sample_interval_seconds=sample_interval_seconds,
                scale_max_width=scale_max_width,
            )
        )

    return _dedupe_by_time(frames, min_gap_seconds)

# TODO: Implement this and make a tool for it. Also requires transcription with timestamps 
# to be done first.
def extract_frame_at_time(video_path: Path, time: float, frames_dir: Path) -> None:
    pass
