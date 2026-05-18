from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_mp3_from_video(
    video_path: Path,
    mp3_path: Path | None = None,
    *,
    bitrate_kbps: int = 192,
) -> Path:
    """
    Extract audio from a video file and save it as MP3.

    Requires ffmpeg on PATH.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH (install ffmpeg).")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if mp3_path is None:
        mp3_path = video_path.with_suffix(".mp3")
    else:
        mp3_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        f"{bitrate_kbps}k",
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {tail}")
    if not mp3_path.is_file():
        raise FileNotFoundError(f"ffmpeg did not create {mp3_path}")
    return mp3_path
