from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

# Web clients expose higher-resolution streams; android often caps around 720p.
_YOUTUBE_PLAYER_CLIENT_FALLBACKS: tuple[str | None, ...] = (
    None,
    "android",
    "web",
    "web_embedded",
)


def _video_format_string(max_height: int | None) -> str:
    """Prefer best separate video+audio, merged to mp4; cap height when set."""
    if max_height is None:
        return "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/best"
    h = int(max_height)
    return (
        f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[height<={h}]+ba/"
        f"b[height<={h}][ext=mp4]/"
        f"b[height<={h}]/best[height<={h}]/best"
    )


def _ydl_video_opts(
    out_path: Path,
    *,
    max_height: int | None,
    youtube_player_client: str | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "format": _video_format_string(max_height),
        "merge_output_format": "mp4",
        "outtmpl": str(out_path) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    node = shutil.which("node")
    if node:
        opts["js_runtimes"] = {"node": {"path": node}}
    if youtube_player_client is not None:
        opts["extractor_args"] = {"youtube": {"player_client": [youtube_player_client]}}
    return opts


def _resolve_downloaded_path(info: dict[str, Any], out_path: Path) -> Path:
    rp = info.get("requested_downloads")
    if isinstance(rp, list) and rp:
        fp = rp[-1].get("filepath")
        if fp and Path(fp).is_file():
            return Path(fp)
    for ext in ("mp4", "webm", "mkv"):
        candidate = Path(f"{out_path}.{ext}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No video file found for {out_path} after yt-dlp download")


def _download_youtube_once(
    url: str,
    out_path: Path,
    *,
    max_height: int | None,
    youtube_player_client: str | None,
) -> Path:
    ydl_opts = _ydl_video_opts(
        out_path,
        max_height=max_height,
        youtube_player_client=youtube_player_client,
    )
    label = youtube_player_client or "default"
    print(f"yt-dlp: trying player_client={label!r}", file=sys.stderr, flush=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp returned unexpected info type")
    path = _resolve_downloaded_path(info, out_path)
    w, h = info.get("width"), info.get("height")
    if w and h:
        print(f"yt-dlp: saved {path.name} ({w}x{h})", file=sys.stderr, flush=True)
    return path


def download_youtube_video(
    url: str,
    out_dir: Path,
    out_filename: str = "video",
    *,
    max_height: int | None = 1080,
    youtube_player_fallback: bool = True,
) -> Path:
    """
    Download a YouTube URL as mp4.

    Uses best available video+audio (up to ``max_height`` pixels tall), then merges.
    Tries several YouTube player clients if the first attempt fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_filename

    clients: tuple[str | None, ...]
    if youtube_player_fallback:
        clients = _YOUTUBE_PLAYER_CLIENT_FALLBACKS
    else:
        clients = (None,)

    last_err: BaseException | None = None
    for pc in clients:
        try:
            return _download_youtube_once(
                url,
                out_path,
                max_height=max_height,
                youtube_player_client=pc,
            )
        except DownloadError as e:
            last_err = e
            continue

    if last_err is not None:
        print(f"*** ERROR *** Error downloading video {url}: {last_err}")
        raise last_err
    raise RuntimeError(f"Failed to download {url}")


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("student-responses/gsu-student-sumprod-video-list-short.csv")
    out_dir = Path("download-test")
    for index, row in df.iterrows():
        url = row["link"]
        alias = row["email"].split("@")[0]
        download_youtube_video(url, out_dir, f"{alias}")
