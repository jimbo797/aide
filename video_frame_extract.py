#!/usr/bin/env python3
"""
Download YouTube submissions and extract frames where the picture changes sharply
(camera vs screen share, window switches, etc.).

Uses yt-dlp (same auth / player-client fallbacks as ``audio-transcription/youtube_transcribe.py``)
and ffmpeg's ``select`` filter with the internal ``scene`` metric (difference vs. the previous frame).

Requires: pip install yt-dlp | System: ffmpeg on PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from video_metadata import looks_like_youtube_url

_YOUTUBE_PLAYER_CLIENT_FALLBACKS: tuple[str | None, ...] = (
    None,
    "android",
    "web_embedded",
    "web",
)

_PTS_TIME_RE = re.compile(r"pts_time\s*:\s*([0-9.]+)")


@dataclass(frozen=True)
class ExtractedFrame:
    """One keyframe saved to disk."""

    path: Path
    time_seconds: float


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _ydl_video_opts(
    out_dir: Path,
    *,
    cookiefile: Path | None,
    cookiesfrombrowser: tuple[str, ...] | None,
    youtube_player_client: str | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        # Prefer mp4 for downstream ffmpeg; fall back to best merged format.
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    if cookiefile is not None:
        opts["cookiefile"] = str(cookiefile)
    if cookiesfrombrowser is not None:
        opts["cookiesfrombrowser"] = cookiesfrombrowser
    if youtube_player_client is not None:
        opts["extractor_args"] = {"youtube": {"player_client": [youtube_player_client]}}
    return opts


def _download_youtube_once(
    url: str,
    out_dir: Path,
    *,
    cookiefile: Path | None,
    cookiesfrombrowser: tuple[str, ...] | None,
    youtube_player_client: str | None,
) -> Path:
    import yt_dlp

    ydl_opts = _ydl_video_opts(
        out_dir,
        cookiefile=cookiefile,
        cookiesfrombrowser=cookiesfrombrowser,
        youtube_player_client=youtube_player_client,
    )
    label = youtube_player_client or "default"
    print(f"yt-dlp: trying YouTube player_client={label!r}", flush=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not isinstance(info, dict):
            raise RuntimeError("yt-dlp returned unexpected info type")
        vid = info.get("id")
        if not vid:
            raise RuntimeError("yt-dlp did not return a video id")
        # Requested_filename after merge is most reliable
        rp = info.get("requested_downloads")
        if isinstance(rp, list) and rp:
            fp = rp[0].get("filepath")
            if fp and Path(fp).is_file():
                return Path(fp)
        # Fallback: common extensions
        for ext in ("mp4", "webm", "mkv"):
            p = out_dir / f"{vid}.{ext}"
            if p.is_file():
                return p
    raise FileNotFoundError(f"No video file found in {out_dir} after yt-dlp download")


def download_youtube_video(
    url: str,
    out_dir: Path,
    *,
    cookiefile: Path | None = None,
    cookiesfrombrowser: tuple[str, ...] | None = None,
    youtube_player_client: str | None = None,
    youtube_player_fallback: bool = True,
) -> Path:
    """
    Download one YouTube URL to ``out_dir`` and return the path to the merged video file.
    """
    try:
        import yt_dlp  # noqa: F401
        from yt_dlp.utils import DownloadError as YtdlDownloadError
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed. pip install yt-dlp") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    if youtube_player_client is not None:
        clients: tuple[str | None, ...] = (youtube_player_client,)
    elif youtube_player_fallback:
        clients = _YOUTUBE_PLAYER_CLIENT_FALLBACKS
    else:
        clients = (None,)

    last_err: BaseException | None = None
    for pc in clients:
        try:
            return _download_youtube_once(
                url,
                out_dir,
                cookiefile=cookiefile,
                cookiesfrombrowser=cookiesfrombrowser,
                youtube_player_client=pc,
            )
        except YtdlDownloadError as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


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


def extract_scene_change_frames(
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
        vf_parts.append(f"scale={scale_max_width}:-1")
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


def iter_youtube_rows_csv(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (email_or_label, url) from a CSV with columns ``link`` or ``url`` (+ optional ``email``)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        link_key = fields.get("link") or fields.get("url")
        if not link_key:
            raise ValueError(f"CSV must have a 'link' or 'url' column: {path}")
        email_key = fields.get("email")
        for row in reader:
            url = (row.get(link_key) or "").strip()
            if not url:
                continue
            label = (row.get(email_key) or "").strip() if email_key else ""
            if not label:
                label = url
            yield label, url


def preprocess_youtube_to_frames(
    url: str,
    work_dir: Path,
    *,
    scene_threshold: float = 0.32,
    min_gap_seconds: float = 0.35,
    cookiefile: Path | None = None,
    cookiesfrombrowser: tuple[str, ...] | None = None,
    youtube_player_client: str | None = None,
    youtube_player_fallback: bool = True,
    keep_video: bool = False,
) -> tuple[Path | None, list[ExtractedFrame]]:
    """
    Download ``url`` under ``work_dir``, extract scene-difference frames into
    ``work_dir / "frames"``. Returns ``(video_path_or_none, frames)`` — the video path is
    kept only when ``keep_video`` is True; otherwise the file is removed after extraction.
    """
    if not looks_like_youtube_url(url):
        raise ValueError(f"Not a YouTube URL: {url!r}")

    work_dir.mkdir(parents=True, exist_ok=True)
    video_dir = work_dir / "video"
    frames_dir = work_dir / "frames"
    video_path = download_youtube_video(
        url,
        video_dir,
        cookiefile=cookiefile,
        cookiesfrombrowser=cookiesfrombrowser,
        youtube_player_client=youtube_player_client,
        youtube_player_fallback=youtube_player_fallback,
    )
    frames = extract_scene_change_frames(
        video_path,
        frames_dir,
        scene_threshold=scene_threshold,
        min_gap_seconds=min_gap_seconds,
    )
    if keep_video:
        return video_path, frames
    try:
        video_path.unlink(missing_ok=True)
    except OSError:
        pass
    return None, frames


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube video(s) and extract frames at sharp visual changes."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="Single YouTube URL")
    src.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        help="CSV with link (or url) column; optional email column for labels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("video_preprocess_out"),
        help="Output directory (default: ./video_preprocess_out)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.32,
        help="ffmpeg scene filter threshold (higher = fewer, stronger cuts only). Default: 0.32",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.35,
        help="Minimum seconds between kept frames (debounce). Default: 0.35",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video files under out/<label>/video/",
    )
    parser.add_argument("--cookies", type=Path, default=None, metavar="FILE")
    parser.add_argument("--cookies-from-browser", type=str, default=None, metavar="BROWSER")
    parser.add_argument("--browser-profile", type=str, default=None)
    parser.add_argument("--youtube-player-client", type=str, default=None)
    parser.add_argument(
        "--no-youtube-player-fallback",
        action="store_true",
    )
    args = parser.parse_args()

    if args.cookies and args.cookies_from_browser:
        parser.error("Use only one of --cookies and --cookies-from-browser")

    cookiesfrombrowser: tuple[str, ...] | None = None
    if args.cookies_from_browser:
        b = args.cookies_from_browser.strip()
        cookiesfrombrowser = (b, args.browser_profile) if args.browser_profile else (b,)

    if not ffmpeg_available():
        raise SystemExit("ffmpeg not found on PATH (install ffmpeg).")

    rows: list[tuple[str, str]]
    if args.url:
        rows = [("submission", args.url.strip())]
    else:
        rows = list(iter_youtube_rows_csv(args.csv))

    summary: list[dict[str, Any]] = []
    for label, url in rows:
        safe = re.sub(r"[^\w.\-@]+", "_", label)[:120]
        wdir = args.out / safe
        wdir.mkdir(parents=True, exist_ok=True)
        _, frames = preprocess_youtube_to_frames(
            url,
            wdir,
            scene_threshold=args.scene_threshold,
            min_gap_seconds=args.min_gap,
            cookiefile=args.cookies,
            cookiesfrombrowser=cookiesfrombrowser,
            youtube_player_client=args.youtube_player_client,
            youtube_player_fallback=not args.no_youtube_player_fallback,
            keep_video=args.keep_video,
        )
        summary.append(
            {
                "label": label,
                "url": url,
                "work_dir": str(wdir.resolve()),
                "frame_count": len(frames),
                "frames": [
                    {"path": str(f.path.resolve()), "time_seconds": f.time_seconds}
                    for f in frames
                ],
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
