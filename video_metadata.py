"""
Video metadata for grading tools: local files (ffprobe) and YouTube URLs (yt-dlp, no download).

Used by the evaluation agent's ``get_video_metadata`` tool.

- **Local files:** requires ``ffprobe`` on PATH (ffmpeg).
- **YouTube:** requires ``yt-dlp`` (``pip install yt-dlp``); uses ``extract_info(..., download=False)``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Match youtube_transcribe.py: try default client, then others when YouTube is picky.
_YOUTUBE_PLAYER_CLIENT_FALLBACKS: tuple[str | None, ...] = (
    None,
    "android",
    "web_embedded",
    "web",
)


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _parse_fraction(s: str | None) -> float | None:
    if not s or "/" not in s:
        return None
    a, _, b = s.partition("/")
    try:
        af, bf = float(a), float(b)
        if bf == 0:
            return None
        return af / bf
    except ValueError:
        return None


def _format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return ""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def probe_local_media_metadata(path: str | Path) -> dict[str, Any]:
    """
    Run ffprobe and return a JSON-serializable dict for tool / API use.

    On success: ``ok`` is True and fields include ``duration_seconds``,
    ``duration_human``, and the first video stream's ``width`` / ``height`` when present.

    On failure: ``ok`` is False and ``error`` explains (missing binary, not a file, probe error).
    """
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"Not a file or missing: {p}"}

    if not ffprobe_available():
        return {
            "ok": False,
            "error": "ffprobe not found on PATH. Install ffmpeg (includes ffprobe) to use video metadata.",
        }

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p.resolve()),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffprobe timed out after 120s"}
    except OSError as e:
        return {"ok": False, "error": f"Could not run ffprobe: {e}"}

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        return {"ok": False, "error": f"ffprobe failed (exit {proc.returncode}): {err}"}

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid ffprobe JSON: {e}"}

    fmt = data.get("format") or {}
    streams = data.get("streams") or []

    duration_seconds: float | None = None
    dur_raw = fmt.get("duration")
    if dur_raw is not None:
        try:
            duration_seconds = float(dur_raw)
        except (TypeError, ValueError):
            pass

    video: dict[str, Any] | None = None
    for s in streams:
        if s.get("codec_type") == "video":
            w = s.get("width")
            h = s.get("height")
            fps = _parse_fraction(s.get("avg_frame_rate")) or _parse_fraction(
                s.get("r_frame_rate")
            )
            video = {
                "codec": s.get("codec_name"),
                "pix_fmt": s.get("pix_fmt"),
                "width": int(w) if w is not None else None,
                "height": int(h) if h is not None else None,
                "fps": round(fps, 3) if fps is not None else None,
            }
            break

    size_b = fmt.get("size")
    file_size_bytes: int | None = None
    if size_b is not None:
        try:
            file_size_bytes = int(float(size_b))
        except (TypeError, ValueError):
            pass

    br = fmt.get("bit_rate")
    bitrate: int | None = None
    if br is not None:
        try:
            bitrate = int(float(br))
        except (TypeError, ValueError):
            pass

    out: dict[str, Any] = {
        "ok": True,
        "source": "local_file",
        "path": str(p.resolve()),
        "format_name": fmt.get("format_name"),
        "duration_seconds": duration_seconds,
        "duration_human": _format_duration(duration_seconds)
        if duration_seconds is not None
        else None,
        "file_size_bytes": file_size_bytes,
        "bitrate": bitrate,
        "video": video,
        "stream_count": len(streams),
    }
    return out


def looks_like_youtube_url(url: str) -> bool:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return False
    try:
        from urllib.parse import urlparse

        host = (urlparse(u).hostname or "").lower()
    except Exception:
        return False
    return (
        host.endswith("youtube.com")
        or host.endswith("youtu.be")
        or host.endswith("youtube-nocookie.com")
    )


def yt_dlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return False


def _youtube_ydl_opts(
    *,
    cookiefile: Path | None,
    cookiesfrombrowser: tuple[str, ...] | None,
    youtube_player_client: str | None,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": False,
    }
    if cookiefile is not None:
        opts["cookiefile"] = str(cookiefile)
    if cookiesfrombrowser is not None:
        opts["cookiesfrombrowser"] = cookiesfrombrowser
    if youtube_player_client is not None:
        opts["extractor_args"] = {"youtube": {"player_client": [youtube_player_client]}}
    return opts


def _youtube_info_to_dict(info: dict[str, Any] | None, requested_url: str) -> dict[str, Any]:
    if not info or not isinstance(info, dict):
        return {"ok": False, "error": "yt-dlp returned no metadata"}

    duration_seconds: float | None = None
    d = info.get("duration")
    if d is not None:
        try:
            duration_seconds = float(d)
        except (TypeError, ValueError):
            pass

    w, h = info.get("width"), info.get("height")
    fps = info.get("fps")
    fps_out: float | None = None
    if fps is not None:
        try:
            fps_out = round(float(fps), 3)
        except (TypeError, ValueError):
            pass

    vc = info.get("vcodec")
    if isinstance(vc, str) and vc in ("none", "null"):
        vc = None

    video: dict[str, Any] | None = None
    if w or h or vc or fps_out:
        video = {
            "codec": vc,
            "width": int(w) if w is not None else None,
            "height": int(h) if h is not None else None,
            "fps": fps_out,
        }

    title = info.get("title")
    desc = info.get("description")
    if isinstance(desc, str) and len(desc) > 400:
        desc = desc[:400] + "…"

    return {
        "ok": True,
        "source": "youtube",
        "url": info.get("webpage_url") or requested_url,
        "video_id": info.get("id"),
        "title": title if isinstance(title, str) else None,
        "description_excerpt": desc if isinstance(desc, str) else None,
        "duration_seconds": duration_seconds,
        "duration_human": _format_duration(duration_seconds)
        if duration_seconds is not None
        else None,
        "format_name": info.get("ext") or "youtube",
        "file_size_bytes": None,
        "bitrate": None,
        "video": video,
        "stream_count": None,
    }


def probe_youtube_metadata(
    url: str,
    *,
    cookiefile: Path | None = None,
    cookiesfrombrowser: tuple[str, ...] | None = None,
    youtube_player_client: str | None = None,
    youtube_player_fallback: bool = True,
) -> dict[str, Any]:
    """
    Fetch public metadata for a YouTube video URL without downloading the video.

    Uses yt-dlp's Python API (same family of options as ``audio-transcription/youtube_transcribe.py``).
    On failure (network, age-restricted without cookies, etc.), returns ``ok: false`` and ``error``.
    """
    u = (url or "").strip()
    if not u:
        return {"ok": False, "error": "Empty URL"}
    if not looks_like_youtube_url(u):
        return {
            "ok": False,
            "error": "URL does not look like a YouTube link (expected youtube.com or youtu.be).",
        }

    try:
        import yt_dlp
    except ImportError:
        return {
            "ok": False,
            "error": "yt-dlp is not installed. pip install yt-dlp",
        }

    if youtube_player_client is not None:
        clients: tuple[str | None, ...] = (youtube_player_client,)
    elif youtube_player_fallback:
        clients = _YOUTUBE_PLAYER_CLIENT_FALLBACKS
    else:
        clients = (None,)

    last_err: str | None = None
    for pc in clients:
        opts = _youtube_ydl_opts(
            cookiefile=cookiefile,
            cookiesfrombrowser=cookiesfrombrowser,
            youtube_player_client=pc,
        )
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(u, download=False)
            return _youtube_info_to_dict(info, u)
        except Exception as e:
            last_err = str(e).strip() or repr(e)
            continue

    return {
        "ok": False,
        "error": last_err or "yt-dlp could not extract YouTube metadata",
    }


def probe_video_metadata_for_tool(
    *,
    local_path: Path | None,
    youtube_url: str | None,
    cookiefile: Path | None = None,
    cookiesfrombrowser: tuple[str, ...] | None = None,
    youtube_player_client: str | None = None,
    youtube_player_fallback: bool = True,
) -> dict[str, Any]:
    """
    Single entry for ``get_video_metadata``: exactly one of local_path or youtube_url must be set.
    """
    if youtube_url and local_path:
        return {"ok": False, "error": "Internal error: both local_path and youtube_url set"}
    if youtube_url:
        return probe_youtube_metadata(
            youtube_url,
            cookiefile=cookiefile,
            cookiesfrombrowser=cookiesfrombrowser,
            youtube_player_client=youtube_player_client,
            youtube_player_fallback=youtube_player_fallback,
        )
    if local_path:
        return probe_local_media_metadata(local_path)
    return {"ok": False, "error": "No video source configured (neither file nor YouTube URL)."}
