from pathlib import Path
from typing import Any
import os
import yt_dlp
import asyncio

DEBUG = True

def download_youtube_video(url: str, out_dir: Path, out_filename: str = "video") -> Path:
    out_path = out_dir / out_filename

    if DEBUG:
        print(f"Downloading video from {url}")

    ydl_opts = {
        "outtmpl": str(out_path) + ".%(ext)s",
        "format": "mp4",
        # "format": "bv*+ba/b",
        # "cookiesfrombrowser": ("chrome",),
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },
        # "verbose": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {
            "node": {
                "path": "/Users/jimmy/.nvm/versions/node/v21.1.0/bin/node"
            }
        },
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # print(info)
    except Exception as e:
        print(f"*** ERROR *** Error downloading video {url}: {e}")
    if DEBUG:
        print(f"Video {url} downloaded successfully")
    return Path(f"{out_path}.mp4")

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("student-responses/gsu-student-sumprod-video-list-short.csv")
    out_dir = Path("download-test")
    for index, row in df.iterrows():
        url = row["link"]
        alias = row["email"].split("@")[0]
        download_youtube_video(url, out_dir, f"{alias}")


# _YOUTUBE_PLAYER_CLIENT_FALLBACKS: tuple[str | None, ...] = (
#     None,
#     "android",
#     "web_embedded",
#     "web",
# )

# def _ydl_video_opts(
#     out_dir: Path,
#     *,
#     cookiefile: Path | None,
#     cookiesfrombrowser: tuple[str, ...] | None,
#     youtube_player_client: str | None,
# ) -> dict[str, Any]:
#     opts: dict[str, Any] = {
#         # Prefer mp4 for downstream ffmpeg; fall back to best merged format.
#         "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/best",
#         "merge_output_format": "mp4",
#         "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
#         "quiet": True,
#         "no_warnings": True,
#     }
#     if cookiefile is not None:
#         opts["cookiefile"] = str(cookiefile)
#     if cookiesfrombrowser is not None:
#         opts["cookiesfrombrowser"] = cookiesfrombrowser
#     if youtube_player_client is not None:
#         opts["extractor_args"] = {"youtube": {"player_client": [youtube_player_client]}}
#     return opts


# def _download_youtube_once(
#     url: str,
#     out_dir: Path,
#     *,
#     cookiefile: Path | None,
#     cookiesfrombrowser: tuple[str, ...] | None,
#     youtube_player_client: str | None,
# ) -> Path:
#     import yt_dlp

#     ydl_opts = _ydl_video_opts(
#         out_dir,
#         cookiefile=cookiefile,
#         cookiesfrombrowser=cookiesfrombrowser,
#         youtube_player_client=youtube_player_client,
#     )
#     label = youtube_player_client or "default"
#     print(f"yt-dlp: trying YouTube player_client={label!r}", flush=True)
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         if not isinstance(info, dict):
#             raise RuntimeError("yt-dlp returned unexpected info type")
#         vid = info.get("id")
#         if not vid:
#             raise RuntimeError("yt-dlp did not return a video id")
#         # Requested_filename after merge is most reliable
#         rp = info.get("requested_downloads")
#         if isinstance(rp, list) and rp:
#             fp = rp[0].get("filepath")
#             if fp and Path(fp).is_file():
#                 return Path(fp)
#         # Fallback: common extensions
#         for ext in ("mp4", "webm", "mkv"):
#             p = out_dir / f"{vid}.{ext}"
#             if p.is_file():
#                 return p
#     raise FileNotFoundError(f"No video file found in {out_dir} after yt-dlp download")


# # MAIN FUNCTION FOR DOWNLOADING YOUTUBE VIDEO
# def download_youtube_video(
#     url: str,
#     out_dir: Path,
#     *,
#     cookiefile: Path | None = None,
#     cookiesfrombrowser: tuple[str, ...] | None = None,
#     youtube_player_client: str | None = None,
#     youtube_player_fallback: bool = True,
# ) -> Path:
#     """
#     Download one YouTube URL to ``out_dir`` and return the path to the merged video file.
#     """
#     try:
#         import yt_dlp  # noqa: F401
#         from yt_dlp.utils import DownloadError as YtdlDownloadError
#     except ImportError as e:
#         raise RuntimeError("yt-dlp is not installed. pip install yt-dlp") from e

#     out_dir.mkdir(parents=True, exist_ok=True)
#     if youtube_player_client is not None:
#         clients: tuple[str | None, ...] = (youtube_player_client,)
#     elif youtube_player_fallback:
#         clients = _YOUTUBE_PLAYER_CLIENT_FALLBACKS
#     else:
#         clients = (None,)
#     last_err: BaseException | None = None
#     for pc in clients:
#         try:
#             return _download_youtube_once(
#                 url,
#                 out_dir,
#                 cookiefile=cookiefile,
#                 cookiesfrombrowser=cookiesfrombrowser,
#                 youtube_player_client=pc,
#             )
#         except YtdlDownloadError as e:
#             last_err = e
#             continue
#     assert last_err is not None
#     raise last_err
