#!/usr/bin/env python3
"""
Download a YouTube video's audio and transcribe it with OpenAI (same API as main.py).

Unlisted videos do not need special flags—only people with the link can open them, same as
in a browser. If yt-dlp says "not available" but the video plays when you are logged in,
use --cookies-from-browser or --cookies so the download uses your session.

If the video plays in an incognito window but yt-dlp still fails, YouTube is likely rejecting
the default internal client: upgrade yt-dlp (`pip install -U yt-dlp`) first. This script then
retries other YouTube player clients automatically unless you disable that with
--no-youtube-player-fallback.

If you see warnings about no JavaScript runtime or "n challenge solving failed", the web
clients may only expose thumbnails until you install a supported JS runtime (see yt-dlp's
EJS wiki: https://github.com/yt-dlp/yt-dlp/wiki/EJS ). The `android` client often still
returns a combined audio+video format (e.g. itag 18); ffmpeg then strips audio to MP3.

Requires: pip install yt-dlp openai python-dotenv
System: ffmpeg must be installed (yt-dlp uses it to extract MP3).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI
from yt_dlp.utils import DownloadError

load_dotenv()

# When the default YouTube client returns "not available" but the video plays in a browser,
# trying other clients often fixes it (YouTube API differences, not login).
# Try android before web_* : without a JS runtime, web clients often fail the "n" challenge
# and only expose storyboard images; android may still serve a progressive A+V stream (itag 18).
_YOUTUBE_PLAYER_CLIENT_FALLBACKS: tuple[str | None, ...] = (
    None,
    "android",
    "web_embedded",
    "web",
)


def _build_ydl_opts(
    workdir: Path,
    *,
    cookiefile: Path | None,
    cookiesfrombrowser: tuple[str, ...] | None,
    youtube_player_client: str | None,
) -> dict:
    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    if cookiefile is not None:
        ydl_opts["cookiefile"] = str(cookiefile)
    if cookiesfrombrowser is not None:
        ydl_opts["cookiesfrombrowser"] = cookiesfrombrowser
    if youtube_player_client is not None:
        ydl_opts["extractor_args"] = {
            "youtube": {"player_client": [youtube_player_client]},
        }
    return ydl_opts


def _clear_workdir(workdir: Path) -> None:
    if not workdir.exists():
        return
    for child in workdir.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child, ignore_errors=True)


def _download_audio_mp3_once(
    url: str,
    workdir: Path,
    *,
    cookiefile: Path | None,
    cookiesfrombrowser: tuple[str, ...] | None,
    youtube_player_client: str | None,
) -> Path:
    """Single yt-dlp attempt; workdir must be empty or cleared by caller."""
    ydl_opts = _build_ydl_opts(
        workdir,
        cookiefile=cookiefile,
        cookiesfrombrowser=cookiesfrombrowser,
        youtube_player_client=youtube_player_client,
    )
    label = youtube_player_client or "default"
    print(f"yt-dlp: trying YouTube player_client={label!r}", flush=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        if not video_id:
            raise RuntimeError("yt-dlp did not return a video id")
        mp3_path = workdir / f"{video_id}.mp3"
        if mp3_path.is_file():
            return mp3_path
    mp3s = sorted(workdir.glob("*.mp3"))
    if len(mp3s) == 1:
        return mp3s[0]
    raise FileNotFoundError(f"Expected one MP3 in {workdir}, found: {mp3s}")


def download_audio_mp3(
    url: str,
    workdir: Path,
    *,
    cookiefile: Path | None = None,
    cookiesfrombrowser: tuple[str, ...] | None = None,
    youtube_player_client: str | None = None,
    youtube_player_fallback: bool = True,
) -> Path:
    """Download best-quality audio and convert to MP3 in workdir."""
    if youtube_player_client is not None:
        clients: tuple[str | None, ...] = (youtube_player_client,)
    elif youtube_player_fallback:
        clients = _YOUTUBE_PLAYER_CLIENT_FALLBACKS
    else:
        clients = (None,)

    last_err: BaseException | None = None
    for pc in clients:
        _clear_workdir(workdir)
        try:
            return _download_audio_mp3_once(
                url,
                workdir,
                cookiefile=cookiefile,
                cookiesfrombrowser=cookiesfrombrowser,
                youtube_player_client=pc,
            )
        except DownloadError as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def transcribe_file(client: OpenAI, audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
        )
    return transcription.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and print an OpenAI transcription."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--keep-audio",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, copy the downloaded MP3 to this path after transcribing",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        metavar="FILE",
        help="Netscape-format cookies.txt (e.g. from 'Get cookies.txt' extension)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="Load cookies from a local browser (e.g. chrome, firefox, safari, brave)",
    )
    parser.add_argument(
        "--browser-profile",
        type=str,
        default=None,
        metavar="NAME",
        help="Browser profile with --cookies-from-browser (e.g. Default)",
    )
    parser.add_argument(
        "--youtube-player-client",
        type=str,
        default=None,
        metavar="NAME",
        help="Force a single YouTube player client (e.g. web_embedded); disables auto fallback",
    )
    parser.add_argument(
        "--no-youtube-player-fallback",
        action="store_true",
        help="Only use yt-dlp's default YouTube client (no automatic retries)",
    )
    args = parser.parse_args()

    if args.cookies and args.cookies_from_browser:
        parser.error("Use only one of --cookies and --cookies-from-browser")
    if args.youtube_player_client and args.no_youtube_player_fallback:
        parser.error(
            "--youtube-player-client already implies a single client; "
            "omit --no-youtube-player-fallback"
        )

    cookiesfrombrowser: tuple[str, ...] | None = None
    if args.cookies_from_browser:
        b = args.cookies_from_browser.strip()
        cookiesfrombrowser = (b, args.browser_profile) if args.browser_profile else (b,)

    client = OpenAI()
    tmp_root = Path(tempfile.mkdtemp(prefix="yt-transcribe-"))
    try:
        audio_path = download_audio_mp3(
            args.url,
            tmp_root,
            cookiefile=args.cookies,
            cookiesfrombrowser=cookiesfrombrowser,
            youtube_player_client=args.youtube_player_client,
            youtube_player_fallback=not args.no_youtube_player_fallback,
        )
        if args.keep_audio:
            args.keep_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, args.keep_audio)
            audio_path = args.keep_audio
        text = transcribe_file(client, audio_path)
        print(text)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
