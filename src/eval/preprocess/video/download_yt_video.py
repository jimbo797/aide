from pathlib import Path
from typing import Any
import os
import yt_dlp
import asyncio

def download_youtube_video(url: str, out_dir: Path, out_filename: str = "video") -> Path:
    out_path = out_dir / out_filename

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
    # if DEBUG:
    #     print(f"Video {url} downloaded successfully")
    return Path(f"{out_path}.mp4")

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("student-responses/gsu-student-sumprod-video-list-short.csv")
    out_dir = Path("download-test")
    for index, row in df.iterrows():
        url = row["link"]
        alias = row["email"].split("@")[0]
        download_youtube_video(url, out_dir, f"{alias}")
