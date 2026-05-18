# ENTRY POINT FOR PREPROCESSING VIDEOS

from pathlib import Path
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))

from video.download_yt_video import download_youtube_video
from video.transcribe import transcribe_video
from video.extract_frames import extact_important_frames
from video.annotate_frames import annotate_frames

def preprocess_video(url: str, alias: str, preprocess_dir: Path) -> None:
    video_dir = preprocess_dir / alias
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = download_youtube_video(url, video_dir, "video")

    # Transcription
    transcript_path = preprocess_dir / alias / "transcript.txt"
    transcribe_video(video_path, transcript_path)

    # Frame extraction and summarization
    frames_dir = preprocess_dir / alias / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = extact_important_frames(video_path, frames_dir)
    summary_path = preprocess_dir / alias / "frames_summary.json"
    annotate_frames(frames, summary_path)

    # Finally, delete the video file
    video_path.unlink(missing_ok=True)


def preprocess_video_list(assigment_list_csv_path: Path, preprocess_dir: Path) -> None:
    df = pd.read_csv(assigment_list_csv_path)

    for index, row in df.iterrows():
        url = row["link"]
        alias = row["email"].split("@")[0]
        print(f"Preprocessing alias {alias}")
        preprocess_video(url, alias, preprocess_dir)


if __name__ == "__main__":

    assignment_list_csv_path = Path("student-responses/gsu-student-sumprod-video-list-short.csv")
    preprocess_dir = Path("preprocess-test")
    preprocess_video_list(assignment_list_csv_path, preprocess_dir)