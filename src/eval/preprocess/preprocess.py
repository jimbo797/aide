# ENTRY POINT FOR PREPROCESSING VIDEOS

from pathlib import Path

import pandas as pd

from src.eval.preprocess.video.annotate_frames import annotate_frames
from src.eval.preprocess.video.download_yt_video import download_youtube_video
from src.eval.preprocess.video.extract_frames import extact_important_frames
from src.eval.preprocess.video.transcribe import transcribe_video
from src.util import log

def preprocess_video(url: str, alias: str, preprocess_dir: Path) -> None:
    video_dir = preprocess_dir / alias
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = download_youtube_video(url, video_dir, "video", max_height=1080)

    # Transcription
    transcript_path = preprocess_dir / alias / "transcript.txt"
    log(alias, "Transcribing")
    transcribe_video(video_path, transcript_path)

    # Frame extraction and summarization
    frames_dir = preprocess_dir / alias / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    log(alias, "Extracting frames")
    frames = extact_important_frames(
        video_path, 
        frames_dir, 
        scene_threshold=0.1, 
        scale_max_width=1080
    )
    summary_path = preprocess_dir / alias / "frames_summary.json"
    
    log(alias, "Annotating frames")
    annotate_frames(frames, summary_path)

    # Delete the video file
    # TODO: Maybe keep it until eval for a student is complete
    video_path.unlink(missing_ok=True)


def preprocess_video_list(assigment_list_csv_path: Path, preprocess_dir: Path) -> None:
    df = pd.read_csv(assigment_list_csv_path)

    for index, row in df.iterrows():
        url = row["link"]
        alias = row["email"].split("@")[0]
        preprocess_video(url, alias, preprocess_dir)


if __name__ == "__main__":

    assignment_list_csv_path = Path("student-responses/gsu-student-sumprod-video-list.csv")
    preprocess_dir = Path("preprocess-test")
    preprocess_video_list(assignment_list_csv_path, preprocess_dir)