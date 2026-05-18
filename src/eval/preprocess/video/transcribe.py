from pathlib import Path

from src.util.extract_audio import extract_mp3_from_video
from src.util.transcribe import transcribe


def transcribe_video(video_path: Path, transcript_path: Path) -> Path:
    mp3_path = video_path.with_suffix(".mp3")
    extract_mp3_from_video(video_path, mp3_path)

    with mp3_path.open("rb") as audio_file:
        text = transcribe(audio_file)

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")

    mp3_path.unlink(missing_ok=True)
