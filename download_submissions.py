from pathlib import Path

import pandas as pd

from src.eval.preprocess.video.download_yt_video import download_youtube_video
from src.util import log


def download_submissions(
    student_responses_csv_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Download every video in a student-responses CSV.

    Videos use the same directory layout and download settings as
    ``preprocess_video``: ``<output_dir>/<email alias>/video.<extension>``.
    Both the newer ``URL`` column and the older ``link`` column are supported.
    A failed submission is logged without stopping the remaining downloads.
    """
    responses = pd.read_csv(student_responses_csv_path)
    url_column = "URL" if "URL" in responses.columns else "link"

    required_columns = {"email", url_column}
    missing_columns = required_columns.difference(responses.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required CSV column(s): {missing}")

    downloaded: dict[str, Path] = {}
    for _, row in responses.iterrows():
        email = str(row["email"]).strip()
        alias = email.split("@", maxsplit=1)[0]
        url = str(row[url_column]).strip()

        if not alias or not url or url.lower() == "nan":
            log(alias or "unknown", "Skipping row with missing email or video URL")
            continue

        try:
            log(alias, "Downloading video")
            downloaded[alias] = download_youtube_video(
                url,
                output_dir / alias,
                "video",
                max_height=1080,
            )
        except Exception as exc:
            log(alias, f"Error downloading video: {exc}")

    return downloaded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download videos listed in a student-responses CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("in"),
    )
    args = parser.parse_args()

    downloads = download_submissions(args.csv_path, args.output_dir)
    print(f"Downloaded {len(downloads)} submission(s) to {args.output_dir}")
