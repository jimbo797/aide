from pathlib import Path

from src.eval.preprocess.excel.preprocess import preprocess_excel_sheet
from src.eval.preprocess.video.preprocess import preprocess_video_file
from src.util import log


EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def preprocess_artifacts(
    alias: str,
    submissions_dir: Path = Path("in"),
    preprocess_dir: Path = Path("out/preprocess"),
    *,
    sheet_name: str | None = None,
    excel_password: str | None = None,
) -> list[Path]:
    """Preprocess every supported artifact in ``submissions_dir / alias``.

    Each source file gets a separate output directory so multiple artifacts of
    the same modality cannot overwrite one another:
    ``<preprocess_dir>/<alias>/<source filename>/``. Periods in artifact path
    components are replaced with dashes for safe directory names.

    Returns the output directories for the artifacts that were processed.
    """
    submission_dir = submissions_dir / alias
    if not submission_dir.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    processed_dirs: list[Path] = []
    artifact_paths = sorted(
        path for path in submission_dir.rglob("*") if path.is_file()
    )

    for artifact_path in artifact_paths:
        suffix = artifact_path.suffix.lower()
        relative_path = artifact_path.relative_to(submission_dir)
        safe_relative_path = Path(
            *(part.replace(".", "-") for part in relative_path.parts)
        )
        artifact_alias = str(Path(alias) / safe_relative_path)
        artifact_output_dir = preprocess_dir / artifact_alias

        if suffix in EXCEL_SUFFIXES:
            log(alias, f"Preprocessing spreadsheet: {relative_path}")
            preprocess_excel_sheet(
                filepath=str(artifact_path),
                alias=artifact_alias,
                preprocess_dir=preprocess_dir,
                sheet_name=sheet_name,
                password=excel_password,
            )
        elif suffix in VIDEO_SUFFIXES:
            log(alias, f"Preprocessing video: {relative_path}")
            preprocess_video_file(
                video_path=artifact_path,
                alias=artifact_alias,
                preprocess_dir=preprocess_dir,
            )
        else:
            log(alias, f"Skipping unsupported artifact: {relative_path}")
            continue

        processed_dirs.append(artifact_output_dir)

    return processed_dirs
