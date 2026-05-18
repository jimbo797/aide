#!/usr/bin/env python3
"""
End-to-end pipeline: CSV (email, YouTube link) → frame extraction → **audio transcript**
(``audio-transcription/youtube_transcribe.py``) → vision summaries → per-leaf eval_agent
→ weighted score aggregate per student.

Transcripts are written to ``<extract work_dir>/transcript.txt`` and prepended into each
student's ``submission.txt`` for ``read_submission`` / ``search_submission``.

Run from the ``aide`` directory (same as other scripts)::

    python3 student_video_eval_pipeline.py student-responses/gsu-student-sumprod-video-list.csv

Requires: ffmpeg, yt-dlp, OPENAI_API_KEY; optional envs match ``video_frame_summarize.py``,
``eval_agent.py``, and ``youtube_transcribe.py`` (audio model default: gpt-4o-transcribe).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

AIDE_DIR = Path(__file__).resolve().parent


def _safe_label(label: str) -> str:
    return re.sub(r"[^\w.\-@]+", "_", label)[:120]


def _run_extract(csv_path: Path, extract_root: Path, extra: list[str]) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        str(AIDE_DIR / "video_frame_extract.py"),
        "--csv",
        str(csv_path.resolve()),
        "--out",
        str(extract_root.resolve()),
        *extra,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(AIDE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        out = (proc.stdout or "").strip()
        sys.stderr.write(proc.stderr or "")
        sys.stderr.write(proc.stdout or "")
        detail = err or out or "(no stdout/stderr)"
        raise RuntimeError(
            f"video_frame_extract.py failed (exit {proc.returncode}): {detail[:2000]}"
        )
    raw_out = (proc.stdout or "").strip()
    if not raw_out:
        sys.stderr.write(proc.stderr or "")
        raise RuntimeError(
            "video_frame_extract.py produced no stdout (expected a JSON array). "
            f"stderr tail: {(proc.stderr or '')[-1500:]!r}"
        )
    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError as e:
        sys.stderr.write(proc.stderr or "")
        raise RuntimeError(
            f"video_frame_extract.py stdout is not valid JSON ({e}). "
            f"First 500 chars: {raw_out[:500]!r}"
        ) from e
    if not isinstance(data, list):
        raise ValueError("video_frame_extract.py did not print a JSON array")
    return data


def _run_transcribe(url: str, extra: list[str]) -> str:
    """Run ``youtube_transcribe.py``; return plain transcript text (stdout)."""
    cmd = [
        sys.executable,
        str(AIDE_DIR / "audio-transcription" / "youtube_transcribe.py"),
        url,
        *extra,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(AIDE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        out = (proc.stdout or "").strip()
        sys.stderr.write(proc.stderr or "")
        sys.stderr.write(proc.stdout or "")
        detail = err or out or "(no stdout/stderr)"
        raise RuntimeError(
            f"youtube_transcribe.py failed (exit {proc.returncode}): {detail[:2000]}"
        )
    return (proc.stdout or "").strip()


def _yt_dlp_shared_args(args: argparse.Namespace) -> list[str]:
    """CLI flags shared by ``video_frame_extract`` and ``youtube_transcribe``."""
    out: list[str] = []
    if args.cookies:
        out.extend(["--cookies", str(args.cookies.resolve())])
    if args.cookies_from_browser:
        out.extend(["--cookies-from-browser", args.cookies_from_browser.strip()])
        if args.browser_profile:
            out.extend(["--browser-profile", args.browser_profile])
    if args.youtube_player_client:
        out.extend(["--youtube-player-client", args.youtube_player_client.strip()])
    if args.no_youtube_player_fallback:
        out.append("--no-youtube-player-fallback")
    return out


def _transcribe_manifest(
    manifest: list[dict[str, Any]],
    *,
    skip_transcribe: bool,
    force_transcribe: bool,
    transcribe_extra: list[str],
) -> None:
    if skip_transcribe:
        print("[transcribe] skipped (--skip-transcribe)", flush=True)
        return
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        wd = (entry.get("work_dir") or "").strip()
        label = (entry.get("label") or "").strip() or url
        if not url or not wd:
            continue
        if (entry.get("error") or "").strip():
            print(f"[transcribe] skip {label} (extract failed)", flush=True)
            continue
        out_txt = Path(wd) / "transcript.txt"
        if out_txt.is_file() and out_txt.stat().st_size > 0 and not force_transcribe:
            print(f"[transcribe] reuse {label}", flush=True)
            continue
        print(f"[transcribe] {label}", flush=True)
        text = _run_transcribe(url, transcribe_extra)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _run_summarize(summary_json: Path, output_json: Path, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(AIDE_DIR / "video_frame_summarize.py"),
        "--summary-json",
        str(summary_json.resolve()),
        "--output",
        str(output_json.resolve()),
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=str(AIDE_DIR), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        detail = (proc.stderr or proc.stdout or "").strip() or "(no stdout/stderr)"
        raise RuntimeError(
            f"video_frame_summarize.py failed (exit {proc.returncode}): {detail[:2000]}"
        )


def _summaries_for_label(all_summaries: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for row in all_summaries if (row.get("label") or "") == label]


def build_submission_text(url: str, work_dir_str: str) -> str:
    """URL line plus optional ``transcript.txt`` from extract ``work_dir``, then rubric hint."""
    blocks: list[str] = [
        "Student assignment video (YouTube):\n" + url + "\n",
    ]
    wd = (work_dir_str or "").strip()
    if wd:
        tp = Path(wd) / "transcript.txt"
        if tp.is_file() and tp.stat().st_size > 0:
            blocks.append("--- Audio transcription (speech-to-text) ---\n\n")
            blocks.append(tp.read_text(encoding="utf-8").rstrip() + "\n")
    blocks.append(
        "\nUse the configured tools (video metadata, frame summaries) as needed to evaluate "
        "this submission against the rubric.\n"
    )
    return "".join(blocks)


def write_student_preprocess_artifacts(
    work: Path,
    manifest: list[dict[str, Any]],
    all_summaries: list[dict[str, Any]],
) -> None:
    """Under ``work/students/<label>/``: ``submission.txt`` and optional ``video_frame_summaries.json``."""
    students_dir = work / "students"
    students_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        label = (entry.get("label") or "").strip()
        url = (entry.get("url") or "").strip()
        work_dir_str = (entry.get("work_dir") or "").strip()
        if not label or not url:
            continue
        sub_dir = students_dir / _safe_label(label)
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "submission.txt").write_text(
            build_submission_text(url, work_dir_str),
            encoding="utf-8",
        )
        per_summaries = _summaries_for_label(all_summaries, label)
        summaries_path = sub_dir / "video_frame_summaries.json"
        if per_summaries:
            summaries_path.write_text(json.dumps(per_summaries, indent=2), encoding="utf-8")
        elif summaries_path.exists():
            summaries_path.unlink()


def aggregate_weighted_leaf_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Simple policy: ``met`` earns ``leaf_weight``; ``not_met`` / ``undetermined`` earn 0.
    Sums only leaves with positive ``leaf_weight`` into ``possible_points``.
    """
    earned = 0.0
    possible = 0.0
    by_verdict: dict[str, int] = {"met": 0, "not_met": 0, "undetermined": 0}
    for r in rows:
        v = r.get("verdict")
        if v not in ("met", "not_met", "undetermined"):
            v = "undetermined"
        by_verdict[str(v)] += 1
        w = r.get("leaf_weight")
        try:
            wf = float(w) if w is not None else 0.0
        except (TypeError, ValueError):
            wf = 0.0
        if wf <= 0:
            continue
        possible += wf
        if r.get("verdict") == "met":
            earned += wf
    pct = (earned / possible * 100.0) if possible > 0 else None
    return {
        "earned_points": round(earned, 4),
        "possible_points": round(possible, 4),
        "percent_of_graded_leaves": round(pct, 2) if pct is not None else None,
        "leaf_counts_by_verdict": by_verdict,
    }


def main() -> None:
    default_csv = AIDE_DIR / "student-responses" / "gsu-student-sumprod-video-list.csv"
    default_rkt = AIDE_DIR / "rubrics" / "gsu-sumprod.json"

    parser = argparse.ArgumentParser(
        description=(
            "CSV of student emails + YouTube links → extract → transcribe → summarize → "
            "eval_agent → aggregates."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=default_csv,
        help=f"CSV with email + link/url columns (default: {default_csv})",
    )
    parser.add_argument(
        "--rkt-json",
        type=Path,
        default=default_rkt,
        help=f"RKT skill tree JSON (default: {default_rkt})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=AIDE_DIR / "pipeline_out" / "gsu-sumprod-batch",
        help="Root output directory for extract trees, summaries, assessments",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override eval agent chat model (else OPENAI_EVAL_AGENT_MODEL or default)",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Pass to video_frame_summarize.py --model",
    )
    parser.add_argument(
        "--max-turns-per-leaf",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Do not call youtube_transcribe.py; still use existing transcript.txt if present",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Re-run transcription even when transcript.txt already exists",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing extract_manifest.json under --work-dir (must exist)",
    )
    parser.add_argument(
        "--skip-summarize",
        action="store_true",
        help="Reuse existing frame_summaries_all.json under --work-dir",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only preprocess (extract + summarize); do not call eval_agent",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Forward --keep-video to video_frame_extract.py",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        metavar="FILE",
        help="Netscape cookies.txt for yt-dlp (extract + transcribe)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="yt-dlp cookies from browser (extract + transcribe)",
    )
    parser.add_argument(
        "--browser-profile",
        type=str,
        default=None,
        metavar="NAME",
        help="With --cookies-from-browser",
    )
    parser.add_argument(
        "--youtube-player-client",
        type=str,
        default=None,
        metavar="NAME",
        help="Force yt-dlp YouTube client (extract + transcribe)",
    )
    parser.add_argument(
        "--no-youtube-player-fallback",
        action="store_true",
        help="Disable multi-client retries for YouTube (extract + transcribe)",
    )
    parser.add_argument(
        "--summarize-sleep",
        type=float,
        default=0.0,
        help="Seconds between vision API calls (passed to video_frame_summarize.py --sleep)",
    )
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=None,
        metavar="PX",
        help="Forwarded to video_frame_summarize.py",
    )
    args = parser.parse_args()
    if args.cookies and args.cookies_from_browser:
        parser.error("Use only one of --cookies and --cookies-from-browser")

    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    extract_root = work / "extract"
    manifest_path = work / "extract_manifest.json"
    all_summaries_path = work / "frame_summaries_all.json"

    extract_extra: list[str] = _yt_dlp_shared_args(args)
    if args.keep_video:
        extract_extra.append("--keep-video")

    if args.skip_extract:
        if not manifest_path.is_file():
            raise SystemExit(f"--skip-extract but missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[extract] CSV → {extract_root}", flush=True)
        manifest = _run_extract(csv_path, extract_root, extract_extra)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[extract] Wrote {manifest_path}", flush=True)

    transcribe_extra = _yt_dlp_shared_args(args)
    _transcribe_manifest(
        manifest,
        skip_transcribe=args.skip_transcribe,
        force_transcribe=args.force_transcribe,
        transcribe_extra=transcribe_extra,
    )

    summarize_extra: list[str] = []
    if args.vision_model:
        summarize_extra.extend(["--model", args.vision_model])
    if args.summarize_sleep > 0:
        summarize_extra.extend(["--sleep", str(args.summarize_sleep)])
    if args.max_long_edge is not None:
        summarize_extra.extend(["--max-long-edge", str(args.max_long_edge)])

    total_frames = sum(int(e.get("frame_count") or 0) for e in manifest if isinstance(e, dict))

    if args.skip_summarize:
        if not all_summaries_path.is_file():
            raise SystemExit(f"--skip-summarize but missing {all_summaries_path}")
        all_summaries = json.loads(all_summaries_path.read_text(encoding="utf-8"))
    elif total_frames == 0:
        print("[summarize] No frames in manifest; writing empty summaries array.", flush=True)
        all_summaries_path.write_text("[]", encoding="utf-8")
        all_summaries = []
    else:
        print(f"[summarize] {manifest_path} → {all_summaries_path}", flush=True)
        _run_summarize(manifest_path, all_summaries_path, summarize_extra)
        all_summaries = json.loads(all_summaries_path.read_text(encoding="utf-8"))

    write_student_preprocess_artifacts(work, manifest, all_summaries)

    if args.skip_eval:
        print("[eval] skipped (--skip-eval)", flush=True)
        pre_roster: list[dict[str, Any]] = []
        for entry in manifest:
            if not isinstance(entry, dict):
                continue
            label = (entry.get("label") or "").strip()
            url = (entry.get("url") or "").strip()
            work_dir_str = (entry.get("work_dir") or "").strip()
            if not label or not url:
                continue
            sub_path = work / "students" / _safe_label(label) / "submission.txt"
            tp_rel: str | None = None
            if work_dir_str:
                tpath = Path(work_dir_str) / "transcript.txt"
                if tpath.is_file():
                    try:
                        tp_rel = str(tpath.resolve().relative_to(work))
                    except ValueError:
                        tp_rel = str(tpath)
            pre_roster.append(
                {
                    "email": label,
                    "youtube_url": url,
                    "extract_work_dir": work_dir_str or None,
                    "transcript_txt": tp_rel,
                    "submission_txt": str(sub_path.relative_to(work)),
                }
            )
        index_path = work / "batch_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "csv": str(csv_path),
                    "phase": "preprocess_only",
                    "extract_manifest": str(manifest_path.relative_to(work)),
                    "frame_summaries_all": str(all_summaries_path.relative_to(work)),
                    "students": pre_roster,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[done] preprocess only; wrote {index_path}", flush=True)
        return

    sys.path.insert(0, str(AIDE_DIR))
    from eval_agent import (  # noqa: E402
        YouTubeMediaConfig,
        evaluate_submission_per_leaf_agents,
        load_rkt_json,
    )

    rkt_path = args.rkt_json.resolve()
    rubric = load_rkt_json(rkt_path)

    students_dir = work / "students"

    roster: list[dict[str, Any]] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        label = (entry.get("label") or "").strip()
        url = (entry.get("url") or "").strip()
        work_dir_str = (entry.get("work_dir") or "").strip()
        if not label or not url:
            continue

        sub_dir = students_dir / _safe_label(label)
        submission_path = sub_dir / "submission.txt"
        if not submission_path.is_file():
            raise SystemExit(f"Missing preprocess artifact: {submission_path}")

        summaries_path = sub_dir / "video_frame_summaries.json"
        summaries_arg: Path | None = summaries_path if summaries_path.is_file() else None

        extract_err = (entry.get("error") or "").strip()
        if extract_err:
            print(f"[eval] skip {label} (extract failed)", flush=True)
            merged: list[dict[str, Any]] = []
            transcripts: dict[str, Any] = {}
        else:
            print(f"[eval] {label}", flush=True)
            text = submission_path.read_text(encoding="utf-8")
            youtube_cfg = YouTubeMediaConfig(url=url)

            eval_kwargs: dict[str, Any] = {
                "rubric": rubric,
                "submission_text": text,
                "max_turns_per_leaf": args.max_turns_per_leaf,
                "video_path": None,
                "youtube": youtube_cfg,
                "video_frame_summaries_path": summaries_arg,
            }
            if args.model:
                eval_kwargs["model"] = args.model

            merged, transcripts = evaluate_submission_per_leaf_agents(**eval_kwargs)

        assessment_path = sub_dir / "assessment_leaves.json"
        assessment_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        transcript_path = sub_dir / "assessment_transcripts.json"
        transcript_path.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False), encoding="utf-8")

        agg = aggregate_weighted_leaf_scores(merged)
        tp_rel: str | None = None
        if work_dir_str:
            tpath = Path(work_dir_str) / "transcript.txt"
            if tpath.is_file():
                try:
                    tp_rel = str(tpath.resolve().relative_to(work))
                except ValueError:
                    tp_rel = str(tpath)

        roster.append(
            {
                "email": label,
                "youtube_url": url,
                "extract_work_dir": work_dir_str or None,
                "transcript_txt": tp_rel,
                "submission_txt": str(submission_path.relative_to(work)),
                "assessment_json": str(assessment_path.relative_to(work)),
                "aggregate": agg,
                "extract_error": extract_err or None,
            }
        )
        pct = agg["percent_of_graded_leaves"]
        pct_s = f"{pct}%" if pct is not None else "n/a"
        if extract_err:
            print(f"         → skipped (no eval): {extract_err[:120]}", flush=True)
        else:
            print(
                f"         → {agg['earned_points']}/{agg['possible_points']} weighted pts ({pct_s})",
                flush=True,
            )

    index_path = work / "batch_index.json"
    index_path.write_text(
        json.dumps(
            {
                "csv": str(csv_path),
                "rkt_json": str(rkt_path),
                "phase": "full",
                "extract_manifest": str(manifest_path.relative_to(work)),
                "frame_summaries_all": str(all_summaries_path.relative_to(work)),
                "students": roster,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] Wrote {index_path}", flush=True)


if __name__ == "__main__":
    main()
