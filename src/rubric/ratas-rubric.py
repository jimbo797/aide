"""
Build a weighted skill tree (RKT JSON) from a rubric.

All inputs are converted to a normalized rubric (see rubric_normalize.py) before skill
extraction. You can author rubrics directly as normalized JSON, or use CSV / TXT and
optionally emit a normalized copy with --write-normalized for editing and reuse.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from request import rubric_skill_extract, rubric_skill_tree_construct
from rubric_normalize import (
    InputFormat,
    normalize_rubric_file,
    save_normalized_rubric,
)
from rkt_io import save_skill_tree
from tree_viz import render_skill_tree
from weighted_rubric import (
    attach_weights_from_rows,
    format_categories_for_tree,
    format_for_skill_extract,
    merge_consecutive_same_weight,
    normalize_rkt_positive_weights,
)


def main() -> None:
    aide_dir = os.path.dirname(os.path.abspath(__file__))
    default_rubric = os.path.join(aide_dir, "rubrics", "open-ended-response.csv")

    parser = argparse.ArgumentParser(
        description="Extract skills and build an RKT skill tree from a rubric (CSV, TXT, or normalized JSON)."
    )
    parser.add_argument(
        "rubric",
        nargs="?",
        default=default_rubric,
        help="Path to .csv, .txt/.md, or normalized rubric .json (see rubric_normalize.py)",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "txt", "normalized"),
        default="auto",
        help="Input format (default: infer from contents/extension)",
    )
    parser.add_argument(
        "--csv-body-column",
        default=None,
        metavar="COL",
        help="CSV only: column for criterion body text (default: first column after name and weight)",
    )
    parser.add_argument(
        "--csv-name-column",
        default="Criteria",
        help="CSV only: column for category title (default: Criteria)",
    )
    parser.add_argument(
        "--csv-weight-column",
        default="Weight",
        help="CSV only: column for weight/points (default: Weight)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="Output RKT skill tree JSON (default: rubrics/<input-basename>.json under aide/)",
    )
    parser.add_argument(
        "--write-normalized",
        default=None,
        metavar="PATH",
        help="Also write the normalized rubric JSON (canonical categories) to this path",
    )
    parser.add_argument(
        "--no-merge-weights",
        action="store_true",
        help="Do not overwrite rubric-line titles/weights from the normalized rubric after LLM build",
    )
    parser.add_argument(
        "--merge-consecutive-same-weight",
        action="store_true",
        help="Merge adjacent categories with the same point value before extraction (weights sum)",
    )
    parser.add_argument(
        "--no-sync-descriptions",
        action="store_true",
        help="Do not replace rubric line descriptions with source titles when counts match",
    )
    args = parser.parse_args()

    rubric_path = Path(args.rubric)
    if not rubric_path.is_file():
        raise SystemExit(f"Rubric file not found: {rubric_path}")

    normalized = normalize_rubric_file(
        rubric_path,
        format=args.format,
        merge_consecutive_same_weight=args.merge_consecutive_same_weight,
        csv_body_column=args.csv_body_column,
        csv_name_column=args.csv_name_column,
        csv_weight_column=args.csv_weight_column,
    )
    rows = normalized.to_weighted_criteria()
    if not rows:
        raise SystemExit(f"No categories in normalized rubric from {rubric_path}")

    if args.write_normalized:
        save_normalized_rubric(args.write_normalized, normalized)

    rubric_str = format_for_skill_extract(rows)
    categories_block = format_categories_for_tree(rows)

    start_time = datetime.now()
    skills = rubric_skill_extract(rubric_str)
    print(skills)
    print("Elapsed (skill extract):", datetime.now() - start_time)

    start_time = datetime.now()
    skill_tree = rubric_skill_tree_construct(skills, categories_block=categories_block)
    if not args.no_merge_weights:
        skill_tree = attach_weights_from_rows(
            skill_tree,
            rows,
            sync_descriptions=not args.no_sync_descriptions,
        )
    skill_tree = normalize_rkt_positive_weights(skill_tree)
    print(render_skill_tree(skill_tree))
    print("Elapsed (skill tree):", datetime.now() - start_time)

    out_path = args.output
    if not out_path:
        out_path = os.path.join(aide_dir, "rubrics", f"{rubric_path.stem}.json")
    print("Saving skill tree to file...", out_path)
    save_skill_tree(skill_tree, out_path)
    if args.write_normalized:
        print("Wrote normalized rubric to", args.write_normalized)


if __name__ == "__main__":
    main()
