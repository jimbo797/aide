#!/usr/bin/env python3
"""
Assess a student response against each leaf of a saved rubric skill tree (JSON).

Each leaf gets verdict: met | not_met | undetermined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from request import assess_response_leaves
from rkt_io import load_skill_tree


def main() -> None:
    aide_dir = Path(__file__).resolve().parent
    default_tree = aide_dir / "rubrics" / "gsu-sumprod.json"
    default_response = aide_dir / "sample-responses" / "gsu-student-response.txt"

    parser = argparse.ArgumentParser(
        description="Assess a student response against each leaf in an RKT JSON rubric."
    )
    parser.add_argument(
        "tree",
        nargs="?",
        default=str(default_tree),
        type=str,
        help="Path to skill tree JSON (from ratas-rubric.py)",
    )
    parser.add_argument(
        "response",
        nargs="?",
        default=str(default_response),
        type=str,
        help="Path to student response text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Write merged assessments as JSON to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print table to stdout (still writes -o if set)",
    )
    args = parser.parse_args()

    tree_path = Path(args.tree)
    if not tree_path.is_file():
        print(f"Tree not found: {tree_path}", file=sys.stderr)
        sys.exit(1)
    resp_path = Path(args.response)
    if not resp_path.is_file():
        print(f"Response file not found: {resp_path}", file=sys.stderr)
        sys.exit(1)

    rubric = load_skill_tree(tree_path)
    response_text = resp_path.read_text(encoding="utf-8")
    merged, _batch = assess_response_leaves(rubric, response_text)

    if not args.quiet:
        for row in merged:
            vid = row["verdict"]
            mark = {"met": "✓", "not_met": "✗", "undetermined": "?"}[vid]
            crit = (row["criterion"] or "")[:72]
            if len(row.get("criterion") or "") > 72:
                crit += "..."
            cat = (row["category"] or "")[:40]
            print(f"{mark} [{vid:14}] L{row['leaf_id']:2} | {cat} | {crit}")
            if row.get("evidence"):
                print(f"         evidence: {row['evidence'][:120]}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nWrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
