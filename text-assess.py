"""
Assess a text response using the decomposed rubric (RKT skill tree).

Usage:
  python text-assess.py [response.txt] [--rubric path/to/rubric.json] [--out results.json]

  - response.txt: path to file containing the student response (default: stdin)
  - --rubric: path to saved skill tree JSON (default: rubrics/open-ended-response.json)
  - --out: optional path to write assessment JSON
"""

import argparse
import json
import os
import sys

from rkt_io import load_skill_tree
from request import assess_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess a text response against the rubric.")
    parser.add_argument(
        "response_file",
        nargs="?",
        default=None,
        help="Path to file containing the student response (default: read from stdin)",
    )
    parser.add_argument(
        "--rubric",
        default=None,
        help="Path to rubric JSON (default: aide/rubrics/open-ended-response.json)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write assessment JSON",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rubric_path = args.rubric or os.path.join(script_dir, "rubrics", "open-ended-response.json")

    if args.response_file:
        with open(args.response_file, encoding="utf-8") as f:
            response_text = f.read()
    else:
        response_text = sys.stdin.read()

    if not response_text.strip():
        print("No response text to assess.", file=sys.stderr)
        sys.exit(1)

    rubric = load_skill_tree(rubric_path)
    assessment = assess_response(rubric, response_text)

    # Print summary to stdout
    for cat in assessment.categories:
        print(f"\n{cat.category_name}")
        print("-" * 40)
        for c in cat.criteria:
            status = "✓" if c.met else "✗"
            print(f"  {status} {c.description}")
            if c.evidence:
                print(f"      Evidence: {c.evidence}")

    # Optionally write JSON
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(assessment.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        print(f"\nWrote assessment to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
