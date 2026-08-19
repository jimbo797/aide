# Amending AIDE scores

After an evaluation run, you can override automated judgments by editing each student's `results.json`. Then run the amendments script to recompute category points and class totals.

Amendments are **human edits in the JSON**. Do not change `verdict`, `aggregation.score`, or `category_points` by hand; add an `amendment` field instead and let the script apply it.

## Where to edit

Results live at:

```text
out/results/<alias>/results.json
```

Each file is a list of categories. A category looks like:

```json
{
  "category": "Video Length",
  "weight": 3.0,
  "leaf_results": [ { "criterion": "...", "verdict": "not_met", "...": "..." } ],
  "aggregation": { "score": 0.0, "reasoning": "...", "weight": 3.0 },
  "category_points": 0.0
}
```

## 1. Criterion amendment

Use this when the automated **leaf verdict** is wrong, but you still want the category's **scoring instructions** to decide points.

Add `"amendment"` to the relevant object in `leaf_results`. Allowed values are `"met"` or `"not_met"`:

```json
{
  "criterion": "Video length is approximately 3 minutes and does not exceed 4 minutes.",
  "verdict": "not_met",
  "amendment": "met",
  "evidence": "..."
}
```

When you run the script, it re-runs the category scoring-instructions step **only for categories that contain at least one amended criterion**. Other categories are left alone. The original automated `verdict` is kept; the amendment is what scoring uses.

## 2. Category amendment

Use this when you want to set the **points earned for the whole category**, regardless of leaf verdicts.

Add `"amendment"` to the `aggregation` object. The value is the replacement **point total** (not a 0–1 fraction):

```json
{
  "aggregation": {
    "category": "Video Length",
    "weight": 3.0,
    "score": 0.0,
    "amendment": 3.0,
    "reasoning": "..."
  }
}
```

The script copies that number into `category_points`. It does not re-run scoring instructions for that category.

If a category has **both** leaf amendments and an aggregation amendment, the **category amendment wins** (the point value you set is used; scoring is not re-run).

## Apply amendments

From the `aide` directory:

```bash
python analysis/amendments.py
```

Useful flags:

```bash
python analysis/amendments.py --results-dir old-runs/summer9/results --rubric rubrics/gsu-summer-forecast.json
python analysis/amendments.py --dry-run
python analysis/amendments.py --model gpt-5.6-terra
```

`--dry-run` prints the new scores without writing files. Criterion amendments that need a re-aggregation still call the model unless the category also has an aggregation amendment.

The script:

1. Reads every `out/results/<alias>/results.json`.
2. For criterion-only amendments, re-aggregates that category with the rubric scoring instructions (requires `OPENAI_API_KEY`).
3. For category amendments, uses `aggregation.amendment` as `category_points`.
4. Sums `category_points` for the student score.
5. Writes updated JSON (original points are stored as `original_category_points`; original aggregation as `original_score` / `original_reasoning` when scoring is re-run).
6. Rewrites `out/results/class_results.csv`.

Re-run the script after you change amendments. `original_*` fields are written only once, so later runs still compare against the first automated score.

## Example

`old-runs/summer9/results/jgrandchamps1/results.json` amends Video Length both ways: the leaf is marked `"amendment": "met"` and `aggregation.amendment` is `3.0`. Because the category amendment is present, the applied score for that category is **3.0** points.
