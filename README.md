# AIDE — Automated Instructional Video Evaluation

AIDE scores student submissions against a weighted rubric using LLM-based evaluation. Preprocessing turns each student's video, spreadsheet, and text files into transcripts, frame summaries, and structured Excel metadata. Evaluation then judges every rubric criterion with tool-backed evidence, aggregates those verdicts into category points, and writes class-level scores and costs.

This project was a thesis project for MIT Media Lab, Personal Robotics Group.

## Quick start

1. **Install dependencies** — From the `aide` directory, run `python -m pip install -r requirements.txt` (see [Requirements](#requirements)).
2. **Credentials** — Create `aide/.env` with your OpenAI API key (see [Environment](#environment)).
3. **Inputs** — Put each student's submitted files under `in/<alias>/` (see [Input directory](#input-directory)).
4. **Rubric** — Use a hand-authored JSON rubric under `rubrics/` (see [Rubrics](#rubrics)).
5. **Run evaluation** — Edit the `if __name__ == "__main__"` block in [`evaluate_class.py`](evaluate_class.py) (rubric path, `in_dir`, output dirs, model), then from the `aide` directory:

```bash
python evaluate_class.py
```

6. **Review and amend** — Use the tools in [`analysis/`](analysis/) for point-loss reports, TA score amendments, and rubric leniency suggestions (see [After a run](#after-a-run)).

## Entry point: `evaluate_class.py`

[`evaluate_class.py`](evaluate_class.py) is the main entry for batch-evaluating an entire class.

For each student directory under `in_dir` it:

1. Uses the directory name as the student **alias**.
2. **Preprocesses** every supported spreadsheet, video, and text file in that directory → `preprocess_dir/<alias>/<artifact-name>/`.
3. **Evaluates** the submission against the rubric → `output_dir/<alias>/results.json`, a running `output_dir/class_results.csv` (`alias`, `score`), per-student `output_dir/<alias>/costs.json`, and `output_dir/class_costs.json`.

Errors on one student are logged; the rest of the class still runs. Progress is checkpointed by rewriting `class_results.csv` and `class_costs.json` after each successful student.

Configure paths and model in the `if __name__ == "__main__"` block (`in_dir`, rubric JSON, `preprocess_dir`, `output_dir`, `model`). Optional knobs:

| Argument | Default | Role |
|----------|---------|------|
| `sheet_name` | `None` | Excel worksheet to process; omit to use the workbook's default handling |
| `scene_threshold` | `0.1` | Scene-change threshold for video keyframes (`None` disables scene detection) |
| `sample_interval_seconds` | `30.0` | Extra periodic frame samples (`None` disables interval sampling) |
| `max_loop_iters` | `None` (leaf default is 3) | Evidence-gathering rounds per criterion |

Core pipeline modules:

| Step | Module |
|------|--------|
| Preprocess | `src.eval.preprocess.preprocess_artifacts.preprocess_artifacts` |
| Score | `src.eval.eval.eval_submission` |

Category weights in the rubric are **percentage points** (e.g. weight `25` = 25% of the total). Each category gets a fraction in `[0, 1]` from leaf verdicts, then `category_points = score × weight`. The student score is the sum of category points.

## How evaluation works

Each **category** is the smallest unit that receives a numeric score. **Criteria** (leaves) are marked `met`, `not_met`, or `undetermined` and are used only to drive that category score.

For every criterion, a leaf agent:

1. **Plans** which tools to call (transcript, frames, Excel, text, or assignment `sources/`).
2. **Gathers evidence** with those tools.
3. **Thinks** about whether the evidence is enough to judge, looping until it is or `max_loop_iters` is reached.
4. **Verdicts** the criterion and records evidence plus reasoning.

Leaf evaluations run concurrently (`asyncio`, default 4 at a time). After all leaves in a category finish, a separate model call applies the category's `scoring_instructions` to produce the `[0, 1]` category score.

Tools available to the leaf agent (only those with matching preprocess artifacts are offered):

- Video: `read_transcript`, `search_transcript`, `list_frame_summaries`, `read_frame_summaries`, `read_metadata`
- Spreadsheet: `read_excel` (cells, formulas, charts)
- Text: `read_txt`
- Assignment reference files under `<alias>/sources/`: `list_sources`, `read_source` (context only — not treated as student work)

## Rubrics

Rubrics are **hand-authored JSON** matching the `Rubric` schema in `src/rubric/rubric_types.py`: a list of categories, each with `description`, `weight`, atomic `criteria`, and `scoring_instructions` for turning criterion verdicts into a category score.

Example used by `evaluate_class.py`: `rubrics/gsu-summer-forecast.json`.

Automatic rubric-tree construction is not wired into the batch runner (see `todo.md`). `rubrics/` is listed in `.gitignore`; keep assignment-specific rubrics local or in a private store if needed.

## After a run

Score comparison against instructor grades lives in notebooks outside this package (`experiments/` and `gsu-materials/`). Inside `aide`, post-run tools are:

| Script | Purpose |
|--------|---------|
| [`analysis/point_losses.py`](analysis/point_losses.py) | Writes `output_dir/<alias>/report.txt`: points lost per category and a short reason |
| [`analysis/amendments.py`](analysis/amendments.py) | Recomputes scores after TAs add `amendment` fields in `results.json` |
| [`analysis/rubric_relaxation.py`](analysis/rubric_relaxation.py) | Suggests rubric wording / scoring-instruction tweaks from class fail rates |

Point-loss reports:

```bash
python analysis/point_losses.py
```

TA amendments (edit `results.json` first; see [`AMENDMENTS.md`](AMENDMENTS.md)):

```bash
python analysis/amendments.py
python analysis/amendments.py --results-dir out/results --rubric rubrics/assignment1.json
python analysis/amendments.py --dry-run
```

You can override a single criterion (`leaf_results[].amendment`: `"met"` or `"not_met"`) or a whole category (`aggregation.amendment`: replacement points). Do not edit `verdict`, `aggregation.score`, or `category_points` by hand.

## Requirements

Pinned Python dependencies are in [`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

The code requires Python 3.10 or newer. Video preprocessing also requires the system `ffmpeg` executable; `ffmpeg-python` is only its Python wrapper. Confirm the executable with `ffmpeg -version`.

## Environment

Create **`aide/.env`** (gitignored) with at least:

```env
OPENAI_API_KEY=sk-...
```

Optional overrides:

- `OPENAI_EVAL_LEAF_MODEL` / `OPENAI_EVAL_AGENT_MODEL` — leaf and aggregation models (default `gpt-4o-mini` if unset; `evaluate_class.py` typically passes an explicit `model`)
- `OPENAI_EVAL_LEAF_MAX_ITERS` — evidence-gathering rounds per leaf (default `3`)
- `OPENAI_VIDEO_FRAME_MODEL` — frame annotation (default `gpt-4o`)
- `OPENAI_VIDEO_FRAME_MAX_TOKENS` — frame-annotation completion cap (default `16384`; `0` omits the cap)

`src/util/openai.py` and several scripts call `load_dotenv()` so running from `aide/` picks up `.env` automatically.

Token costs use rates in [`model_pricing.json`](model_pricing.json) (USD per 1M tokens, keyed by model name).

## Input directory

The `in/` directory contains the raw student submissions. Each immediate child directory represents one student, and its name becomes the alias used throughout preprocessing and evaluation:

```text
in/
├── student1/
│   ├── submission.mp4
│   ├── workbook.xlsx
│   └── sources/          # optional assignment reference files (not student work)
│       └── prompt.txt
└── student2/
    └── final-submission.xlsm
```

Files may be nested under an alias; AIDE searches them recursively. A `sources/` folder at the top of a student directory is skipped during preprocessing and is only readable via `list_sources` / `read_source`.

Supported extensions:

- Spreadsheets: `.xlsx`, `.xlsm`, `.xltx`, `.xltm`
- Video: `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`
- Text: `.txt`

Unsupported files are logged and skipped. `in/` is gitignored; do not commit student data.

There is still a YouTube download helper (`src/eval/preprocess/video/download_yt_video.py` / `preprocess_video`) for CSV link lists, but the class batch runner expects **local files** under each alias.

## Outputs (local, gitignored)

| Path | Contents |
|------|----------|
| `out/preprocess/<alias>/<artifact-name>/` | Video: `transcript.txt`, `frames/`, `frames_summary.json`, `metadata.json`. Spreadsheet: `excel.json`. Text: `content.txt` |
| `out/results/<alias>/results.json` | Per-category leaf results (verdicts, evidence, tool traces) and aggregation |
| `out/results/<alias>/costs.json` | Tokens, API calls, USD cost, and wall time for that student |
| `out/results/<alias>/report.txt` | Point-loss summary (after `analysis/point_losses.py`) |
| `out/results/class_results.csv` | `alias`, `score` for completed runs |
| `out/results/class_costs.json` | Aggregated token usage and cost for the class |

The `out/` directory is gitignored. Historical runs are kept under `old-runs/` (also gitignored).

## Related docs

- [`AMENDMENTS.md`](AMENDMENTS.md) — how TAs override leaf verdicts or category points
- [`todo.md`](todo.md) — remaining work and improvement ideas
