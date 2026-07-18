# AIDE — Automated Instructional Video Evaluation

AIDE scores student video submissions against a weighted rubric using LLM-based evaluation. Preprocessing turns each YouTube link into a transcript and frame summaries; evaluation scores every rubric criterion and aggregates category points.

## Quick start

1. **Install dependencies** — From the `aide` directory, run `python -m pip install -r requirements.txt` (see [Requirements](#requirements)).
2. **Credentials** — Create `aide/.env` with your OpenAI API key (see [Environment](#environment)).
3. **Inputs** — Put each student's submitted files under `in/<alias>/` (see [Input directory](#input-directory)).
4. **Rubric** — Use a hand-authored JSON rubric under `rubrics/` (see [Rubrics](#rubrics-manual-today)).
5. **Run evaluation** — From the `aide` directory:

```bash
python evaluate_class.py
```

6. **Analyze results** — Open `analysis.ipynb` and run cells to compare AIDE scores to instructor `true_score` values.

## Entry point: `evaluate_class.py`

[`evaluate_class.py`](evaluate_class.py) is the main entry for batch-evaluating an entire class.

For each student directory under `in/` it:

1. Uses the directory name as the student **alias**.
2. **Preprocesses** every supported spreadsheet and video file in that directory → `out/preprocess/<alias>/<artifact-name>/`.
3. **Evaluates** the submission against the rubric → per-student JSON in `out/results/<alias>.json` and a running `out/results/class_results.csv` (`alias`, `score`).

Errors on one student are logged and skipped so the rest of the class can finish. Progress is checkpointed by rewriting `class_results.csv` after each successful student.

Configure paths and model in the `if __name__ == "__main__"` block at the bottom of the file (`in_dir`, rubric JSON, `preprocess_dir`, `output_dir`, `model`, and spreadsheet `sheet_name`).

Core pipeline modules:

| Step | Module |
|------|--------|
| Preprocess | `src.eval.preprocess.preprocess_artifacts.preprocess_artifacts` |
| Score | `src.eval.eval.eval_submission` |

Category weights in the rubric are treated as **percentage points** (e.g. weight `25` = 25% of the total); each category gets a fraction in `[0, 1]` from leaf verdicts, then `category_points = score × weight`.

## Analysis: `analysis.ipynb`

[`analysis.ipynb`](analysis.ipynb) compares **AIDE scores** (`out/results/class_results.csv`) to **instructor true scores** from the reference CSV configured in the notebook.

Typical workflow in the notebook:

- Merge on student `alias` (`email` local-part).
- Summary stats (mean, std) for AIDE vs true scores.
- Side-by-side score distributions.
- Per-student delta (`true_score − score`) and error histograms.

Run the notebook from the repo root or adjust `AIDE_ROOT = Path("aide")` if your working directory differs.

## Rubrics (manual today)

Rubric generation is **manual**: authors edit JSON that matches the `Rubric` schema in `src/rubric/rubric_types.py` (categories with `description`, `weight`, `criteria`, and `scoring_instructions`).

Example used by `evaluate_class.py`: `rubrics/gsu-sumprod-manual.json`.

Automated rubric-to-tree tooling (`ratas-rubric.py`, skill trees) is documented in [`RUBRIC_AND_SCORING.md`](RUBRIC_AND_SCORING.md) but is **not** what the class batch runner loads today—the batch path expects a finished manual JSON file.

`rubrics/*.json` is listed in `.gitignore`; keep assignment-specific rubrics local or in a private store if needed.

## Requirements

The pinned Python dependencies are listed in [`requirements.txt`](requirements.txt). Install them with:

```bash
python -m pip install -r requirements.txt
```

The code requires Python 3.10 or newer. Video preprocessing also requires the system `ffmpeg` executable; `ffmpeg-python` in `requirements.txt` is only its Python wrapper. Confirm the executable is available with `ffmpeg -version`.

## Environment

Create **`aide/.env`** (gitignored) with at least:

```env
OPENAI_API_KEY=sk-...
```

Optional overrides used elsewhere in the codebase:

- `OPENAI_EVAL_LEAF_MODEL` / `OPENAI_EVAL_AGENT_MODEL` — leaf and aggregation models
- `OPENAI_VIDEO_FRAME_MODEL` — frame annotation (default `gpt-4o`)

`src/util/openai.py` and several scripts call `load_dotenv()` so running from `aide/` picks up `.env` automatically.

The Python packages, including `yt-dlp`, are installed from `requirements.txt`. The system `ffmpeg` executable must be installed separately.

## Input directory

The `in/` directory contains the raw student submissions. Each immediate child directory represents one student, and its name becomes the alias used throughout preprocessing and evaluation:

```text
in/
├── student1/
│   ├── submission.mp4
│   └── workbook.xlsx
└── student2/
    └── final-submission.xlsm
```

Files may also be organized in nested directories beneath an alias; AIDE searches them recursively. Supported spreadsheet extensions are `.xlsx`, `.xlsm`, `.xltx`, and `.xltm`. Supported video extensions are `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, and `.webm`. Unsupported files are logged and skipped.

Set `sheet_name` in `evaluate_class.py` to the worksheet AIDE should process. The `in/` directory is gitignored because submissions may contain private student data; do not commit its contents.

## Outputs (local, gitignored)

| Path | Contents |
|------|----------|
| `out/preprocess/<alias>/<artifact-name>/` | Preprocessed spreadsheet metadata or video transcript, frames, and frame summaries |
| `out/results/<alias>.json` | Per-category leaf results and aggregation |
| `out/results/class_results.csv` | `alias`, `score` for completed runs |

The `out/` directory is gitignored.

## Related docs

- [`RUBRIC_AND_SCORING.md`](RUBRIC_AND_SCORING.md) — rubric trees, leaf scoring modes, CLI tools
- [`VIDEO_FRAME_PREPROCESS.md`](VIDEO_FRAME_PREPROCESS.md) — frame extraction and vision summarization
