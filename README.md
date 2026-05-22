# AIDE — Automated Instructional Video Evaluation

AIDE scores student video submissions against a weighted rubric using LLM-based evaluation. Preprocessing turns each YouTube link into a transcript and frame summaries; evaluation scores every rubric criterion and aggregates category points.

## Quick start

1. **Credentials** — Create `aide/.env` with your OpenAI API key (see [Environment](#environment)).
2. **Rubric** — Use a hand-authored JSON rubric under `rubrics/` (see [Rubrics](#rubrics-manual-today)).
3. **Class list** — Provide a local CSV of submissions (see [Student assignment list](#student-assignment-list)); this file is **not** committed to the public repo.
4. **Run evaluation** — From the `aide` directory:

```bash
python evaluate_class.py
```

5. **Analyze results** — Open `analysis.ipynb` and run cells to compare AIDE scores to instructor `true_score` values.

## Entry point: `evaluate_class.py`

[`evaluate_class.py`](evaluate_class.py) is the main entry for batch-evaluating an entire class.

For each row in the assignment CSV it:

1. Derives a student **alias** from `email` (part before `@`).
2. **Preprocesses** the video URL: download, transcribe, extract frames, annotate frames → `out/preprocess/<alias>/`.
3. **Evaluates** the submission against the rubric → per-student JSON in `out/results/<alias>.json` and a running `out/results/class_results.csv` (`alias`, `score`).

Errors on one student are logged and skipped so the rest of the class can finish. Progress is checkpointed by rewriting `class_results.csv` after each successful student.

Configure paths and model in the `if __name__ == "__main__"` block at the bottom of the file (CSV path, rubric JSON, `preprocess_dir`, `output_dir`, `model`).

Core pipeline modules:

| Step | Module |
|------|--------|
| Preprocess | `src.eval.preprocess.preprocess.preprocess_video` |
| Score | `src.eval.eval.eval_submission` |

Category weights in the rubric are treated as **percentage points** (e.g. weight `25` = 25% of the total); each category gets a fraction in `[0, 1]` from leaf verdicts, then `category_points = score × weight`.

## Analysis: `analysis.ipynb`

[`analysis.ipynb`](analysis.ipynb) compares **AIDE scores** (`out/results/class_results.csv`) to **instructor true scores** from the same assignment CSV.

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

## Environment

Create **`aide/.env`** (gitignored) with at least:

```env
OPENAI_API_KEY=sk-...
```

Optional overrides used elsewhere in the codebase:

- `OPENAI_EVAL_LEAF_MODEL` / `OPENAI_EVAL_AGENT_MODEL` — leaf and aggregation models
- `OPENAI_VIDEO_FRAME_MODEL` — frame annotation (default `gpt-4o`)

`src/util/openai.py` and several scripts call `load_dotenv()` so running from `aide/` picks up `.env` automatically.

**External tools** for preprocessing: `ffmpeg`, `yt-dlp`, and Python deps (`openai`, `pydantic`, `pandas`, `python-dotenv`, etc.).

## Student assignment list

Submissions are driven by a **CSV** with columns:

| Column | Purpose |
|--------|---------|
| `email` | Student email; alias = local-part before `@` |
| `link` | YouTube (or compatible) video URL |
| `true_score` | Instructor reference score (for analysis only; not used by `evaluate_class.py`) |

Example shape (values illustrative):

```csv
email,link,true_score
student1@university.edu, https://www.youtube.com/watch?v=..., 85
student2@university.edu, https://youtu.be/..., 90
```

**Privacy:** Real class lists with student emails and scores are **not** published in the public repository. `.gitignore` excludes `student-responses/`. Maintain your CSV locally (e.g. `student-responses/gsu-student-sumprod-video-list.csv`) and point `evaluate_class.py` at your copy.

## Outputs (local, gitignored)

| Path | Contents |
|------|----------|
| `out/preprocess/<alias>/` | `transcript.txt`, `frames/`, `frames_summary.json` |
| `out/results/<alias>.json` | Per-category leaf results and aggregation |
| `out/results/class_results.csv` | `alias`, `score` for completed runs |

The `out/` directory is gitignored.

## Related docs

- [`RUBRIC_AND_SCORING.md`](RUBRIC_AND_SCORING.md) — rubric trees, leaf scoring modes, CLI tools
- [`VIDEO_FRAME_PREPROCESS.md`](VIDEO_FRAME_PREPROCESS.md) — frame extraction and vision summarization
