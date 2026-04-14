# Rubrics and scoring (AIDE)

Turn a **weighted rubric** into a **Rubric Knowledge Tree (RKT)** JSON, then score student work **per atomic leaf** using an LLM.

## Prerequisites

- Python packages used by `request.py` and `eval_agent.py` (e.g. `openai`, `instructor`, `python-dotenv`, `pandas` for CSV rubrics; `yt-dlp` only if you use YouTube metadata in the agent).
- API credentials (e.g. `.env` for your OpenAI / instructor setup).
- Run commands from the `aide` directory so imports resolve:

```bash
cd aide
```

---

## 1. Rubric → skill tree JSON (`ratas-rubric.py`)

`ratas-rubric.py`:

1. **Normalizes** the source (CSV, TXT, or normalized rubric JSON).
2. **Extracts skills** (`prompts/skill-extraction.txt`).
3. **Builds the tree** (`prompts/skill-tree-construction.txt`) as `RKTRoot`.
4. **Attaches weights** (`weighted_rubric.attach_weights_from_rows`).
5. **Writes** `rubrics/<basename>.json` unless you pass `-o`.

### CSV rubric (recommended)

One row per top-level criterion: **`Criteria`**, **`Weight`**, plus a body column (often the first column after those, e.g. `Score 4`). See `rubrics/open-ended-response.csv`.

```bash
python3 ratas-rubric.py rubrics/open-ended-response.csv --csv-body-column "Score 4"
```

### Plain-text rubric

Heuristic parsing for files like `rubrics/gsu-sumprod.txt` (point blocks, bullets). Prefer CSV if parsing is fragile.

### Build

```bash
python3 ratas-rubric.py
python3 ratas-rubric.py path/to/rubric.csv -o rubrics/out.json
```

The script prints an ASCII tree via `tree_viz.render_skill_tree`.

---

## 2. Scoring: two modes

Both modes use the **same** RKT JSON and the same **leaf list** (`flatten_rubric_leaves` in `request.py`). Verdicts are always `met` | `not_met` | `undetermined`, with optional **evidence**.

| Mode | Entry point | Behavior |
|------|-------------|----------|
| **Batch (one LLM call)** | `assess_leaves.py` / `assess_response_leaves` | Sends all leaves + full response text in one request (`prompts/leaf-assessment.txt`). Cheaper and faster. |
| **Per-leaf agent** | `eval_agent.py` | Loads the tree once; for **each leaf**, a short tool loop: `read_submission`, `search_submission`, optional `get_video_metadata` (local file via ffprobe or YouTube via yt-dlp, no download), then `submit_leaf_verdict`. Better when the model needs to search or read long submissions in chunks, or use video duration metadata. |

Neither mode computes a final course grade by itself; combine verdicts with weights in your own policy or spreadsheet.

### Batch CLI (`assess_leaves.py`)

```bash
python3 assess_leaves.py rubrics/gsu-sumprod.json sample-responses/gsu-student-response.txt -o assessments/out.json
python3 assess_leaves.py TREE.json RESPONSE.txt -o out.json --quiet
```

### Per-leaf agent CLI (`eval_agent.py`)

Requires a **pre-built** RKT JSON (from step 1). Optional video metadata:

- **`--video PATH`** — local file; metadata via **ffprobe** (ffmpeg).
- **`--youtube-url URL`** — metadata only via **yt-dlp** (`pip install yt-dlp`); optional **`--cookies`** / **`--cookies-from-browser`** for unlisted or login-gated videos (same idea as `audio-transcription/youtube_transcribe.py`).

```bash
python3 eval_agent.py rubrics/gsu-sumprod.json sample-responses/gsu-student-response.txt -o assessments/agent.json
python3 eval_agent.py rubrics/gsu-sumprod.json transcript.txt --youtube-url 'https://www.youtube.com/watch?v=...' -o out.json
```

Programmatic tree build without the agent: `eval_agent.materialize_rubric_tree` (same pipeline as `ratas-rubric.py`).

### Programmatic batch scoring

```python
from rkt_io import load_skill_tree
from request import assess_response_leaves

rubric = load_skill_tree("rubrics/gsu-sumprod.json")
text = open("sample-responses/gsu-student-response.txt", encoding="utf-8").read()
merged, batch = assess_response_leaves(rubric, text)
```

`merged` matches the JSON written by `assess_leaves.py -o`.

---

## 3. Reference

| Piece | Role |
|-------|------|
| `prompts/skill-extraction.txt` | Rubric text → flat skill list. |
| `prompts/skill-tree-construction.txt` | Skills + ordered categories → `RKTRoot`. |
| `prompts/leaf-assessment.txt` | All leaves + response → `LeafAssessmentBatch` (batch mode). |
| `type.py` | `RKTRoot`, rules, `LeafAssessmentBatch`. |
| `weighted_rubric.py` | CSV/TXT → weighted rows; formatting for extraction. |
| `rubric_normalize.py` | Canonical normalized rubric JSON. |
| `video_metadata.py` | ffprobe (local) / yt-dlp (URL) for `get_video_metadata`. |

---

## 4. Checklist

1. Author rubric (CSV or TXT).
2. `python3 ratas-rubric.py your-rubric.csv -o rubrics/your-rubric.json`
3. Save student text (transcript, essay, LMS export) as `.txt`.
4. Either `python3 assess_leaves.py rubrics/your-rubric.json student.txt -o …` **or** `python3 eval_agent.py rubrics/your-rubric.json student.txt -o …` (add `--video PATH` or r URL` on the eval_agent command when you need metadata).
5. Aggregate points using your grading policy.
