# Aide

Tools for rubric trees, assessment, and media helpers used in grading experiments.

## Video frames: extract then summarize

Two scripts work together: one downloads YouTube submissions and saves **keyframes at sharp visual changes**; the other sends those images to a **vision language model** for detailed text (including on-screen copy and spreadsheet-style content).

### 1. `video_frame_extract.py` — download + scene-based keyframes

**What it does:** Uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to download each video, then [ffmpeg](https://ffmpeg.org/)’s `scene` metric on consecutive decoded frames. When the score crosses `--scene-threshold`, that frame is treated as visually different from the previous one (e.g. webcam → screen share, window switch). The first frame is always kept. `--min-gap` drops extra detections that fall too close in time (debounce).

**Requirements:** `pip install yt-dlp`, `ffmpeg` on your `PATH`.

**Examples:**

```bash
# Single URL — JSON summary printed to stdout
python3 video_frame_extract.py --url 'https://youtu.be/VIDEO_ID' --out ./my_out

# Batch CSV with columns link (or url) and optional email
python3 video_frame_extract.py --csv sample-responses/gsu-student-sumprod-video-list.csv --out ./my_out

# Save JSON for the next step
python3 video_frame_extract.py --url 'https://...' --out ./my_out > frames.json
```

**Useful flags:** `--scene-threshold` (default `0.32`, higher = fewer, stronger cuts), `--min-gap` (default `0.35` seconds), `--keep-video` (keep merged MP4 under `out/<label>/video/`), same cookie / YouTube client options as `audio-transcription/youtube_transcribe.py` (`--cookies`, `--cookies-from-browser`, `--youtube-player-client`, `--no-youtube-player-fallback`).

**Outputs:** Under `--out/<label>/` you get `frames/` JPEGs. The printed JSON array includes `path` and `time_seconds` per frame for downstream tools.

**Programmatic:** Import `preprocess_youtube_to_frames`, `extract_scene_change_frames`, or `download_youtube_video` from `video_frame_extract.py`.

---

### 2. `video_frame_summarize.py` — vision model annotations

**What it does:** Reads each frame image, encodes it as a **base64 data URI** (`data:image/jpeg;base64,...`), and calls the OpenAI Chat Completions API with `image_url` and `detail: high`. The model returns a JSON object with structured fields (overview, verbatim visible text, tabular/sheet content, UI/apps, etc.).

**Requirements:** `pip install openai python-dotenv pydantic`, `OPENAI_API_KEY` (e.g. in `.env`). Optional: `OPENAI_VIDEO_FRAME_MODEL` (defaults to `gpt-4o`). Optional: `pip install pillow` for `--max-long-edge` to shrink huge images before upload.

**Examples:**

```bash
# After saving extract output to frames.json
python3 video_frame_summarize.py --summary-json frames.json --output summaries.json

# Or point at a directory of images (timestamps in the prompt will be unknown)
python3 video_frame_summarize.py --frames-dir ./my_out/submission/frames --output summaries.json

# List jobs without calling the API
python3 video_frame_summarize.py --summary-json frames.json --dry-run
```

**Useful flags:** `--model`, `--sleep` (seconds between API calls), `--max-long-edge`, `--label-filter` (substring on `label` when using `--summary-json`).

**Output:** JSON array of objects with `label`, `frame_path`, `time_seconds`, `model`, and `annotation` (the structured fields).

---

### End-to-end

```bash
python3 video_frame_extract.py --url 'https://www.youtube.com/watch?v=...' --out ./video_out > frames.json
python3 video_frame_summarize.py --summary-json frames.json --output summaries.json
```

---

## Related

- **YouTube audio only:** `audio-transcription/youtube_transcribe.py` — see [audio-transcription/README.md](audio-transcription/README.md).
- **Metadata without download:** `video_metadata.py` (`probe_youtube_metadata`, `probe_local_media_metadata`).
- **Rubric and scoring pipeline:** [RUBRIC_AND_SCORING.md](RUBRIC_AND_SCORING.md).
