# YouTube transcription

`youtube_transcribe.py` downloads a YouTube video’s audio with [yt-dlp](https://github.com/yt-dlp/yt-dlp), converts it to MP3 (via ffmpeg), then transcribes it with the OpenAI Audio API using the same model as `main.py` (`gpt-4o-transcribe`). The transcript is printed to stdout.

## Requirements

- **Python 3.10+** (uses `from __future__ import annotations` and `str | None` style types)
- **ffmpeg** on your PATH (yt-dlp uses it to extract MP3). Install with your OS package manager (e.g. `brew install ffmpeg` on macOS).
- **OpenAI API key** in the environment. The script loads `.env` from the current working directory via `python-dotenv`, so you can put `OPENAI_API_KEY=...` in a `.env` file next to where you run the command, or export the variable in your shell.

Install Python dependencies:

```bash
pip install yt-dlp openai python-dotenv
```

Keep yt-dlp up to date; YouTube changes often break older versions:

```bash
pip install -U yt-dlp
```

## Basic usage

From this directory (or anywhere, using the full path to the script):

```bash
python youtube_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The full transcript is printed to the terminal. Redirect to a file if you want:

```bash
python youtube_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID" > transcript.txt
```

## Options

| Flag | Purpose |
|------|--------|
| `--keep-audio PATH` | After transcribing, copy the downloaded MP3 to `PATH` (parent directories are created if needed). |
| `--cookies FILE` | Netscape-format `cookies.txt` (e.g. from a browser extension such as “Get cookies.txt”). Use when the video plays in the browser but yt-dlp says the video is not available. |
| `--cookies-from-browser BROWSER` | Load cookies from a local browser profile. Examples: `chrome`, `firefox`, `safari`, `brave`. |
| `--browser-profile NAME` | Use with `--cookies-from-browser` to pick a profile (e.g. `Default`). |
| `--youtube-player-client NAME` | Force a single YouTube “player client” (e.g. `web_embedded`, `web`). Implies no automatic client fallback. |
| `--no-youtube-player-fallback` | Only yt-dlp’s default YouTube client; disables automatic retries with other clients. |

Do not pass both `--cookies` and `--cookies-from-browser`. Do not combine `--youtube-player-client` with `--no-youtube-player-fallback` (the script will error; forcing one client already disables fallback).

## Behavior notes

- **Unlisted videos** work like in a browser: anyone with the link can fetch them; no extra flag is required for “unlisted” itself.
- **Player client fallback**: By default the script tries the default client, then `android`, `web_embedded`, and `web`. That helps when one client returns “not available” while others still work.
- **JavaScript / “n” challenge**: If you see warnings about no JavaScript runtime or “n challenge solving failed”, web clients may only expose thumbnails until you add a supported JS runtime (see [yt-dlp EJS wiki](https://github.com/yt-dlp/yt-dlp/wiki/EJS)). The `android` client often still serves a combined audio+video stream; ffmpeg then keeps the MP3.
- **Temporary files**: Audio is downloaded under a temporary directory and removed after the run unless you use `--keep-audio`.

## Related script

`main.py` transcribes a local file (`test-audio.mp3` in the same folder) with the same OpenAI model; it does not download from YouTube.
