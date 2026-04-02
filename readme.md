# Gemini Image Detection Workflow

![cover](example/cover.png)

This repository is a small Gemini-based labeling workflow:

- `gemini.py` runs object detection over a directory of images and saves:
  - one JSON file per image
  - one annotated preview image per image
- `converter.py` turns Gemini detections into Label Studio task JSON
- `check_gemini_api.py` verifies that your Gemini API key is configured and working

The repository is still intentionally lightweight, but the scripts now share a
clearer CLI surface and keep generated artifacts out of version control.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `gemini.py` | Batch Gemini inference over images in a folder |
| `converter.py` | Export Gemini detections into Label Studio task JSON |
| `check_gemini_api.py` | Quick SDK and API-key preflight |
| `requirements.txt` | Minimal runtime dependencies |
| `test_image/` | Sample input images |
| `example/` | Screenshots and visual examples for the README |
| `output_results/` | Generated JSON and annotated previews, created on demand |
| `run.ipynb` | Notebook walkthrough for ad hoc experimentation |

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set your API key with either environment variable:

```bash
export GOOGLE_API_KEY="AIza...your_key..."
# or
export GEMINI_API_KEY="AIza...your_key..."
```

Check the environment before running batch inference:

```bash
python3 check_gemini_api.py
```

If you only want to verify that the key exists locally:

```bash
python3 check_gemini_api.py --skip-api-call
```

## Run Batch Detection

The default workflow reads from `test_image/` and writes to `output_results/`:

```bash
python3 gemini.py
```

Useful overrides:

```bash
python3 gemini.py \
  --input-dir test_image \
  --output-dir output_results \
  --model gemini-3-flash-preview \
  --delay 1.0
```

The default prompt is tuned for fish detection. You can swap the prompt at the
command line or through `GEMINI_PROMPT` when you want to reuse the script for a
different object class.

## Export to Label Studio

Generate a local Label Studio import file:

```bash
python3 converter.py
```

This writes `tasks.json` with absolute image paths that match the images in
`test_image/`.

If you want a GCS-backed import as well:

```bash
python3 converter.py \
  --gcs-bucket your-bucket-name \
  --gcs-prefix fish_images/
```

That also writes `import_to_ls_gcs.json`.

## Notes

- Generated files such as `output_results/`, `tasks.json`, and
  `import_to_ls_gcs.json` are ignored by Git.
- Raw frame dumps such as `frames_jpg/` are also ignored so the repo root stays
  usable.
- `run.ipynb` is kept for interactive work, but the scripts are the canonical
  entry points for repeatable runs.

## References

- [Gemini API docs](https://ai.google.dev/gemini-api/docs)
- [Label Studio docs](https://labelstud.io/)
