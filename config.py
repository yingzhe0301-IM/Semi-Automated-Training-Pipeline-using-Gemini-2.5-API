"""Shared repository defaults.

Environment-variable overrides stay in the scripts, but all repo-level default
values live here so they only need to be updated in one place.
"""

from pathlib import Path

DEFAULT_INPUT_DIR = Path("test_image")
DEFAULT_OUTPUT_DIR = Path("output_results")

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_LABEL_STUDIO_MODEL_VERSION = DEFAULT_MODEL

DEFAULT_PROMPT = (
    "Detect the fish in the image. The response must be a JSON list of objects. "
    "Each object must contain a 'label' key with the text 'fish' and a 'box_2d' "
    "key with the bounding box coordinates as [ymin, xmin, ymax, xmax], "
    "normalized to a 0-1000 scale. If no fish are detected, return an empty "
    "list []."
)
DEFAULT_CHECK_PROMPT = "Reply with exactly the word OK."

DEFAULT_DELAY_SECONDS = 1.0
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DEFAULT_LOCAL_OUTPUT = Path("tasks.json")
DEFAULT_GCS_OUTPUT = Path("import_to_ls_gcs.json")

BOX_COLOR = "#4FC3F7"
TEXT_COLOR = "#E1F5FE"
