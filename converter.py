"""Convert Gemini detections into Label Studio task JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_GCS_OUTPUT,
    DEFAULT_INPUT_DIR,
    DEFAULT_LABEL_STUDIO_MODEL_VERSION,
    DEFAULT_LOCAL_OUTPUT,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
)

DEFAULT_JSON_DIR = DEFAULT_OUTPUT_DIR
DEFAULT_IMAGE_DIR = DEFAULT_INPUT_DIR
DEFAULT_MODEL_VERSION = os.getenv(
    "LABEL_STUDIO_MODEL_VERSION", DEFAULT_LABEL_STUDIO_MODEL_VERSION
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Label Studio task files from Gemini detection JSON."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"Directory containing Gemini JSON outputs. Default: {DEFAULT_JSON_DIR}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing original images. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=DEFAULT_LOCAL_OUTPUT,
        help=f"Output file for local-image tasks. Default: {DEFAULT_LOCAL_OUTPUT}",
    )
    parser.add_argument(
        "--gcs-output",
        type=Path,
        default=DEFAULT_GCS_OUTPUT,
        help=f"Output file for GCS-backed tasks. Default: {DEFAULT_GCS_OUTPUT}",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=os.getenv("LABEL_STUDIO_GCS_BUCKET"),
        help="Optional GCS bucket name for cloud-backed imports.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=os.getenv("LABEL_STUDIO_GCS_PREFIX", ""),
        help="Optional GCS prefix, such as 'fish_images/'.",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"Model version written into predictions. Default: {DEFAULT_MODEL_VERSION}",
    )
    parser.add_argument(
        "--skip-local-output",
        action="store_true",
        help="Do not generate a local tasks.json file.",
    )
    return parser.parse_args()


def normalize_gcs_prefix(prefix: str) -> str:
    cleaned = prefix.strip("/")
    return f"{cleaned}/" if cleaned else ""


def load_detections(json_path: Path) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  - Skipping {json_path.name}: invalid JSON ({exc}).")
        return None

    if not isinstance(payload, list):
        print(f"  - Skipping {json_path.name}: expected a JSON list.")
        return None
    return [item for item in payload if isinstance(item, dict)]


def build_ls_results(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for detection in detections:
        box_coords = detection.get("box_2d")
        label = detection.get("label")
        if not (
            isinstance(box_coords, list)
            and len(box_coords) == 4
            and all(isinstance(value, (int, float)) for value in box_coords)
            and isinstance(label, str)
            and label
        ):
            continue

        y1_1000, x1_1000, y2_1000, x2_1000 = box_coords
        results.append(
            {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": (x1_1000 / 1000) * 100,
                    "y": (y1_1000 / 1000) * 100,
                    "width": ((x2_1000 - x1_1000) / 1000) * 100,
                    "height": ((y2_1000 - y1_1000) / 1000) * 100,
                    "rotation": 0,
                    "rectanglelabels": [label],
                },
            }
        )
    return results


def find_original_image(image_dir: Path, stem: str) -> Path | None:
    for extension in SUPPORTED_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def build_task(image_reference: str, results: list[dict[str, Any]], model_version: str) -> dict[str, Any]:
    return {
        "data": {"image": image_reference},
        "predictions": [{"model_version": model_version, "result": results}],
    }


def write_json(output_path: Path, payload: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_to_label_studio_format(args: argparse.Namespace) -> int:
    if not args.json_dir.is_dir():
        print(f"Error: JSON directory '{args.json_dir}' was not found.")
        return 1
    if not args.image_dir.is_dir():
        print(f"Error: image directory '{args.image_dir}' was not found.")
        return 1

    json_files = sorted(args.json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in '{args.json_dir}'.")
        return 0

    local_tasks: list[dict[str, Any]] = []
    gcs_tasks: list[dict[str, Any]] = []
    gcs_prefix = normalize_gcs_prefix(args.gcs_prefix)

    print(f"Starting conversion from '{args.json_dir}'...")
    for json_path in json_files:
        detections = load_detections(json_path)
        if detections is None:
            continue

        original_image = find_original_image(args.image_dir, json_path.stem)
        if original_image is None:
            print(f"  - Skipping {json_path.name}: matching source image not found.")
            continue

        results = build_ls_results(detections)
        local_tasks.append(
            build_task(str(original_image.resolve()), results, args.model_version)
        )

        if args.gcs_bucket:
            gcs_reference = f"gs://{args.gcs_bucket}/{gcs_prefix}{original_image.name}"
            gcs_tasks.append(build_task(gcs_reference, results, args.model_version))

    if not args.skip_local_output:
        write_json(args.local_output, local_tasks)
        print(f"  - Wrote {len(local_tasks)} local tasks to '{args.local_output}'.")

    if args.gcs_bucket:
        write_json(args.gcs_output, gcs_tasks)
        print(f"  - Wrote {len(gcs_tasks)} GCS tasks to '{args.gcs_output}'.")
    else:
        print("  - GCS export skipped. Set --gcs-bucket to generate cloud-backed tasks.")

    print("Conversion complete.")
    return 0


def main() -> int:
    return convert_to_label_studio_format(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
