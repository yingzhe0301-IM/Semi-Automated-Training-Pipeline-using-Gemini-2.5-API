"""Batch object detection with Gemini.

This script scans an input directory, asks Gemini for bounding boxes, validates
the JSON response, and writes both structured detections and annotated images.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from config import (
    BOX_COLOR,
    DEFAULT_DELAY_SECONDS,
    DEFAULT_INPUT_DIR,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT,
    SUPPORTED_EXTENSIONS,
    TEXT_COLOR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini object detection on every image in a directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing images. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON and annotated images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        help=f"Gemini model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("GEMINI_PROMPT", DEFAULT_PROMPT),
        help="Detection prompt sent to Gemini.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("GEMINI_REQUEST_DELAY", DEFAULT_DELAY_SECONDS)),
        help=f"Delay between API calls in seconds. Default: {DEFAULT_DELAY_SECONDS}",
    )
    return parser.parse_args()


def load_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def list_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_detections(payload: Any, image_name: str) -> list[dict[str, Any]] | None:
    if not isinstance(payload, list):
        print(f"  - !!! REJECTED: {image_name} returned non-list JSON.")
        return None

    validated: list[dict[str, Any]] = []
    for index, detection in enumerate(payload, start=1):
        if not isinstance(detection, dict):
            print(f"  - !!! REJECTED: Detection #{index} in {image_name} is not an object.")
            return None

        label = detection.get("label")
        box_coords = detection.get("box_2d")
        if not isinstance(label, str) or not label:
            print(f"  - !!! REJECTED: Detection #{index} in {image_name} has no label.")
            return None
        if not (
            isinstance(box_coords, list)
            and len(box_coords) == 4
            and all(isinstance(value, (int, float)) for value in box_coords)
        ):
            print(
                f"  - !!! REJECTED: Detection #{index} in {image_name} has malformed box_2d."
            )
            return None

        ymin, xmin, ymax, xmax = box_coords
        if not all(0 <= value <= 1000 for value in box_coords):
            print(
                f"  - !!! REJECTED: Detection #{index} in {image_name} is outside the 0-1000 range."
            )
            return None
        if ymin >= ymax or xmin >= xmax:
            print(
                f"  - !!! REJECTED: Detection #{index} in {image_name} has inverted coordinates."
            )
            return None

        validated.append({"label": label, "box_2d": [ymin, xmin, ymax, xmax]})

    return validated


def annotate_and_save_image(
    image: Image.Image, detections: list[dict[str, Any]], output_path: Path
) -> None:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = annotated_image.size

    for detection in detections:
        label = detection["label"]
        y1_1000, x1_1000, y2_1000, x2_1000 = detection["box_2d"]
        abs_y1 = int(y1_1000 / 1000 * height)
        abs_x1 = int(x1_1000 / 1000 * width)
        abs_y2 = int(y2_1000 / 1000 * height)
        abs_x2 = int(x2_1000 / 1000 * width)

        draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline=BOX_COLOR, width=4)
        draw.text((abs_x1, max(abs_y1 - 18, 0)), label, fill=TEXT_COLOR)

    annotated_image.save(output_path)


def process_image(
    image_path: Path,
    client: genai.Client,
    output_dir: Path,
    model: str,
    prompt: str,
) -> None:
    image_name = image_path.name
    output_json_path = output_dir / f"{image_path.stem}.json"
    output_image_path = output_dir / f"{image_path.stem}_annotated.png"

    try:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB") if raw_image.mode != "RGB" else raw_image.copy()

        config = types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(
            model=model,
            contents=[image, prompt],
            config=config,
        )

        detections = validate_detections(json.loads(response.text), image_name)
        if detections is None:
            return
        if not detections:
            print(f"  - No detections in {image_name}.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(detections, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        annotate_and_save_image(image, detections, output_image_path)

        print(f"  - ✓ JSON saved to {output_json_path}")
        print(f"  - ✓ Annotated image saved to {output_image_path}")
    except json.JSONDecodeError as exc:
        print(f"  - !!! Failed to parse JSON for {image_name}: {exc}")
    except Exception as exc:
        print(f"  - !!! An error occurred while processing {image_name}: {exc}")


def main() -> int:
    args = parse_args()
    api_key = load_api_key()
    if not api_key:
        print("Error: set GOOGLE_API_KEY or GEMINI_API_KEY before running this script.")
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: input directory '{args.input_dir}' was not found.")
        return 1

    image_files = list_image_files(args.input_dir)
    if not image_files:
        print(f"No supported images found in '{args.input_dir}'.")
        return 0

    client = genai.Client(api_key=api_key)

    print("--- Starting Batch Image Detection ---")
    print(f"Input directory: '{args.input_dir}'")
    print(f"Output directory: '{args.output_dir}'")
    print(f"Model: '{args.model}'")
    print(f"Found {len(image_files)} images to process.")

    for index, image_path in enumerate(image_files, start=1):
        print(f"\n[{index}/{len(image_files)}] Processing: {image_path.name}")
        process_image(
            image_path=image_path,
            client=client,
            output_dir=args.output_dir,
            model=args.model,
            prompt=args.prompt,
        )
        if args.delay > 0 and index < len(image_files):
            time.sleep(args.delay)

    print("\n--- Batch processing complete! ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
