"""
Batch Fish Detector — Gemini 2.5 Flash
======================================

This script scans all images in *INPUT_FOLDER* for fish, then writes:

  • an annotated PNG with bounding boxes
  • a JSON file listing every detected fish

This version includes a validation step: JSON and annotated images are only
saved if **all** detected bounding boxes from the API are correctly formatted.

Prerequisites
-------------
1. Install dependencies

       pip install google-genai pillow charset-normalizer chardet

2. Get an API key

       https://aistudio.google.com/  →  “Get API key”

3. Set the environment variable *(choose one name)*

   Linux / macOS (bash/zsh):

       export GOOGLE_API_KEY="AIza...your_key..."

   Windows PowerShell:

       setx GOOGLE_API_KEY "AIza...your_key..."

   The library also accepts **GEMINI_API_KEY**.

Usage
-----
Put images in *INPUT_FOLDER* (default: `test_image/`) and run

    python gemini.py

Results are written to *OUTPUT_FOLDER* (default: `output_results/`).
"""

import os
import json
import time
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

# --- 1. CONFIGURATION ---
INPUT_FOLDER = "test_image"
OUTPUT_FOLDER = "output_results"

PROMPT = """Detect all fish in the image. The response must be a JSON list of objects. Each object must contain a 'label' key with the text 'fish' and a 'box_2d' key with the bounding box coordinates as [ymin, xmin, ymax, xmax], normalized to a 0-1000 scale. If no fish are detected, return an empty list []."""


# --- 2. CORE PROCESSING FUNCTION FOR A SINGLE IMAGE ---
def process_image(image_path, client):
    """
    Takes a path to an image, calls the API, validates the response, and saves the files.
    """
    image_basename = os.path.basename(image_path)
    base_filename = os.path.splitext(image_basename)[0]

    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        config = types.GenerateContentConfig(response_mime_type="application/json")

        # --- API CALL (REMOVED THE ERRONEOUS 'request_options' ARGUMENT) ---
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, PROMPT],
            config=config
        )

        bounding_boxes_from_api = json.loads(response.text)

        if not bounding_boxes_from_api:
            print(f"  - No fish detected in {image_basename}.")
            return

        # --- VALIDATION STEP ---
        # First, check if all bounding boxes in the response are valid.
        is_response_valid = True
        for bounding_box in bounding_boxes_from_api:
            box_coords = bounding_box.get("box_2d")
            if not (isinstance(box_coords, list) and len(box_coords) == 4):
                print(f"  - !!! REJECTED: Malformed box found in {image_basename}. Response will not be saved.")
                is_response_valid = False
                break # No need to check further, the whole response is invalid

        # --- Only proceed if the entire response was valid ---
        if is_response_valid:
            # Save the JSON label file (now guaranteed to be clean)
            output_json_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(bounding_boxes_from_api, f, ensure_ascii=False, indent=2)
            print(f"  - ✓ Valid JSON label saved.")

            # Annotate and save the image
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            width, height = annotated_image.size

            for bounding_box in bounding_boxes_from_api:
                label = bounding_box["label"]
                y1_1000, x1_1000, y2_1000, x2_1000 = bounding_box["box_2d"]
                abs_y1 = int(y1_1000 / 1000 * height)
                abs_x1 = int(x1_1000 / 1000 * width)
                abs_y2 = int(y2_1000 / 1000 * height)
                abs_x2 = int(x2_1000 / 1000 * width)

                draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline="lime", width=4)
                draw.text([abs_x1, abs_y1 - 20], label, fill="lime", font_size=18)

            output_image_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_annotated.png")
            annotated_image.save(output_image_path)
            print(f"  - ✓ Annotated image saved.")

    except Exception as e:
        print(f"  - !!! An error occurred while processing {image_basename}: {e}")


# --- 3. MAIN SCRIPT EXECUTION ---
def main():
    """
    Orchestrates the batch processing of all images in the input folder.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: API Key not found. Please set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found. Please create it and add your images.")
        return
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("--- Starting Batch Image Detection ---")
    print(f"Input folder: '{INPUT_FOLDER}'")
    print(f"Output will be saved to: '{OUTPUT_FOLDER}'")

    client = genai.Client()

    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print("No images found in the input folder.")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    for i, filename in enumerate(image_files):
        print(f"\n[{i + 1}/{total_images}] Processing: {filename}")
        image_path = os.path.join(INPUT_FOLDER, filename)
        process_image(image_path, client)
        time.sleep(1)

    print("\n--- Batch processing complete! ---")


if __name__ == "__main__":
    main()