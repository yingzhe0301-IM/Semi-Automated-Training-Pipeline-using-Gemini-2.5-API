import os
import json
from PIL import Image

# --- NEW CONFIGURATION: SET YOUR BUCKET DETAILS HERE ---
BUCKET_NAME = "yingzhe_gemini_fish"  # 替换成你刚刚创建的存储桶名字
BUCKET_PREFIX = "fish_images/"  # 我们在桶里创建的文件夹名

# --- Other Configurations ---
GEMINI_JSON_FOLDER = "output_results"
ORIGINAL_IMAGE_FOLDER = "test_image"
LABEL_STUDIO_EXPORT_FILE = "import_to_ls_gcs.json"  # New output file name


def convert_to_label_studio_format():
    label_studio_tasks = []
    print(f"Starting conversion for GCS from '{GEMINI_JSON_FOLDER}'...")

    json_files = [f for f in os.listdir(GEMINI_JSON_FOLDER) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found to convert.")
        return

    for json_filename in json_files:
        json_path = os.path.join(GEMINI_JSON_FOLDER, json_filename)

        with open(json_path, 'r', encoding='utf-8') as f:
            gemini_results = json.load(f)

        ls_results = []
        for bounding_box in gemini_results:
            # ... (the conversion logic is the same)
            box_coords = bounding_box.get("box_2d")
            label = bounding_box.get("label")
            if not (isinstance(box_coords, list) and len(box_coords) == 4):
                continue
            y1_1000, x1_1000, y2_1000, x2_1000 = box_coords
            x_percent = (x1_1000 / 1000) * 100
            y_percent = (y1_1000 / 1000) * 100
            width_percent = ((x2_1000 - x1_1000) / 1000) * 100
            height_percent = ((y2_1000 - y1_1000) / 1000) * 100
            ls_results.append({
                "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                "value": {"rectanglelabels": [label], "x": x_percent, "y": y_percent, "width": width_percent,
                          "height": height_percent}
            })

        # --- THIS IS THE CRITICAL CHANGE ---
        # Generate the GCS path for Label Studio
        image_basename = os.path.splitext(json_filename)[0]
        # We need to find the original extension
        original_image_name = ""
        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            if os.path.exists(os.path.join(ORIGINAL_IMAGE_FOLDER, image_basename + ext)):
                original_image_name = image_basename + ext
                break

        image_path_for_ls = f"gs://{BUCKET_NAME}/{BUCKET_PREFIX}{original_image_name}"

        label_studio_tasks.append({
            "data": {"image": image_path_for_ls},
            "predictions": [{"model_version": "gemini-2.5-flash", "result": ls_results}]
        })

    with open(LABEL_STUDIO_EXPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(label_studio_tasks, f, ensure_ascii=False, indent=2)

    print(f"\nConversion complete! ✨")
    print(f"{len(label_studio_tasks)} tasks written to '{LABEL_STUDIO_EXPORT_FILE}'.")


if __name__ == "__main__":
    convert_to_label_studio_format()