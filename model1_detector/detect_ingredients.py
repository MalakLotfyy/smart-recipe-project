# ──────────────────────────────────────────────────────────────
# detect_ingredients.py
# Model 1 — Run YOLOv8 on a photo and extract detected ingredients
# CSE351 Smart Recipe Project
# ──────────────────────────────────────────────────────────────
# AI assistance used for boilerplate (Anthropic Claude)
# Detection thresholds, class filtering: team's own decisions
# ──────────────────────────────────────────────────────────────

from ultralytics import YOLO
from PIL import Image
import cv2
import argparse
import os

# ── Path to your trained model weights ──
# After training, change this to: "runs/train/ingredient_detector/weights/best.pt"
# For now, we use the pretrained COCO model so you can test without training first
DEFAULT_MODEL = "yolov8s.pt"

# ── Confidence threshold ──
# Only count detections where the model is > 50% confident
# Lower this if the model misses items, raise it if it detects wrong things
CONFIDENCE_THRESHOLD = 0.5


def detect_ingredients(image_path: str, model_path: str = DEFAULT_MODEL) -> list[dict]:
    """
    Runs YOLOv8 on an image and returns a list of detected ingredients.

    Args:
        image_path: path to the photo (jpg, png, etc.)
        model_path: path to the trained YOLOv8 weights file

    Returns:
        List of dicts like:
        [
            {"name": "egg",    "confidence": 0.92, "count": 3},
            {"name": "tomato", "confidence": 0.87, "count": 1},
        ]
    """
    # Load model
    model = YOLO(model_path)

    # Run detection
    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        save=True,              # saves annotated image to runs/detect/
        save_txt=True,          # saves label .txt file (same format as training data)
        verbose=False,
    )

    # ── Parse results into a clean list ──
    detected = {}   # key: class name, value: {confidence, count}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # e.g. "egg"
            confidence = float(box.conf[0])     # e.g. 0.92

            if class_name not in detected:
                detected[class_name] = {"name": class_name, "confidence": confidence, "count": 1}
            else:
                detected[class_name]["count"] += 1
                # keep the highest confidence seen for this class
                detected[class_name]["confidence"] = max(detected[class_name]["confidence"], confidence)

    ingredients = list(detected.values())

    # ── Print summary ──
    if ingredients:
        print(f"\n✅ Detected {len(ingredients)} ingredient(s) in '{os.path.basename(image_path)}':")
        for item in ingredients:
            print(f"   {item['name']:<20} confidence: {item['confidence']:.0%}   count: {item['count']}")
    else:
        print("❌ No ingredients detected. Try a clearer photo or lower CONFIDENCE_THRESHOLD.")

    return ingredients


def draw_detections(image_path: str, model_path: str = DEFAULT_MODEL, output_path: str = "detected.jpg"):
    """
    Draws bounding boxes on the image and saves it.
    Useful for the presentation demo!
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Get annotated frame
    annotated = results[0].plot()   # numpy array with boxes drawn
    cv2.imwrite(output_path, annotated)
    print(f"Annotated image saved to: {output_path}")
    return output_path


# ── Run from command line ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect ingredients in a photo")
    parser.add_argument("--image",  type=str, required=True, help="Path to photo")
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL, help="Path to model weights")
    parser.add_argument("--output", type=str, default="detected.jpg", help="Output annotated image path")
    args = parser.parse_args()

    ingredients = detect_ingredients(args.image, args.model)
    draw_detections(args.image, args.model, args.output)
