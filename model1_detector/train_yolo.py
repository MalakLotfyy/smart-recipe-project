# ──────────────────────────────────────────────────────────────
# train_yolo.py
# Model 1 — Train YOLOv8 to detect fridge ingredients
# CSE351 Smart Recipe Project
# ──────────────────────────────────────────────────────────────
# AI assistance used for boilerplate setup (Anthropic Claude)
# Core training config, class selection, and analysis: team's own work
# ──────────────────────────────────────────────────────────────

from ultralytics import YOLO
import torch
import os

# ── Check if a GPU is available (much faster training) ──
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")
if device == "cpu":
    print("TIP: Run this on Google Colab for free GPU access!")

# ── Choose model size ──
# yolov8n = nano  (fastest, less accurate — good for testing)
# yolov8s = small (good balance)
# yolov8m = medium (more accurate, slower to train)
MODEL_SIZE = "yolov8s.pt"  # pre-trained on COCO, we fine-tune on our food data

# ── Load the pre-trained model ──
# YOLOv8 starts from weights trained on 80 COCO classes
# We then fine-tune it to recognize OUR ingredient classes
# This is called Transfer Learning — much faster than training from scratch
model = YOLO(MODEL_SIZE)

print("Starting training...")
print("This may take 30–60 minutes on CPU, ~10 min on Colab GPU")

# ── Train ──
results = model.train(
    data=os.path.join(os.path.dirname(__file__), "\data\merged_dataset\dataset.yaml"),

    epochs=50,          # number of full passes through the dataset
                        # increase to 100 if you have time and a GPU

    imgsz=640,          # input image size (standard for YOLOv8)

    batch=16,           # images per training step
                        # reduce to 8 if you run out of memory

    patience=10,        # stop early if no improvement after 10 epochs
                        # prevents overfitting

    device=device,

    project="runs/train",   # where results are saved
    name="ingredient_detector",

    # Data augmentation — YOLOv8 does this automatically:
    # flips, rotations, brightness changes, mosaic
    # This helps the model work on photos taken in different kitchens/lighting
    augment=True,

    # Save the best model weights (best validation mAP)
    save=True,
)

print("\n✅ Training complete!")
print(f"Best model saved to: runs/train/ingredient_detector/weights/best.pt")
print(f"\nResults summary:")
print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
print("\nFor your report — include these metrics + the confusion matrix image")
print("found in: runs/train/ingredient_detector/")
