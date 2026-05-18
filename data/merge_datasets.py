# merge_datasets.py
# Merges multiple Roboflow YOLO datasets into one
# Run once. Then train.
# AI assistance: Anthropic Claude — logic and remapping: team's own work

import os
import random
import shutil

# ══════════════════════════════════════════════
# EDIT THIS SECTION ONLY
# ══════════════════════════════════════════════

# Order matters — this determines the class numbers YOLO learns
MASTER_CLASSES = [
    "Apple",           # 0
    "Banana",          # 1
    "Beef",            # 2
    "Bread",           # 3
    "Bream-Fish",      # 4
    "Butter",          # 5
    "Cabbage",         # 6
    "Capsicum",        # 7
    "Carrots",         # 8
    "Cauliflower",     # 9
    "Chicken",         # 10
    "Chillies",        # 11
    "Cooking Cream",   # 12
    "Corn",            # 13
    "Cucumber",        # 14
    "Egg",             # 15
    "Flour",           # 16
    "Garlic",          # 17
    "Ginger",          # 18
    "Kiwi",            # 19
    "Lemon",           # 20
    "Lentils",         # 21
    "Lettuce",         # 22
    "Milk",            # 23
    "Mozzarella",      # 24
    "Oil",             # 25
    "Onion",           # 26
    "Orange",          # 27
    "Pasta",           # 28
    "Pear",            # 29
    "Potato",          # 30
    "Rice",            # 31
    "Spaghetti",       # 32
    "Strawberry",      # 33
    "Tomato",          # 34
    "Tomato Puree",    # 35
    "Tuna"             # 36
]

# List your datasets here
# Each entry: (path_to_dataset_root, {old_class_id: new_class_id})
# old_class_id = what's in that dataset's yaml (0,1,2...)
# new_class_id = what it maps to in MASTER_CLASSES above

DATASETS = [
    # 1. Ingredient Detection 3.yolov8
    {
        "path": "data/Ingredient Detection 3.yolov8",
        "class_map": {0: 3, 1: 5, 2: 8, 3: 11, 4: 14, 5: 15, 6: 17, 7: 18, 8: 20, 9: 34},
        "prefix": "ds1",
    },

    # 2. Ingredient Detection.yolov8
    {
        "path": "data/Ingredient Detection.yolov8",
        "class_map": {0: 0, 1: 6, 2: 7, 3: 8, 4: 9, 5: 11, 6: 14, 7: 17, 8: 18, 9: 20, 10: 30, 11: 34},
        "prefix": "ds2",
    },

    # 3. Ingredient Detection 2.yolov8
    {
        "path": "data/Ingredient Detection 2.yolov8",
        "class_map": {0: 7, 1: 8, 2: 14, 3: 17, 4: 18, 5: 22, 6: 26, 7: 30, 8: 34},
        "prefix": "ds3",
    },

    # 4. Smart Recipe. Fishyolov8
    {
        "path": "data/Smart Recipe. Fishyolov8",
        "class_map": {0: 4},
        "prefix": "ds4",
    },

    # 5. Food Ingredients Detection Beef.yolov8
    {
        "path": "data/Food Ingredients Detection Beef.yolov8",
        "class_map": {0: 2},
        "prefix": "ds5",
    },

    # 6. Food Ingredients Detection Chicken.yolov8
    {
        "path": "data/Food Ingredients Detection Chicken.yolov8",
        "class_map": {0: 10},
        "prefix": "ds6",
    },

    # 8. Fruits.yolov8
    {
        "path": "data/Fruits.yolov8",
        "class_map": {0: 0, 1: 1, 2: 19, 3: 27, 4: 29},
        "prefix": "ds8",
    },
    
    # 9. local supermarket.yolov8
    {
        "path": "data/local supermarket.yolov8",
        "class_map": {0: 12, 1: 13, 2: 16, 3: 21, 4: 23, 5: 24, 6: 25, 7: 28, 8: 31, 9: 32, 10: 35, 11: 36},
        "prefix": "ds9",
    },

    # 10. AI - Recipe Generator_Dataset_1.yolov8
    {
        "path": "data/AI - Recipe Generator_Dataset_1.yolov8",
        # Maps both Red Onion(1) and Yellow Onion(3) to the master Onion class(26)
        "class_map": {0: 30, 1: 26, 2: 34, 3: 26},
        "prefix": "ds10",
    },

    # 11. AI - Recipe Generator_Dataset_2.yolov8
    {
        "path": "data/AI - Recipe Generator_Dataset_2.yolov8",
        "class_map": {0: 0, 1: 1, 2: 15, 3: 27, 4: 33},
        "prefix": "ds11",
    }
]

#where to save the dataset after merging and remapping
OUTPUT_DIR = "data/merged_dataset"

# ══════════════════════════════════════════════
# DO NOT EDIT BELOW THIS LINE
# ══════════════════════════════════════════════

def process_split(split):
    """Process train, valid, or test split."""
    out_images = os.path.join(OUTPUT_DIR, split, "images")
    out_labels = os.path.join(OUTPUT_DIR, split, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    total_images = 0

    for dataset in DATASETS:
        prefix    = dataset["prefix"]
        class_map = dataset["class_map"]
        base      = dataset["path"]

        # Roboflow uses 'valid' not 'val' — handle both
        img_dir = os.path.join(base, split, "images")
        lbl_dir = os.path.join(base, split, "labels")

        if not os.path.exists(img_dir):
            print(f"  Skipping {prefix}/{split} — folder not found")
            continue

        images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

        for img_file in images:
            stem = os.path.splitext(img_file)[0]   # filename without extension
            ext  = os.path.splitext(img_file)[1]

            new_name = f"{prefix}_{stem}"           # e.g. "fruits_1", "eggs_23"

            # ── Copy image ──
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(out_images, new_name + ext)
            shutil.copy2(src_img, dst_img)

            # ── Remap and copy label ──
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            dst_lbl = os.path.join(out_labels, new_name + ".txt")

            if os.path.exists(src_lbl):
                with open(src_lbl, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_class = int(parts[0])
                    if old_class not in class_map:
                        print(f"  WARNING: class {old_class} in {img_file} not in class_map — skipping box")
                        continue
                    new_class = class_map[old_class]
                    new_lines.append(f"{new_class} {' '.join(parts[1:])}\n")

                with open(dst_lbl, "w") as f:
                    f.writelines(new_lines)
            else:
                # No label file = background image (no objects) — create empty label
                open(dst_lbl, "w").close()

            total_images += 1

    print(f"  {split}: {total_images} images processed")
    return total_images


def write_yaml(total):
    """Write the master dataset.yaml."""
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    names_str = "\n".join([f"  - {name}" for name in MASTER_CLASSES])

    content = f"""# Merged dataset — generated by merge_datasets.py
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: valid/images

nc: {len(MASTER_CLASSES)}
names:
{names_str}
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"\nYAML saved: {yaml_path}")


import random

if __name__ == "__main__":
    print("Merging datasets...\n")
    
    # Cleanup old merged dataset if it exists so we start completely fresh
    if os.path.exists(OUTPUT_DIR):
        print("Cleaning up old merged dataset...")
        shutil.rmtree(OUTPUT_DIR)

    total = 0
    # We will only pull from the train folders of your datasets
    for split in ["train"]: 
        print(f"Processing {split}...")
        total += process_split(split)

    # --- NEW AUTO-SPLIT LOGIC ---
    train_images_dir = os.path.join(OUTPUT_DIR, "train", "images")
    train_labels_dir = os.path.join(OUTPUT_DIR, "train", "labels")
    valid_images_dir = os.path.join(OUTPUT_DIR, "valid", "images")
    valid_labels_dir = os.path.join(OUTPUT_DIR, "valid", "labels")

    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    # Grab all the images we just merged into the train folder
    all_train_images = [f for f in os.listdir(train_images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    print("\nNo validation folders found. Auto-splitting 15% of training data for validation...")
    
    # Shuffle them so the AI gets a random mix of all ingredients on the test
    random.seed(42) 
    random.shuffle(all_train_images)
    
    # Calculate 15% of the total images
    split_idx = int(len(all_train_images) * 0.15)
    val_images_to_move = all_train_images[:split_idx]

    # Move the images and their matching text files to the valid folder
    for img in val_images_to_move:
        shutil.move(os.path.join(train_images_dir, img), os.path.join(valid_images_dir, img))
        
        stem = os.path.splitext(img)[0]
        lbl_file = stem + ".txt"
        if os.path.exists(os.path.join(train_labels_dir, lbl_file)):
            shutil.move(os.path.join(train_labels_dir, lbl_file), os.path.join(valid_labels_dir, lbl_file))

    print(f"Moved {len(val_images_to_move)} images to the validation set.")
    # ---------------------------

    write_yaml(total)
    print(f"\nDone. {total} total images processed.")
    print(f"Train set: {len(all_train_images) - len(val_images_to_move)} images")
    print(f"Valid set: {len(val_images_to_move)} images")
    print(f"Classes: {len(MASTER_CLASSES)}")
    print("\nNext: run train_yolo.py pointing to the new dataset.yaml")