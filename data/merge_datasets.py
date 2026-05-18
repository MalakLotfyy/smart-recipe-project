# merge_datasets.py
# Merges multiple Roboflow YOLO datasets into one
# Run once. Then train.
# AI assistance: Anthropic Claude — logic and remapping: team's own work

import os
import shutil

# ══════════════════════════════════════════════
# EDIT THIS SECTION ONLY
# ══════════════════════════════════════════════

# Your final master class list — decide this yourself
# Order matters — this determines the class numbers YOLO learns
MASTER_CLASSES = [
   "Apple",           # 0
    "Avocado",         # 1
    "Banana",          # 2
    "Beef",            # 3
    "Bitter Melon",    # 4
    "Bread",           # 5
    "Bream-Fish",      # 6
    "Brinjal",         # 7
    "Butter",          # 8
    "Cabbage",         # 9
    "Calabash",        # 10
    "Capsicum",        # 11
    "Carrots",         # 12
    "Cauliflower",     # 13
    "Cherry",          # 14
    "Chicken",         # 15
    "Chillies",        # 16
    "Cooking Cream",   # 17
    "Corn",            # 18
    "Cucumber",        # 19
    "Dragon Fruit",    # 20
    "Egg",             # 21
    "Flour",           # 22
    "Garlic",          # 23
    "Ginger",          # 24
    "Green Chili",     # 25
    "Guava",           # 26
    "Kiwi",            # 27
    "Lady Finger",     # 28
    "Lemon",           # 29
    "Lentils",         # 30
    "Lettuce",         # 31
    "Mango",           # 32
    "Milk",            # 33
    "Mozzarella",      # 34
    "Oil",             # 35
    "Onion",           # 36
    "Orange",          # 37
    "Oren",            # 38
    "Pasta",           # 39
    "Peach",           # 40
    "Pear",            # 41
    "Pineapple",       # 42
    "Potato",          # 43
    "Rice",            # 44
    "Spaghetti",       # 45
    "Sponge Gourd",    # 46
    "Strawberry",      # 47
    "Sugar Apple",     # 48
    "Tomato",          # 49
    "Tomato Puree",    # 50
    "Tuna",            # 51
    "Watermelon"       # 52
]

# List your datasets here
# Each entry: (path_to_dataset_root, {old_class_id: new_class_id})
# old_class_id = what's in that dataset's yaml (0,1,2...)
# new_class_id = what it maps to in MASTER_CLASSES above

DATASETS = [
    # 1. Ingredient Detection 3.yolov8
    {
        "path": "data/Ingredient Detection 3.yolov8",
        "class_map": {0: 5, 1: 8, 2: 12, 3: 16, 4: 19, 5: 21, 6: 23, 7: 24, 8: 29, 9: 49},
        "prefix": "ds1",
    },

    # 2. Ingredient Detection.yolov8
    {
        "path": "data/Ingredient Detection.yolov8",
        "class_map": {0: 0, 1: 9, 2: 11, 3: 12, 4: 13, 5: 16, 6: 19, 7: 23, 8: 24, 9: 29, 10: 43, 11: 49},
        "prefix": "ds2",
    },

    # 3. Ingredient Detection 2.yolov8
    {
        "path": "data/Ingredient Detection 2.yolov8",
        "class_map": {0: 11, 1: 12, 2: 19, 3: 23, 4: 24, 5: 31, 6: 36, 7: 43, 8: 49},
        "prefix": "ds3",
    },

    # 4. Smart Recipe. Fishyolov8
    {
        "path": "data/Smart Recipe. Fishyolov8",
        "class_map": {0: 6},
        "prefix": "ds4",
    },

    # 5. Food Ingredients Detection Beef.yolov8
    {
        "path": "data/Food Ingredients Detection Beef.yolov8",
        "class_map": {0: 3},
        "prefix": "ds5",
    },

    # 6. Food Ingredients Detection Chicken.yolov8
    {
        "path": "data/Food Ingredients Detection Chicken.yolov8",
        "class_map": {0: 15},
        "prefix": "ds6",
    },

#remove 7. Food Ingredients Detection Tomato.yolov8

    # 8. Fruits.yolov8
    {
        "path": "data/Fruits.yolov8",
        "class_map": {0: 0, 1: 2, 2: 27, 3: 37, 4: 41},
        "prefix": "ds8",
    },
    
    # 9. local supermarket.yolov8
    {
        "path": "data/local supermarket.yolov8",
        "class_map": {0: 17, 1: 18, 2: 22, 3: 30, 4: 33, 5: 34, 6: 35, 7: 39, 8: 44, 9: 45, 10: 50, 11: 51},
        "prefix": "ds9",
    },
]

# Where to save the merged dataset
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


if __name__ == "__main__":
    print("Merging datasets...\n")

    # Validate paths before starting
    for ds in DATASETS:
        if not os.path.exists(ds["path"]):
            print(f"ERROR: path not found: {ds['path']}")
            print("Fix the path in DATASETS list and run again.")
            exit(1)

    total = 0
    for split in ["train", "valid", "test"]:
        print(f"Processing {split}...")
        total += process_split(split)

    write_yaml(total)

    print(f"\nDone. {total} total images merged into: {OUTPUT_DIR}")
    print(f"Classes: {len(MASTER_CLASSES)}")
    print("\nNext: run train_yolo.py pointing to the new dataset.yaml")
