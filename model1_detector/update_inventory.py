# ──────────────────────────────────────────────────────────────
# update_inventory.py
# Manages the fridge inventory — saves, updates, and reads
# detected ingredients from Model 1
# CSE351 Smart Recipe Project
# ──────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime

INVENTORY_PATH = os.path.join(os.path.dirname(__file__), "../data/fridge_inventory/inventory.json")


def load_inventory() -> dict:
    """Load current fridge inventory from JSON file."""
    if not os.path.exists(INVENTORY_PATH):
        return {}   # empty fridge on first run

    with open(INVENTORY_PATH, "r") as f:
        return json.load(f)


def save_inventory(inventory: dict):
    """Save the inventory dict back to JSON."""
    os.makedirs(os.path.dirname(INVENTORY_PATH), exist_ok=True)
    with open(INVENTORY_PATH, "w") as f:
        json.dump(inventory, f, indent=2)


def update_from_detection(detected_ingredients: list[dict]):
    """
    Takes the output of detect_ingredients() and updates the fridge inventory.

    Each new scan MERGES with the existing inventory (doesn't replace it).
    So if you scan eggs today and milk tomorrow, both stay in the inventory.

    Args:
        detected_ingredients: list from detect_ingredients.py, e.g.
            [{"name": "egg", "confidence": 0.92, "count": 3}, ...]
    """
    inventory = load_inventory()

    for item in detected_ingredients:
        name = item["name"]
        inventory[name] = {
            "name": name,
            "count": item["count"],
            "confidence": round(item["confidence"], 2),
            "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    save_inventory(inventory)
    print(f"\n📦 Inventory updated! You now have {len(inventory)} items in your fridge.")
    return inventory


def remove_item(item_name: str):
    """Remove an item from inventory (e.g. after using it in a recipe)."""
    inventory = load_inventory()
    if item_name in inventory:
        del inventory[item_name]
        save_inventory(inventory)
        print(f"Removed '{item_name}' from inventory.")
    else:
        print(f"'{item_name}' not found in inventory.")


def get_ingredient_names() -> list[str]:
    """Returns just the ingredient name list — used by Model 2 recommender."""
    inventory = load_inventory()
    return list(inventory.keys())


def display_inventory():
    """Pretty-print the current fridge contents."""
    inventory = load_inventory()
    if not inventory:
        print("🧊 Your fridge is empty! Take a photo to scan your ingredients.")
        return

    print("\n🧊 Current Fridge Inventory:")
    print(f"{'Item':<20} {'Count':<8} {'Confidence':<12} {'Last Seen'}")
    print("-" * 55)
    for item in inventory.values():
        print(f"{item['name']:<20} {item['count']:<8} {item['confidence']:.0%}          {item['last_seen']}")


# ── Test this file directly ──
if __name__ == "__main__":
    # Simulate what Model 1 would return
    sample_detection = [
        {"name": "egg",    "confidence": 0.92, "count": 4},
        {"name": "tomato", "confidence": 0.85, "count": 2},
        {"name": "onion",  "confidence": 0.78, "count": 1},
        {"name": "cheese", "confidence": 0.91, "count": 1},
    ]

    print("Simulating a detection from Model 1...")
    update_from_detection(sample_detection)
    display_inventory()
    print("\nIngredient list for Model 2:", get_ingredient_names())
