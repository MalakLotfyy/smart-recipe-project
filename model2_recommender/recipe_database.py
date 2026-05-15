# ──────────────────────────────────────────────────────────────
# recipe_database.py
# ──────────────────────────────────────────────────────────────
import json
import os

# FIX: Dynamically get the directory of this script to safely locate the JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "recipes.json")

# Your original starting database
INITIAL_RECIPES = [
    {
        "id": 1,
        "name": "Shakshuka",
        "cuisine": "Egyptian/Middle Eastern",
        "ingredients": ["egg", "tomato", "onion", "pepper", "garlic", "olive_oil"],
        "required": ["egg", "tomato"],
        "tags": ["vegetarian", "halal", "breakfast", "quick"],
        "difficulty": "easy",
        "time_minutes": 20,
        "description": "Eggs poached in a spiced tomato and pepper sauce.",
        "steps": [
            "Heat olive oil in a pan over medium heat.",
            "Sauté onion and garlic until soft.",
            "Add chopped tomatoes and peppers. Season with cumin and paprika.",
            "Simmer for 10 minutes until sauce thickens.",
            "Make small wells and crack eggs into them.",
            "Cover and cook until eggs are just set. Serve with bread."
        ],
        "allergens": [],
    },
    # (Other initial recipes removed for brevity, keep yours here if you want a fallback)
]

def load_database() -> list[dict]:
    """Loads recipes from the JSON file. Creates it if it doesn't exist."""
    if not os.path.exists(DB_FILE):
        print(f"Creating new {DB_FILE} with initial data...")
        with open(DB_FILE, 'w') as f:
            json.dump(INITIAL_RECIPES, f, indent=4)
        return INITIAL_RECIPES
        
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_new_recipe(new_recipe: dict) -> dict:
    """Assigns an ID to a new recipe, saves it to JSON, and returns it."""
    db = load_database()
    
    # Generate a new unique ID
    new_id = max((r.get("id", 0) for r in db), default=0) + 1
    new_recipe["id"] = new_id
    
    db.append(new_recipe)
    
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)
        
    return new_recipe

def get_all_recipes() -> list[dict]:
    return load_database()

def get_recipe_by_id(recipe_id: int) -> dict | None:
    db = load_database()
    for r in db:
        if r["id"] == recipe_id:
            return r
    return None