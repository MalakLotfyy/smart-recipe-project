# ──────────────────────────────────────────────────────────────
# llm_agent.py
# Model 2, Step 2 — LLM Agent
# ──────────────────────────────────────────────────────────────
import os
import requests
from google import genai
from dotenv import load_dotenv

load_dotenv()   

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
SPOONACULAR_KEY   = os.getenv("SPOONACULAR_API_KEY", "")

# ────────────────────────────────────────
# PART 1 — LLM recipe writer
# ────────────────────────────────────────

def generate_recipe_with_llm(recipe: dict, available_ingredients: list[str]) -> str:
    """
    Uses Gemini to write a friendly, detailed recipe
    tailored to the exact ingredients the user has available.
    """
    if not GEMINI_API_KEY:
        print("⚠️  No Gemini API key found. Using stored recipe steps.")
        return _format_recipe_without_llm(recipe, available_ingredients)

    # FIX: Use the modern SDK instead of raw requests
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""You are a friendly home cooking assistant. Write a clear, easy-to-follow recipe.

Recipe: {recipe['name']}
Cuisine: {recipe['cuisine']}
Available ingredients: {', '.join(available_ingredients)}
All recipe ingredients: {', '.join(recipe['ingredients'])}
Missing ingredients: {', '.join(recipe.get('missing_ingredients', [])) or 'none'}
Cooking time: {recipe['time_minutes']} minutes
Difficulty: {recipe['difficulty']}

Write the recipe with:
1. A warm one-sentence intro
2. Ingredients section (use what's available, suggest substitutes for missing items)
3. Step-by-step instructions (numbered, clear)
4. One quick tip at the end

Keep it practical and friendly. Use simple language."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"⚠️  LLM call failed: {e}. Using stored steps.")
        return _format_recipe_without_llm(recipe, available_ingredients)


def _format_recipe_without_llm(recipe: dict, available_ingredients: list[str]) -> str:
    missing = recipe.get("missing_ingredients", [])
    lines = [
        f"# {recipe['name']}",
        f"*{recipe['description']}*",
        f"\n**Cuisine:** {recipe['cuisine']} | **Time:** {recipe['time_minutes']} min | **Difficulty:** {recipe['difficulty']}",
        "\n## Instructions",
    ]
    # Checks for 'steps', falls back to 'instructions', or defaults to an empty list
    steps_list = recipe.get("steps") or recipe.get("instructions") or recipe.get("steps_instructions") or []

    for i, step in enumerate(steps_list, 1):
        lines.append(f"{i}. {step}")
    if missing:
        lines.append(f"\n💡 **You're missing:** {', '.join(missing)} — but you can try without them or substitute!")
    return "\n".join(lines)


# ────────────────────────────────────────
# PART 2 — Nutrition API (Unchanged)
# ────────────────────────────────────────

def fetch_nutrition(recipe_name: str, ingredients: list[str]) -> dict:
    if not SPOONACULAR_KEY:
        return _estimate_nutrition(ingredients)

    search_url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "apiKey": SPOONACULAR_KEY,
        "query": recipe_name,
        "includeNutrition": True,
        "number": 1,
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        data = response.json()

        if data.get("results"):
            nutrition = data["results"][0].get("nutrition", {})
            nutrients = {n["name"]: n["amount"] for n in nutrition.get("nutrients", [])}
            return {
                "calories":      round(nutrients.get("Calories", 0)),
                "protein_g":     round(nutrients.get("Protein", 0), 1),
                "carbs_g":       round(nutrients.get("Carbohydrates", 0), 1),
                "fat_g":         round(nutrients.get("Fat", 0), 1),
                "fiber_g":       round(nutrients.get("Fiber", 0), 1),
                "source": "Spoonacular API",
            }
    except Exception as e:
        print(f"⚠️  Nutrition API failed: {e}")

    return _estimate_nutrition(ingredients)


def _estimate_nutrition(ingredients: list[str]) -> dict:
    estimates = {
        "egg":         {"calories": 70,  "protein_g": 6,   "carbs_g": 0.5, "fat_g": 5},
        "tomato":      {"calories": 20,  "protein_g": 1,   "carbs_g": 4,   "fat_g": 0.2},
        "onion":       {"calories": 40,  "protein_g": 1,   "carbs_g": 9,   "fat_g": 0.1},
        "cheese":      {"calories": 110, "protein_g": 7,   "carbs_g": 0.4, "fat_g": 9},
        "chicken":     {"calories": 165, "protein_g": 31,  "carbs_g": 0,   "fat_g": 3.6},
        "ground_beef": {"calories": 250, "protein_g": 26,  "carbs_g": 0,   "fat_g": 15},
        "rice":        {"calories": 130, "protein_g": 2.7, "carbs_g": 28,  "fat_g": 0.3},
        "pasta":       {"calories": 158, "protein_g": 6,   "carbs_g": 31,  "fat_g": 1},
        "yogurt":      {"calories": 60,  "protein_g": 5,   "carbs_g": 7,   "fat_g": 0.4},
        "bread":       {"calories": 80,  "protein_g": 3,   "carbs_g": 15,  "fat_g": 1},
        "carrot":      {"calories": 25,  "protein_g": 0.6, "carbs_g": 6,   "fat_g": 0.1},
        "cucumber":    {"calories": 10,  "protein_g": 0.5, "carbs_g": 2,   "fat_g": 0.1},
        "olive_oil":   {"calories": 60,  "protein_g": 0,   "carbs_g": 0,   "fat_g": 7},
    }

    totals = {"calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    for ing in ingredients:
        if ing in estimates:
            for key in totals:
                totals[key] += estimates[ing][key]

    totals["source"] = "Estimated"
    return totals


# ────────────────────────────────────────
# PART 3 — Full agent pipeline
# ────────────────────────────────────────

def run_recipe_agent(recipe: dict, available_ingredients: list[str]) -> dict:
    print(f"🤖 Agent working on: {recipe['name']}...")
    recipe_text = generate_recipe_with_llm(recipe, available_ingredients)
    nutrition = fetch_nutrition(recipe["name"], recipe["ingredients"])

    return {
        "recipe_text": recipe_text,
        "nutrition": nutrition,
        "recipe_meta": recipe,
    }