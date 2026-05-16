# ──────────────────────────────────────────────────────────────
# recommender.py
# ──────────────────────────────────────────────────────────────
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys, os
sys.path.append(os.path.dirname(__file__))
# import sys
sys.stdout.reconfigure(encoding='utf-8')

from recipe_database import get_all_recipes
from generator import generate_missing_recipe # <-- Import the new generator

class RecipeRecommender:
    def __init__(self):
        self.recipes = get_all_recipes()
        self.vectorizer = CountVectorizer()
        self._fit()

    def _fit(self):
        recipe_docs = [" ".join(r["ingredients"]) for r in self.recipes]
        # Handle case where DB is empty
        if recipe_docs:
            self.recipe_matrix = self.vectorizer.fit_transform(recipe_docs)

    def recommend(self, available_ingredients: list[str], user_preferences: dict, top_n: int = 5) -> list[dict]:
        # Refresh recipes in case a new one was added to the JSON
        self.recipes = get_all_recipes()
        self._fit()

        #lazem kolo lowercasse
        available_lower = [i.lower() for i in available_ingredients]
        
        fridge_doc = " ".join(available_lower)
        try:
            fridge_vector = self.vectorizer.transform([fridge_doc])
            similarities = cosine_similarity(fridge_vector, self.recipe_matrix).flatten()
        except Exception:
            similarities = [0] * len(self.recipes)

        scored = []
        for i, recipe in enumerate(self.recipes):
            score = float(similarities[i])

            has_required = all(
                req.lower() in available_lower 
                for req in recipe.get("required", [])
            )
            if not has_required:
                continue 

            score += 0.3

            dietary = user_preferences.get("dietary", [])
            if dietary and not any(d in recipe["tags"] for d in dietary):
                continue    

            allergies = user_preferences.get("allergies", [])
            if any(allergen in recipe.get("allergens", []) for allergen in allergies):
                continue

            max_time = user_preferences.get("max_time", 9999)
            if recipe["time_minutes"] > max_time:
                continue

            cuisine_pref = user_preferences.get("cuisine", [])
            if cuisine_pref and recipe["cuisine"] not in cuisine_pref:
                continue

            missing = [ing for ing in recipe["ingredients"] if ing.lower() not in available_lower]

            scored.append({
                **recipe,
                "score": round(score, 3),
                "match_percent": round(score * 100 / 1.3),
                "missing_ingredients": missing,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        
        # ── NEW LOGIC: GENERATE IF MATCH IS LOW ──
        best_match_percent = scored[0]["match_percent"] if scored else 0
        
        if best_match_percent < 50: # Threshold can be adjusted
            print(f"Best match is only {best_match_percent}%. Generating a new recipe...")
            new_recipe = generate_missing_recipe(available_ingredients, user_preferences)
            
            if new_recipe:
                # Format it for the UI
                new_recipe["score"] = 9.99
                new_recipe["match_percent"] = 100 # Perfect match since it was custom made
                new_recipe["missing_ingredients"] = []
                
                # Put the new recipe at the top of the list
                scored.insert(0, new_recipe)

        return scored[:top_n]

# ── Test this file directly ──
if __name__ == "__main__":
    recommender = RecipeRecommender()
    
    # Give the user very weird ingredients that aren't in the database to force generation
    fridge = ["tofu", "soy_sauce", "broccoli", "ginger", "noodles"]
    user_prefs = {"dietary": ["vegan"], "allergies": [], "max_time": 30}

    
    print("Fridge:", fridge)
    results = recommender.recommend(fridge, user_prefs, top_n=3)
    
    print("\nTop recipe recommendations:\n")
    for rank, recipe in enumerate(results, 1):
        print(f"#{rank} {recipe['name']} ({recipe['cuisine']})")
        print(f"   Match: {recipe.get('match_percent', 100)}% | Time: {recipe['time_minutes']} min")
        print()