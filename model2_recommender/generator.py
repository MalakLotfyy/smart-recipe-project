# ──────────────────────────────────────────────────────────────
# generator.py
# Uses Gemini Structured Outputs to generate JSON-compatible recipes
# ──────────────────────────────────────────────────────────────
import json
import os
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from recipe_database import save_new_recipe

load_dotenv()

# FIX: Explicitly load your specific GEMINI_API_KEY from .env
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Define the EXACT structure of your database dictionaries
class RecipeSchema(BaseModel):
    name: str
    cuisine: str
    ingredients: list[str] = Field(description="List of all ingredients used in lowercase")
    required: list[str] = Field(description="The core ingredients required from the available list")
    tags: list[str] = Field(description="e.g., vegetarian, halal, dinner, quick")
    difficulty: str = Field(description="easy, medium, or hard")
    time_minutes: int
    description: str
    steps: list[str]
    allergens: list[str]

def generate_missing_recipe(available_ingredients: list[str], preferences: dict) -> dict | None:
    """Calls Gemini to invent a recipe based on fridge contents and saves it."""
    print("\n[API] Generating a brand new recipe to match your fridge...")
    
    prompt = f"""
    You are an expert culinary AI. The user's fridge only has these ingredients: {available_ingredients}
    User Preferences: {preferences}
    
    Invent a delicious, realistic recipe that primarily uses these available ingredients. 
    It is okay to add a few common pantry staples (like salt, pepper, oil, water) if absolutely necessary.
    Ensure all fields are filled out realistically.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': RecipeSchema,
                'temperature': 0.7
            },
        )
        
        # Parse AI response into a Python dictionary
        new_recipe_data = json.loads(response.text)
        
        # Save it to our JSON database!
        saved_recipe = save_new_recipe(new_recipe_data)
        print(f"[API SUCCESS] Invented and saved: {saved_recipe['name']}!")
        return saved_recipe
        
    except Exception as e:
        print(f"[API ERROR] Failed to generate recipe: {e}")
        return None