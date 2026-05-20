# ──────────────────────────────────────────────────────────────
# generator.py
# Purpose: Uses Groq's API to dynamically generate new recipes 
# when the local database doesn't have a good match.
# ──────────────────────────────────────────────────────────────
import json
import os
import streamlit as st
from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from recipe_database import save_new_recipe

# Load local environment variables (for testing outside of Streamlit Cloud)
load_dotenv()

# ────────────────────────────────────────
# SCHEMA DEFINITION
# We use Pydantic to define an exact blueprint for the data. 
# This is a crucial safety measure to ensure the LLM doesn't hallucinate 
# random keys that would crash our frontend UI.
# ────────────────────────────────────────
class RecipeSchema(BaseModel):
    name: str = Field(description="The absolute shortest, most common name for the dish. Maximum 2 words. Examples: 'Milk Pasta', 'Tomato Soup', 'Fried Rice'. No adjectives like 'simple', 'creamy', 'delicious', or 'quick'.") 
    cuisine: str
    ingredients: list[str] = Field(description="List of all ingredients used in lowercase")
    required: list[str] = Field(description="The core ingredients required from the available list")
    tags: list[str] = Field(description="e.g., vegetarian, halal, dinner, quick")
    difficulty: str = Field(description="easy, medium, or hard")
    time_minutes: int
    description: str
    instructions: list[str] = Field(description="The step-by-step cooking instructions") 
    allergens: list[str]


def generate_missing_recipe(available_ingredients: list[str], preferences: dict) -> dict | None:
    """
    Calls the Groq API to invent a recipe based on fridge contents, 
    forces it into a strict JSON format, and saves it to the database.
    """
    print("\n[API] Generating a brand new recipe to match your fridge...")
    
    # 1. SAFELY FETCH API KEY: Tries Streamlit Secrets first (Cloud), falls back to .env (Local)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")
        
    if not api_key:
        print("[API ERROR] No Groq API key found locally or in secrets.")
        return None

    # Initialize the Groq client
    client = Groq(api_key=api_key)
    
    # 2. PROMPT ENGINEERING: Strict instructions to prevent marketing buzzwords 
    # and force the model to respect dietary/ingredient limitations.
    prompt = f"""
    You are an expert culinary AI. The user's fridge only has these ingredients: {available_ingredients}
    User Preferences: {preferences}
    
    Invent a delicious, realistic recipe that primarily uses these available ingredients. 
    It is okay to add a few common pantry staples (like salt, pepper, oil, water) if absolutely necessary.
    
    CRITICAL RULES FOR THE OUTPUT NAME:
    1. The 'name' MUST be a super short, direct, and common 1 or 2-word name. 
    2. NEVER use adjectives or marketing buzzwords in the name (DO NOT use: Simple, Creamy, Easy, Quick, Delicious, Homemade, Best, etc.).
    3. Bad Name Example: 'Simple Creamy Milk Pasta' 
    4. Good Name Example: 'Milk Pasta' or 'White Pasta'
    5. The name must be easily searchable and represent the core recipe instantly.
    
    CRITICAL RULES FOR THE INSTRUCTIONS:
    1. Provide the cooking steps inside the 'instructions' field strictly.
    
    OUTPUT FORMAT:
    You must return ONLY a valid JSON object. Do not wrap it in markdown. 
    The JSON keys must match: name, cuisine, ingredients, required, tags, difficulty, time_minutes, description, instructions, allergens.
    """
    
    try:
        # 3. EXECUTE API CALL: We explicitly tell Groq to return a JSON object
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5 # Kept at 0.5 for a balance of creativity and structural obedience
        )
        
        # 4. PARSE & SAVE: Convert the text response into a Python dictionary
        new_recipe_data = json.loads(response.choices[0].message.content)
        saved_recipe = save_new_recipe(new_recipe_data)
        
        print(f"[API SUCCESS] Invented and saved: {saved_recipe['name']}!")
        return saved_recipe
        
    except Exception as e:
        print(f"[API ERROR] Failed to generate recipe: {e}")
        return None