# ──────────────────────────────────────────────────────────────
# streamlit_app.py — ULTIMATE MERGED VERSION
# Combines Live Camera/Manual inputs with Modern UI & Audio TTS
# ──────────────────────────────────────────────────────────────
import streamlit as st
import sys, os
import json
import tempfile
import base64
from PIL import Image
from gtts import gTTS
from io import BytesIO

# --- Path Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model1_detector.detect_ingredients import detect_ingredients, draw_detections
from model1_detector.update_inventory import (
    update_from_detection, load_inventory, get_ingredient_names, remove_item
)
from model2_recommender.recommender import RecipeRecommender
from model2_recommender.llm_agent import run_recipe_agent

# UPDATE THIS to match your MASTER_CLASSES
ALL_CLASSES = [
    "Apple", "Banana", "Beef", "Bread", "Bream-Fish", "Butter", "Cabbage", 
    "Capsicum", "Carrots", "Cauliflower", "Chicken", "Chillies", "Cooking Cream", 
    "Corn", "Cucumber", "Egg", "Flour", "Garlic", "Ginger", "Kiwi", "Lemon", 
    "Lentils", "Lettuce", "Milk", "Mozzarella", "Oil", "Onion", "Orange", 
    "Pasta", "Pear", "Potato", "Rice", "Spaghetti", "Strawberry", "Tomato", 
    "Tomato Puree", "Tuna"
]

Pantry_extras = [ "Avocado",  "Bitter Melon",   
    "Brinjal", "Calabash", "Cherry",  "Dragon Fruit", "Green Chili", "Guava", "Lady Finger", 
    "Mango", "Oren", "Peach",  "Pineapple", "Sponge Gourd", "Sugar Apple", 
    "Watermelon","Water", "Salt", "Black Pepper", "Garlic Powder", "Onion Powder",
    "Cumin", "Paprika", "Cinnamon", "Turmeric", "Vinegar", "Sugar", "Soy Sauce", "Olive Oil",]

#combine them 
manual_mode_samples = sorted(ALL_CLASSES + Pantry_extras)


# ── AUDIO LOGIC ──
def play_recipe_audio(full_recipe_text):
    try:
        if "Instructions" in full_recipe_text:
            speech_content = full_recipe_text.split("Instructions")[-1].strip()
        else:
            speech_content = full_recipe_text
            
        tts_engine = gTTS(text=speech_content, lang='en')
        audio_stream = BytesIO()
        tts_engine.write_to_fp(audio_stream)
        audio_encoded = base64.b64encode(audio_stream.getvalue()).decode()
        audio_component = f"""
            <div style="margin: 10px 0;">
                <audio controls autoplay style="width: 100%; border-radius: 10px;">
                    <source src="data:audio/mp3;base64,{audio_encoded}" type="audio/mp3">
                </audio>
            </div>
        """
        st.markdown(audio_component, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS Error: {e}")

# ── ADAPTIVE CSS INJECTION (Supports Dark & Light Mode) ──
def apply_custom_style():
    st.markdown("""
        <style>
        /* Buttons */
        div.stButton > button {
            border-radius: 8px;
            background-color: #FF4B4B;
            color: white !important;
            border: none;
            transition: 0.3s;
            font-weight: bold;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #ff3333;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        /* Cards Effect */
        .stExpander {
            background-color: var(--secondary-background-color) !important;
            border-radius: 12px !important;
            border: 1px solid var(--faded-text-color) !important;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# ── HELPER FUNCTION: DETECTION ──
def run_detection_and_show(image_pil, current_model_path):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_pil.save(tmp.name)
        tmp_path = tmp.name

    try:
        detected = detect_ingredients(tmp_path, current_model_path)
    except Exception as e:
        st.error(f"Detection error: {e}")
        os.unlink(tmp_path)
        return

    if not detected:
        st.warning("Nothing detected. Try better lighting, a closer shot, or use Manual mode.")
        os.unlink(tmp_path)
        return

    st.success(f"✅ Found {len(detected)} ingredient(s)!")

    # Show annotated image
    try:
        out_path = tmp_path.replace(".jpg", "_annotated.jpg")
        draw_detections(tmp_path, current_model_path, out_path)
        st.image(out_path, caption="Detected items with bounding boxes", use_container_width=True)
    except Exception:
        pass

    for item in detected:
        st.write(f"🥦 **{item['name'].title()}** —  {item['confidence']:.0%} conf  —  count: {item['count']}")

    update_from_detection(detected)
    st.info("📦 Fridge updated! Switch to 'My Fridge' tab.")
    os.unlink(tmp_path)

# ── UI CONFIG & HEADER ──
st.set_page_config(page_title="Smart Recipe Generator", page_icon="🍳", layout="wide")
apply_custom_style()

col_head1, col_head2 = st.columns([1, 8])
with col_head1:
    st.markdown("# 🍳")
with col_head2:
    st.title("Smart Recipe Generator")
    st.markdown("<p style='color: #666; margin-top: -15px;'>Your Multimodal AI Kitchen Companion</p>", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Cooking Dashboard")
    dietary_pref = st.multiselect("Dietary Restrictions", ["vegetarian", "vegan", "halal"], default=["halal"])
    allergy_list = st.multiselect("Avoid Allergens", ["gluten", "dairy", "nuts", "eggs"], default=[])
    cooking_limit = st.slider("Max Prep Time (mins)", 5, 120, 45)
    user_prefs = {"dietary": dietary_pref, "allergies": allergy_list, "max_time": cooking_limit}
    
    st.divider()
    st.markdown("### 🧠 Model Settings")

    import os

    # 1. Find exactly where this streamlit_app.py file lives
    current_dir = os.path.dirname(__file__)

    # 2. Go up one folder (..), then into model1_detector, then grab best.pt
    default_model_path = os.path.join(current_dir, "..", "model1_detector", "best.pt")

    # 3. Use that dynamically built path in your app
    model_path = st.text_input("Path to best.pt", value=default_model_path, help="Path to your best.pt file")

    #model_path = st.text_input("Path to best.pt", value="model1_detector/best.pt", help="Path to your best.pt file")
    st.caption("AI Agent filters recipes based on these settings.")

# ── TABS ──
tab_scan, tab_fridge, tab_recipes = st.tabs(["📷 Scan Ingredients", "🧊 My Fridge", "🍽️ Smart Recommendations"])

# ════════════════ TAB 1: SCAN ════════════════
with tab_scan:
    st.markdown("### 📷 Add ingredients to your fridge")
    mode = st.radio("Choose input method:", ["📸 Live Camera", "🖼️ Upload Image", "✏️ Manual"], horizontal=True)

    if mode == "📸 Live Camera":
        st.info("Works on phone browser too — opens front or back camera.")
        camera_photo = st.camera_input("📷 Capture photo")
        if camera_photo:
            image = Image.open(camera_photo)
            if st.button("🔍 Detect Ingredients", type="primary", key="cam_btn"):
                with st.spinner("Model scanning your photo..."):
                    run_detection_and_show(image, model_path)

    elif mode == "🖼️ Upload Image":
        st.info("Upload any photo — fridge, table of ingredients, shopping bag.")
        uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Your photo", use_container_width=True)
            if st.button("🔍 Detect Ingredients", type="primary", key="upload_btn"):
                with st.spinner("Model scanning your photo..."):
                    run_detection_and_show(image, model_path)

    elif mode == "✏️ Manual":
        st.info("Select ingredients from the list. Useful for items the camera misses.")
        manual_items = st.multiselect("Select what you have:", manual_mode_samples)
        qty = st.number_input("Quantity for each selected item", min_value=1, max_value=20, value=1)
        if st.button("➕ Add to Fridge", type="primary"):
            if manual_items:
                manual_detected = [{"name": item.lower(), "confidence": 1.0, "count": int(qty)} for item in manual_items]
                update_from_detection(manual_detected)
                st.success(f"Added: {', '.join(manual_items)}")
                st.balloons()
            else:
                st.warning("Select at least one ingredient.")

# ════════════════ TAB 2: FRIDGE ════════════════
with tab_fridge:
    st.markdown("### 🧊 Digital Inventory")
    current_stock = load_inventory()
    
    if current_stock:
        display_cols = st.columns(4)
        for idx, item in enumerate(current_stock.values()):
            with display_cols[idx % 4]:
                st.markdown(f"""
                   <div style="background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px; border: 1px solid var(--faded-text-color); text-align: center; margin-bottom: 10px;">
                        <span style="font-size: 0.8em; color: var(--text-color); opacity: 0.7;">Ingredient</span><br>
                        <b style="color: var(--text-color);">{item['name'].replace('_', ' ').title()}</b><br>
                        <span style="color: #FF4B4B; font-weight: bold;">x{item['count']}</span>
                        <br><span style="font-size: 0.7em; color: var(--text-color); opacity: 0.5;">{item.get('confidence', 1.0):.0%} conf</span>
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        items_to_del = st.multiselect("Select items you've used:", list(current_stock.keys()))
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Remove Selected"):
                for entry in items_to_del: remove_item(entry)
                st.rerun()
        with col_b:
            if st.button("🧹 Clear Entire Fridge"):
                for entry in list(current_stock.keys()): remove_item(entry)
                st.rerun()
    else:
        st.info("Fridge is empty. Start by scanning ingredients!")

# ════════════════ TAB 3: RECIPES ════════════════
with tab_recipes:
    available_ingredients = get_ingredient_names()
    
    if available_ingredients:
        engine = RecipeRecommender()
        top_matches = engine.recommend(available_ingredients, user_prefs, top_n=5)
        
        if top_matches:
            st.success(f"{len(top_matches)} recipes found!")
            for rank, rec in enumerate(top_matches, 1):
                match_pct = rec.get("match_percent", 0)
                missing = rec.get("missing_ingredients", [])
                
                with st.expander(f"{'✅' if not missing else '🔶'} Option #{rank}: {rec['name']} ({match_pct}% Match)"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.write(f"**Brief:** {rec['description']}")
                        st.write(f"**Tags:** {', '.join(rec['tags'])}")
                        if missing:
                            st.warning(f"Missing: {', '.join(missing)}")
                    with c2:
                        st.metric("⏱ Prep Time", f"{rec['time_minutes']} min")
                    
                    # Generate Button
                    if st.button(f"🚀 Generate Full Guide", key=f"gen_{rec['id']}"):
                        with st.spinner("AI is thinking..."):
                            agent_output = run_recipe_agent(rec, available_ingredients)
                            st.session_state[f"active_recipe_{rec['id']}"] = agent_output

                    # Display Saved Recipe Output
                    storage_id = f"active_recipe_{rec['id']}"
                    if storage_id in st.session_state:
                        data = st.session_state[storage_id]
                        st.markdown("---")
                        st.markdown("#### 📖 Cooking Methodology")
                        st.markdown(data["recipe_text"])
                        
                        # Audio Button
                        if st.button("🔊 Play Audio Guide", key=f"audio_btn_{rec['id']}"):
                            play_recipe_audio(data["recipe_text"])
                        
                        st.markdown("---")
                        st.write("#### 📊 Nutritional Overview")
                        nutri = data["nutrition"]
                        nc = st.columns(4)
                        nc[0].metric("🔥 Calories", f"{nutri.get('calories','?')} kcal")
                        nc[1].metric("💪 Protein", f"{nutri.get('protein_g','?')}g")
                        nc[2].metric("🍞 Carbs", f"{nutri.get('carbs_g','?')}g")
                        nc[3].metric("🥑 Fats", f"{nutri.get('fat_g','?')}g")
                        st.caption(f"Source: {nutri.get('source','estimated')}")
                        
                        # Rating System
                        st.divider()
                        rating = st.select_slider(
                            "Rate this recipe:",
                            options=["⭐","⭐⭐","⭐⭐⭐","⭐⭐⭐⭐","⭐⭐⭐⭐⭐"],
                            value="⭐⭐⭐",
                            key=f"rate_{rec['id']}",
                        )
                        if st.button("Submit rating", key=f"submit_{rec['id']}"):
                            ratings_path = "data/fridge_inventory/ratings.json"
                            try:
                                with open(ratings_path) as f: ratings_data = json.load(f)
                            except Exception:
                                ratings_data = {}
                            ratings_data[str(rec["id"])] = len(rating)
                            os.makedirs(os.path.dirname(ratings_path), exist_ok=True)
                            with open(ratings_path, "w") as f: json.dump(ratings_data, f, indent=2)
                            st.success("Rating saved!")
        else:
            st.warning("No matches found. Try relaxing dietary filters or adding more ingredients.")
    else:
        st.warning("No ingredients found in fridge. Scan first!")