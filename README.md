# 🍳 Smart Recipe Generator

**Smart Recipe Generator** is a multimodal AI web application designed to eliminate food waste and answer the daily question: *"What can I make with what I have?"* Instead of manually typing out ingredients, users simply snap a photo of their fridge or groceries. The app utilizes a custom-trained object detection model to build a digital inventory, ranks recipes using a content-based recommendation engine, and dynamically invents brand-new recipes on the fly if your ingredients don't match the existing database.

---

### 🧠 Core Architecture

This project utilizes a three-tier hybrid pipeline to balance speed, cost, and generative power:

**1. Computer Vision (Model 1):** * A custom **YOLOv8** model fine-tuned on a merged dataset of 53 distinct ingredient classes.

* Automatically detects, counts, and logs food items from user-uploaded images or live camera feeds directly into a persistent digital inventory (`inventory.json`).

**2. Content-Based Recommendation Engine (Model 2 - Stage 1):** * A deterministic matchmaking system built with **Scikit-Learn**.

* Uses `CountVectorizer` and **Cosine Similarity** to represent the user's fridge and the local recipe database as mathematical vectors.
* It calculates the optimal matches and applies strict filters based on user preferences (e.g., maximum prep time, dietary restrictions, and allergy avoidance).

**3. Generative AI Agent (Model 2 - Stage 2):** * An intelligent fallback system powered by the **Google GenAI SDK (Gemini 2.5 Flash)**.

* If the recommendation engine detects that the best existing recipe match is below a 50% similarity threshold, the LLM is triggered. It autonomously invents a structured, JSON-formatted recipe using *only* the available ingredients and saves it to the database, allowing the system's knowledge to grow organically over time.

---

### ✨ Key Features

* **Multimodal Input:** Add ingredients via Live Camera, Image Upload, or Manual Entry.
* **Smart Filtering:** Filter recommendations by max cooking time, vegetarian/vegan/halal diets, and specific allergens.
* **Nutritional Breakdown:** Integrated with the Spoonacular API to provide calorie, protein, carb, and fat estimates per serving.
* **Audio Cooking Guide:** Uses **gTTS (Google Text-to-Speech)** to read recipe instructions aloud for hands-free cooking.
* **Modern UI:** A sleek, responsive dashboard built entirely in **Streamlit**.

---

### 📊 Model Performance

Our custom YOLOv8 object detection model achieved excellent results over 50 epochs, trained on 53 distinct ingredient classes:

* **mAP50:** 97.8%
* **Optimal Confidence Threshold:** 0.45

---

### 📂 Project Structure

```text
smart_recipe_project/
├── model1_detector/
│   ├── dataset.yaml          ← tells YOLO where your data is
│   ├── train_yolo.py         ← trains Model 1
│   ├── detect_ingredients.py ← runs detection on a photo
│   └── update_inventory.py   ← saves detections to fridge inventory
├── model2_recommender/
│   ├── recipe_database.py    ← sample recipe database
│   ├── recommender.py        ← content-based recommendation engine
│   └── llm_agent.py          ← LLM agent that writes recipe + fetches nutrition
├── app/
│   └── streamlit_app.py      ← full UI connecting everything
├── data/
│   ├── fridge_inventory/
│   │   └── inventory.json    ← auto-updated by Model 1
│   └── my_dataset/           ← YOUR YOLO dataset goes here
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── requirements.txt

```

---

### ⚙️ Setup & Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt

```


2. **Environment Variables:**
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_gemini_key_here
SPOONACULAR_API_KEY=your_spoonacular_key_here

```



---

### 🚀 How to Run

**Step 1 — Train Model 1 (Skip if using pretrained weights)**

```bash
python model1_detector/train_yolo.py

```

**Step 2 — Test detection on a photo locally**

```bash
python model1_detector/detect_ingredients.py --image path/to/photo.jpg

```

**Step 3 — Launch the full web application**

```bash
streamlit run app/streamlit_app.py

```

---

### 🏷️ Dataset Tips

* Download pre-labeled food datasets from [Roboflow Universe](https://universe.roboflow.com) (search "grocery items", "fridge ingredients").
* For your own photos: use Roboflow or Label Studio to label them.
* Always export your dataset in **YOLOv8 format**.
