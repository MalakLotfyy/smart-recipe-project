# Smart Recipe Generator 🍳
**CSE351 — Introduction to Artificial Intelligence | Spring 2026**

## Project Overview
A two-model AI system:
- **Model 1:** YOLOv8 detects ingredients from photos → saves to fridge inventory
- **Model 2:** Recommendation engine + LLM agent → suggests recipes based on inventory + user preferences

## Project Structure
```
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

## Setup
```bash
pip install -r requirements.txt
```

## Dataset Tips
- Download pre-labeled food datasets from: https://universe.roboflow.com (search "grocery items", "fridge ingredients")
- For your own photos: use Roboflow or Label Studio to label them
- Export always in **YOLOv8 format**

## How to Run
```bash
# Step 1 — Train Model 1 (skip if using pretrained)
python model1_detector/train_yolo.py

# Step 2 — Test detection on a photo
python model1_detector/detect_ingredients.py --image path/to/photo.jpg

# Step 3 — Launch the full app
streamlit run app/streamlit_app.py
```
