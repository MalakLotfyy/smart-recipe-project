# 🍳 Smart Recipe Generator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)

An intelligent, multimodal web application that helps you figure out what to cook based on the ingredients you already have. By combining a custom-trained computer vision model with an interactive recommendation engine, this app acts as your personal digital sous-chef.

**[🌐 View the Live App Here](https://smart-recipe-project.streamlit.app/))** 

---

## ✨ Key Features

* **📷 Multimodal Ingredient Scanning:** * **Live Camera:** Scan real-world ingredients instantly using your webcam or phone camera.
  * **Image Upload:** Upload photos of your fridge or countertop.
  * **Manual Entry:** Quickly add pantry staples using a comprehensive, searchable dropdown menu.

* **🧠 Custom Object Detection (YOLOv8):** Powered by a custom-trained AI model (`best.pt`) capable of identifying 37 distinct classes of fruits, vegetables, and pantry items with high confidence.

* **🧊 Digital Fridge Inventory:** A dynamic inventory system that automatically logs your scanned ingredients, updates quantities, and allows you to clear items as you use them.

* **🍽️ Smart Recipe Engine:** Recommends recipes based strictly on your available inventory, with customizable filters for:
  * Maximum Prep Time
  * Dietary Restrictions (Vegetarian, Vegan, Halal)
  * Allergen Avoidance

* **📺 Interactive Cooking Guides:** * **Automated YouTube Integration:** Dynamically searches and embeds the perfect step-by-step cooking video for your chosen recipe.
  * **Text-to-Speech (TTS):** Includes an audio player that reads the cooking methodology out loud so you can listen while you chop.
  * **Nutritional Overview:** Provides a high-level breakdown of calories, protein, carbs, and fats.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Frontend & UI** | Streamlit, Custom CSS (Adaptive Dark/Light Mode) |
| **Computer Vision** | Ultralytics YOLOv8, OpenCV, Pillow |
| **Backend Logic** | Python, Pandas, JSON data handling |
| **Integrations** | gTTS (Google Text-to-Speech), `urllib` (YouTube web scraping) |

---

## 🚀 How to Run Locally

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/smart_recipe_project.git](https://github.com/YOUR_GITHUB_USERNAME/smart_recipe_project.git)
cd smart_recipe_project
2. Install dependencies
Ensure you have Python installed, then run:

Bash
pip install -r requirements.txt
(Note: Make sure your requirements file includes streamlit, ultralytics, opencv-python, Pillow, gTTS, and pandas)

3. Verify the AI Model
Ensure that the custom-trained weights file (best.pt) is located in the correct directory:
model1_detector/best.pt

4. Launch the App
Bash
streamlit run app/streamlit_app.py
The application will open automatically in your default web browser at http://localhost:8501.
or from the cloud -> https://smart-recipe-project.streamlit.app/
```
---

📂 Project Structure
Plaintext
smart_recipe_project/
├── app/
│   └── streamlit_app.py         # Main Streamlit application and UI logic
├── model1_detector/
│   ├── best.pt                  # Custom YOLOv8 weights (37 classes)
│   ├── detect_ingredients.py    # Inference script and bounding box logic
│   └── update_inventory.py      # JSON state management for the fridge
├── model2_recommender/
│   ├── recommender.py           # Recipe matching and filtering algorithms
│   ├── llm_agent.py             # Methodology and nutrition generation
│   └── recipes.json             # Recipe database
├── data/
│   └── fridge_inventory/        # Stores dynamic user inventory and ratings
├── .gitignore                   # Ignores heavy datasets and cached runs
└── README.md

---
🧠 Model Training Notes
The object detection model was trained using Ultralytics YOLOv8 on a curated dataset of over 4,000 images, optimized down to 37 specific ingredient classes for high accuracy and minimal false positives. The dataset and training runs are excluded from this repository to maintain a lightweight, deployable codebase.
