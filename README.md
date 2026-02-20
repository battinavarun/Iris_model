# 🌸 Iris ML Project — Beginner's Guide

## Project Structure
```
iris_ml_project/
│
├── iris.csv                  # Dataset
├── 1_explore.py              # Step 1: Explore the data
├── 2_train.py                # Step 2: Train & save the model
├── 3_predict.py              # Step 3: Make predictions locally
├── app.py                    # Step 4: Flask web API
├── requirements.txt          # All dependencies
└── README.md                 # This file
```

---

## 🚀 Step-by-Step Process

### Step 1 — Set Up Your Environment
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Explore the Data
```bash
python 1_explore.py
```
This will show you what the dataset looks like, its shape, class distribution, and basic stats.

### Step 3 — Train the Model
```bash
python 2_train.py
```
This trains a Random Forest classifier on the Iris dataset and saves the model as `model.pkl`.

### Step 4 — Test Predictions Locally
```bash
python 3_predict.py
```
This loads the saved model and makes a sample prediction.

### Step 5 — Run the Web API
```bash
python app.py
```
Then open your browser or use a tool like Postman to send a POST request to:
`http://127.0.0.1:5000/predict`

#### Example Request (JSON):
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Example Response:
```json
{
  "prediction": "setosa",
  "confidence": 0.97
}
```

---

## 🌐 Deploy to the Cloud (Free — Render.com)

1. Push your project to a GitHub repository
2. Go to https://render.com and sign up free
3. Click **New → Web Service**
4. Connect your GitHub repo
5. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
6. Click **Deploy** — your API will be live in minutes!

---

## 🧠 How the ML Pipeline Works

```
Raw Data (iris.csv)
       ↓
Explore & Understand (1_explore.py)
       ↓
Split into Train/Test sets (80/20)
       ↓
Train Random Forest Model (2_train.py)
       ↓
Evaluate Accuracy on Test Set
       ↓
Save Model to disk (model.pkl)
       ↓
Load Model & Serve via Flask API (app.py)
       ↓
Send HTTP Request → Get Prediction
```