# ============================================================
# app.py — Step 4: Flask Web API (for Deployment)
# ============================================================
# This turns your ML model into a web API that anyone can
# call from anywhere using HTTP requests.
#
# Command: python app.py
# Then visit: http://127.0.0.1:5000
# ============================================================

import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# ── Create Flask app ──────────────────────────────────────────
app = Flask(__name__)

# ── Load the saved model at startup ──────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ── HTML homepage (a simple UI to test the API) ───────────────
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🌸 Iris Flower Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; padding: 20px; background: #f5f5f5; }
        h1   { color: #6b4c8e; }
        input { width: 100%; padding: 8px; margin: 6px 0 14px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { background: #6b4c8e; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #4a3065; }
        #result { margin-top: 24px; padding: 16px; border-radius: 8px; display: none; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        label { font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h1>🌸 Iris Flower Predictor</h1>
    <p>Enter flower measurements to predict the species:</p>

    <label>Sepal Length (cm):</label>
    <input type="number" id="sepal_length" placeholder="e.g. 5.1" step="0.1">

    <label>Sepal Width (cm):</label>
    <input type="number" id="sepal_width" placeholder="e.g. 3.5" step="0.1">

    <label>Petal Length (cm):</label>
    <input type="number" id="petal_length" placeholder="e.g. 1.4" step="0.1">

    <label>Petal Width (cm):</label>
    <input type="number" id="petal_width" placeholder="e.g. 0.2" step="0.1">

    <br>
    <button onclick="predict()">🔮 Predict Species</button>

    <div id="result"></div>

    <script>
        async function predict() {
            const data = {
                sepal_length: parseFloat(document.getElementById('sepal_length').value),
                sepal_width:  parseFloat(document.getElementById('sepal_width').value),
                petal_length: parseFloat(document.getElementById('petal_length').value),
                petal_width:  parseFloat(document.getElementById('petal_width').value)
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const div = document.getElementById('result');
            div.style.display = 'block';
            div.className = 'success';
            div.innerHTML = `
                <h3>🌸 Prediction: <strong>${result.prediction.toUpperCase()}</strong></h3>
                <p>Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong></p>
                <p>Probabilities: ${Object.entries(result.probabilities)
                    .map(([k,v]) => `${k}: ${(v*100).toFixed(1)}%`).join(' | ')}</p>
            `;
        }
    </script>
</body>
</html>
"""


# ── Route 1: Homepage ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template_string(HOME_HTML)


# ── Route 2: Health check ─────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Simple health check — useful for deployment platforms."""
    return jsonify({"status": "ok", "model": "Random Forest Iris Classifier"})


# ── Route 3: Prediction API ───────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with flower measurements and returns predicted species.

    Request body (JSON):
    {
        "sepal_length": 5.1,
        "sepal_width":  3.5,
        "petal_length": 1.4,
        "petal_width":  0.2
    }

    Response:
    {
        "prediction": "setosa",
        "confidence": 0.97,
        "probabilities": { "setosa": 0.97, "versicolor": 0.02, "virginica": 0.01 }
    }
    """
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate that all required fields are present
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Convert to DataFrame (model expects this format)
        X_new = pd.DataFrame([{
            "sepal_length": float(data["sepal_length"]),
            "sepal_width":  float(data["sepal_width"]),
            "petal_length": float(data["petal_length"]),
            "petal_width":  float(data["petal_width"])
        }])

        # Make prediction
        prediction  = model.predict(X_new)[0]
        probs       = model.predict_proba(X_new)[0]
        confidence  = float(max(probs))

        # Build probability dict
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probs)}

        return jsonify({
            "prediction":    prediction,
            "confidence":    round(confidence, 4),
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run the server ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Flask server...")
    print("   Open your browser at: http://127.0.0.1:5000")
    print("   API endpoint:          http://127.0.0.1:5000/predict  (POST)")
    print("   Health check:          http://127.0.0.1:5000/health   (GET)")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, host="0.0.0.0", port=5000)