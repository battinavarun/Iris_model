# ============================================================
# 3_predict.py — Step 3: Make Predictions with the Saved Model
# ============================================================
# This script loads the saved model and makes predictions.
# Great for testing before deploying!
#
# Command: python 3_predict.py
# ============================================================

import pickle
import pandas as pd

print("=" * 50)
print("🔮  LOCAL PREDICTION TESTER")
print("=" * 50)

# ── 1. Load the saved model ───────────────────────────────────
print("\n📂 Loading saved model (model.pkl)...")
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print("   ✅ Model loaded!")

# ── 2. Define some test flowers ──────────────────────────────
# These are [sepal_length, sepal_width, petal_length, petal_width]
test_samples = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},  # likely setosa
    {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},  # likely versicolor
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1},  # likely virginica
]

print("\n🌸 Making predictions on test samples:\n")
print(f"{'#':<4} {'Sepal L':<10} {'Sepal W':<10} {'Petal L':<10} {'Petal W':<10} {'Predicted':<14} {'Confidence'}")
print("-" * 72)

for i, sample in enumerate(test_samples, 1):
    # Convert sample to DataFrame (model expects this format)
    X_new = pd.DataFrame([sample])

    # Get prediction
    prediction = model.predict(X_new)[0]

    # Get confidence (probability of the predicted class)
    probabilities = model.predict_proba(X_new)[0]
    confidence = max(probabilities) * 100

    print(
        f"{i:<4} {sample['sepal_length']:<10} {sample['sepal_width']:<10} "
        f"{sample['petal_length']:<10} {sample['petal_width']:<10} "
        f"{prediction:<14} {confidence:.1f}%"
    )

# ── 3. Interactive mode ───────────────────────────────────────
print("\n" + "=" * 50)
print("🎮 Interactive Mode — Enter your own flower measurements!")
print("   (Press Ctrl+C to exit)")
print("=" * 50)

while True:
    try:
        print()
        sl = float(input("   Sepal Length (e.g. 5.1): "))
        sw = float(input("   Sepal Width  (e.g. 3.5): "))
        pl = float(input("   Petal Length (e.g. 1.4): "))
        pw = float(input("   Petal Width  (e.g. 0.2): "))

        X_input = pd.DataFrame([{
            "sepal_length": sl,
            "sepal_width":  sw,
            "petal_length": pl,
            "petal_width":  pw
        }])

        prediction  = model.predict(X_input)[0]
        probs       = model.predict_proba(X_input)[0]
        confidence  = max(probs) * 100
        classes     = model.classes_

        print(f"\n   🌸 Prediction : {prediction.upper()}")
        print(f"   📊 Confidence : {confidence:.1f}%")
        print(f"   📈 All probabilities:")
        for cls, prob in zip(classes, probs):
            bar = "█" * int(prob * 30)
            print(f"      {cls:<14} {prob*100:5.1f}%  {bar}")

    except KeyboardInterrupt:
        print("\n\n✅ Exiting. Move on to: python app.py")
        break
    except ValueError:
        print("   ⚠️  Please enter valid numbers!")