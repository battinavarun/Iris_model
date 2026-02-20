# ============================================================
# 2_train.py — Step 2: Train & Save the ML Model
# ============================================================
# This script:
#   1. Loads and preprocesses the data
#   2. Splits data into training & test sets
#   3. Trains a Random Forest classifier
#   4. Evaluates accuracy
#   5. Saves the model to model.pkl
#
# Command: python 2_train.py
# ============================================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 50)
print("🤖  MODEL TRAINING")
print("=" * 50)

# ── 1. Load data ─────────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv("iris.csv")

# ── 2. Split features (X) and label (y) ──────────────────────
# X = all columns EXCEPT the species column (these are the inputs)
# y = just the species column (this is what we want to predict)
X = df.drop(columns=["species"])
y = df["species"]

print(f"   Features (X): {list(X.columns)}")
print(f"   Target   (y): species  [{', '.join(y.unique())}]")

# ── 3. Split into training and testing sets ───────────────────
# 80% of data is used for training, 20% for testing
# random_state=42 ensures the same split every time you run it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️  Train/Test split:")
print(f"   Training samples : {len(X_train)} ({len(X_train)/len(df)*100:.0f}%)")
print(f"   Testing  samples : {len(X_test)}  ({len(X_test)/len(df)*100:.0f}%)")

# ── 4. Create and train the model ────────────────────────────
# Random Forest = many decision trees working together (more accurate!)
print("\n🌲 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,   # number of trees in the forest
    random_state=42     # for reproducibility
)
model.fit(X_train, y_train)
print("   ✅ Training complete!")

# ── 5. Evaluate on the test set ──────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

print("🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=["setosa", "versicolor", "virginica"])
print(f"                setosa  versicolor  virginica")
for label, row in zip(["setosa", "versicolor", "virginica"], cm):
    print(f"   {label:<12}  {row[0]:<7} {row[1]:<11} {row[2]}")

# ── 6. Feature importance ────────────────────────────────────
print("\n⭐ Feature Importances (which features matter most):")
for feature, importance in sorted(
    zip(X.columns, model.feature_importances_), key=lambda x: -x[1]
):
    bar = "█" * int(importance * 40)
    print(f"   {feature:<15} {importance:.4f}  {bar}")

# ── 7. Save model to disk ────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Model saved as 'model.pkl'")
print("\n✅ Training complete! Move on to: python 3_predict.py")