# ============================================================
# 1_explore.py — Step 1: Explore & Understand the Dataset
# ============================================================
# Run this first to understand what data you're working with.
# Command: python 1_explore.py
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Load the dataset ──────────────────────────────────────
df = pd.read_csv("iris.csv")

print("=" * 50)
print("🌸  IRIS DATASET EXPLORATION")
print("=" * 50)

# ── 2. Basic info ────────────────────────────────────────────
print(f"\n📦 Shape of dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n📋 First 5 rows:")
print(df.head())

print("\n📊 Column names & data types:")
print(df.dtypes)

# ── 3. Check for missing values ──────────────────────────────
print("\n❓ Missing values per column:")
print(df.isnull().sum())

# ── 4. Class distribution ────────────────────────────────────
print("\n🌿 Class distribution (how many samples per flower type):")
print(df["species"].value_counts())

# ── 5. Statistical summary ───────────────────────────────────
print("\n📈 Statistical summary (mean, min, max, etc.):")
print(df.describe())

# ── 6. Visualize the data ────────────────────────────────────
print("\n📊 Generating plots... (close the window to continue)")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Iris Dataset — Feature Distributions by Species", fontsize=14)

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
colors   = {"setosa": "red", "versicolor": "green", "virginica": "blue"}

for ax, feature in zip(axes.flat, features):
    for species, color in colors.items():
        subset = df[df["species"] == species]
        ax.hist(subset[feature], alpha=0.6, label=species, color=color, bins=15)
    ax.set_title(feature.replace("_", " ").title())
    ax.set_xlabel("Value (cm)")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("exploration_plot.png")
plt.show()
print("✅ Plot saved as 'exploration_plot.png'")
print("\n✅ Exploration complete! Move on to: python 2_train.py")