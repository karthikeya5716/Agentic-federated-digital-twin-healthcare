"""
Phase 1: Centralized Baseline Model - Heart Disease
------------------------------------------------------
Purpose: Train a single model on ALL patient data (no federation) to get a
baseline accuracy. Later, federated learning results are compared against
this number to demonstrate the privacy-accuracy tradeoff.

Dataset: UCI Heart Disease (Cleveland subset)
Source: https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Edition)/Chapter%2005/heart.csv
(Public mirror, no login required. Also available at archive.ics.uci.edu/ml/datasets/heart+disease)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the dataset
# -----------------------------
# This mirror already has proper column headers and a binary "target" column
# (1 = heart disease present, 0 = absent), so minimal renaming is needed.
url = "https://raw.githubusercontent.com/dataspelunking/MLwR/master/Machine%20Learning%20with%20R%20(2nd%20Edition)/Chapter%2005/heart.csv"

try:
    df = pd.read_csv(url)
except Exception as e:
    print("Primary URL failed, trying UCI mirror...")
    # Fallback mirror (Kaggle-style UCI heart.csv format), commonly used in tutorials
    url_fallback = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = pd.read_csv(url_fallback)

print("Dataset shape:", df.shape)
print(df.columns.tolist())
print(df.head())

# -----------------------------
# 2. Standardize target column name
# -----------------------------
# Different mirrors name the label column differently ("target" vs "num" vs "condition")
target_candidates = ["target", "num", "condition", "output"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    raise ValueError(f"Could not find target column. Available columns: {df.columns.tolist()}")

if target_col != "target":
    df = df.rename(columns={target_col: "target"})

# Some versions encode target as 0-4 (severity); binarize to 0 = no disease, 1 = disease
if df["target"].nunique() > 2:
    df["target"] = (df["target"] > 0).astype(int)

# -----------------------------
# 3. Basic cleaning
# -----------------------------
df = df.dropna()
df = df.drop_duplicates()

# -----------------------------
# 4. Split features/target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

print("\nClass balance:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. Train baseline model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 7. Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== CENTRALIZED BASELINE RESULTS (Heart Disease) =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# -----------------------------
# 8. Save results for later comparison
# -----------------------------
with open("baseline_results_heart.txt", "w") as f:
    f.write("CENTRALIZED BASELINE - HEART DISEASE\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# -----------------------------
# 9. Confusion matrix plot
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Centralized Model - Confusion Matrix (Heart Disease)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("centralized_confusion_matrix_heart.png", dpi=150)
print("\nSaved: baseline_results_heart.txt, centralized_confusion_matrix_heart.png")

# -----------------------------
# 10. Feature importance (seeds the Agentic AI explainability layer later)
# -----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n===== FEATURE IMPORTANCE (used later for explainability) =====")
print(importances)

plt.figure(figsize=(6, 5))
importances.plot(kind="barh")
plt.title("Feature Importance - Centralized Model (Heart Disease)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("centralized_feature_importance_heart.png", dpi=150)
print("Saved: centralized_feature_importance_heart.png")