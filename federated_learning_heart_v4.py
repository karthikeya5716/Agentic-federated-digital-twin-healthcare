"""
Phase 2: Federated Learning Simulation - Heart Disease (Improved FedAvg)
-----------------------------------------------------------------------
Purpose:
Simulate 5 hospitals holding private, non-overlapping training data.
Each hospital trains locally and sends only model weights to the server.
The server performs weighted FedAvg.

Evaluation design:
- Keep the SAME 20% held-out test split as the centralized baseline.
- Split the remaining 80% into local-training (80%) and validation (20%).
- Use validation performance to select the best global round.
- Use the untouched test set only once for final reporting.

This avoids choosing the final round using the test set.
"""

import copy
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =====================================================
# Reproducibility
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# 1. Load & clean data (same source as Phase 1)
# =====================================================
url = (
    "https://raw.githubusercontent.com/dataspelunking/MLwR/master/"
    "Machine%20Learning%20with%20R%20(2nd%20Edition)/Chapter%2005/heart.csv"
)

try:
    df = pd.read_csv(url)
except Exception:
    url_fallback = (
        "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/"
        "master/dataset.csv"
    )
    df = pd.read_csv(url_fallback)

target_candidates = ["target", "num", "condition", "output"]
target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col is None:
    raise ValueError(
        f"Could not find target column. Available columns: {df.columns.tolist()}"
    )

if target_col != "target":
    df = df.rename(columns={target_col: "target"})

if df["target"].nunique() > 2:
    df["target"] = (df["target"] > 0).astype(int)

df = df.dropna().drop_duplicates()

X = df.drop("target", axis=1).values.astype(np.float32)
y = df["target"].values.astype(np.float32)

# =====================================================
# 2. Keep the same 20% test split as centralized baseline
# =====================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Validation split comes ONLY from the original training data.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=43,
    stratify=y_train_full,
)

# Fit scaler ONLY on local-training data.
# This avoids using validation/test feature statistics during training.
# Keep feature names when fitting/scaling so prediction-time DataFrames
# use the same schema without sklearn feature-name warnings.
feature_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X_train_df = pd.DataFrame(X_train, columns=feature_columns)
X_val_df = pd.DataFrame(X_val, columns=feature_columns)
X_test_df = pd.DataFrame(X_test, columns=feature_columns)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_val = scaler.transform(X_val_df)
X_test = scaler.transform(X_test_df)

# =====================================================
# 3. Federated configuration
# =====================================================
NUM_CLIENTS = 5
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3
LOCAL_LR = 0.005
BATCH_SIZE = 16
INPUT_DIM = X.shape[1]

# =====================================================
# 4. Split private training data across hospitals
# =====================================================
def split_among_clients(X_data, y_data, num_clients):
    idx = np.arange(len(X_data))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    splits = np.array_split(idx, num_clients)
    return [(X_data[s], y_data[s]) for s in splits]


client_data = split_among_clients(X_train, y_train, NUM_CLIENTS)

print("===== HOSPITAL DATA SPLITS =====")
for i, (cx, cy) in enumerate(client_data):
    print(
        f"Hospital {i + 1}: {len(cx)} patients, "
        f"{int(cy.sum())} disease cases"
    )

print(
    f"\nTraining patients: {len(X_train)} | "
    f"Validation patients: {len(X_val)} | "
    f"Test patients: {len(X_test)}"
)

# =====================================================
# 5. Neural network
# =====================================================
class HeartNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_weights(model):
    return copy.deepcopy(model.state_dict())


def set_weights(model, weights):
    model.load_state_dict(weights)


# =====================================================
# 6. Local training
# =====================================================
def train_local(model, X_local, y_local, epochs=LOCAL_EPOCHS, lr=LOCAL_LR):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_local, dtype=torch.float32)
    y_t = torch.tensor(y_local, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model


# =====================================================
# 7. Weighted FedAvg
# =====================================================
def average_weights(weighted_states):
    """
    Weighted FedAvg:
    each hospital's model contributes according to its number
    of local training samples.
    """
    total_samples = sum(num_samples for _, num_samples in weighted_states)

    avg = OrderedDict()

    for key in weighted_states[0][0].keys():
        avg[key] = sum(
            state[key].float() * (num_samples / total_samples)
            for state, num_samples in weighted_states
        )

    return avg


# =====================================================
# 8. Evaluation
# =====================================================
def evaluate(model, X_data, y_data):
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_data, dtype=torch.float32)
        probs = model(X_t).numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_data, preds)
    prec = precision_score(y_data, preds, zero_division=0)
    rec = recall_score(y_data, preds, zero_division=0)
    f1 = f1_score(y_data, preds, zero_division=0)

    return acc, prec, rec, f1


# =====================================================
# 9. Federated training
# =====================================================
global_model = HeartNet(INPUT_DIM)
global_weights = get_weights(global_model)

best_global_weights = copy.deepcopy(global_weights)
best_val_f1 = -1.0
best_round = 0

round_results = []

print(
    f"\n===== STARTING IMPROVED FEDERATED TRAINING: "
    f"{NUM_CLIENTS} hospitals, {NUM_ROUNDS} rounds =====\n"
)

for round_num in range(1, NUM_ROUNDS + 1):
    weighted_local_states = []

    for client_idx, (X_c, y_c) in enumerate(client_data):
        local_model = HeartNet(INPUT_DIM)
        set_weights(local_model, global_weights)

        train_local(local_model, X_c, y_c)

        weighted_local_states.append(
            (get_weights(local_model), len(X_c))
        )

    # Server aggregates only model weights.
    global_weights = average_weights(weighted_local_states)
    set_weights(global_model, global_weights)

    val_acc, val_prec, val_rec, val_f1 = evaluate(
        global_model, X_val, y_val
    )

    round_results.append(
        {
            "round": round_num,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
        }
    )

    print(
        f"Round {round_num:2d}/{NUM_ROUNDS} | "
        f"Val Accuracy: {val_acc:.4f} | "
        f"Val Recall: {val_rec:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    # Model selection uses validation only.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_round = round_num
        best_global_weights = copy.deepcopy(global_weights)

# =====================================================
# 10. Final evaluation on untouched test set
# =====================================================
set_weights(global_model, best_global_weights)

final_acc, final_prec, final_rec, final_f1 = evaluate(
    global_model, X_test, y_test
)

cm = confusion_matrix(y_test, (global_model(
    torch.tensor(X_test, dtype=torch.float32)
).detach().numpy().flatten() >= 0.5).astype(int))

print("\n===== IMPROVED FEDERATED LEARNING RESULTS =====")
print(f"Best validation round: {best_round}")
print(f"Accuracy:  {final_acc:.4f}")
print(f"Precision: {final_prec:.4f}")
print(f"Recall:    {final_rec:.4f}")
print(f"F1 Score:  {final_f1:.4f}")

print("\n===== CENTRALIZED VS FEDERATED =====")
print("Centralized Baseline : Accuracy 0.8033 | Recall 0.9394")
print(
    f"Federated (5 hospitals): Accuracy {final_acc:.4f} | "
    f"Recall {final_rec:.4f}"
)

# =====================================================
# 11. Save results
# =====================================================
with open("federated_results_heart.txt", "w") as f:
    f.write("IMPROVED FEDERATED LEARNING - 5 HOSPITALS (Weighted FedAvg)\n")
    f.write(f"Best validation round: {best_round}\n")
    f.write(f"Accuracy: {final_acc:.4f}\n")
    f.write(f"Precision: {final_prec:.4f}\n")
    f.write(f"Recall: {final_rec:.4f}\n")
    f.write(f"F1 Score: {final_f1:.4f}\n")

pd.DataFrame(round_results).to_csv(
    "federated_round_results_heart.csv",
    index=False,
)

torch.save(best_global_weights, "federated_global_model_heart.pt")
joblib.dump(scaler, "federated_scaler_heart.pkl")

print(
    "\nSaved: federated_results_heart.txt, "
    "federated_round_results_heart.csv, "
    "federated_global_model_heart.pt, "
    "federated_scaler_heart.pkl"
)