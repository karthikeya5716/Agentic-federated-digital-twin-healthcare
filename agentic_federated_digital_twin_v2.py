"""
Phase 4/5: Federated Global Model + Digital Twin + Agentic AI
---------------------------------------------------------------
Uses the trained global Federated Learning model for risk prediction,
then sends the patient's Digital Twin state + model explanation factors
to Gemini for personalized, non-diagnostic recommendations.

Required files in the same folder:
- digital_twin.py
- federated_global_model_heart.pt
- federated_scaler_heart.pkl
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import joblib
from google import genai

from digital_twin import get_twin_state, add_prediction, add_recommendation


# =====================================================
# API KEY
# =====================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Set it in the terminal before running this script."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.6-flash"


# =====================================================
# Model configuration
# =====================================================
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age": "age in years",
    "sex": "sex (1 = male, 0 = female)",
    "cp": "chest pain type (0-3)",
    "trestbps": "resting blood pressure (mm Hg)",
    "chol": "serum cholesterol (mg/dl)",
    "fbs": "fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)",
    "restecg": "resting ECG result (0-2)",
    "thalach": "maximum heart rate achieved during exercise",
    "exang": "exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "slope of peak exercise ST segment (0-2)",
    "ca": "number of major vessels colored by fluoroscopy (0-3)",
    "thal": "thalassemia test result (1=normal, 2=fixed defect, 3=reversible defect)",
}


# =====================================================
# Same architecture as federated_learning_heart_v2.py
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


# =====================================================
# Load the federated global model + scaler
# =====================================================
def load_federated_model():
    model_path = "federated_global_model_heart.pt"
    scaler_path = "federated_scaler_heart.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run federated_learning_heart_v3.py first."
        )

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"{scaler_path} not found. Run federated_learning_heart_v3.py first."
        )

    model = HeartNet(len(FEATURE_NAMES))
    weights = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(weights)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler


# =====================================================
# Predict using the federated global model
# =====================================================
def predict_federated_risk(model, scaler, combined_record):
    missing = [
        f for f in FEATURE_NAMES
        if f not in combined_record or combined_record[f] is None
    ]

    if missing:
        raise ValueError(f"Missing required fields for prediction: {missing}")

    row = [combined_record[f] for f in FEATURE_NAMES]
    row_df = pd.DataFrame([row], columns=FEATURE_NAMES)
    row_scaled = scaler.transform(row_df)

    with torch.no_grad():
        probability = float(
            model(torch.tensor(row_scaled, dtype=torch.float32))
            .item()
        )

    if probability < 0.33:
        label = "low"
    elif probability < 0.66:
        label = "moderate"
    else:
        label = "high"

    return probability, label


# =====================================================
# Explainability for the Agentic layer
# =====================================================
def calculate_factor_context(combined_record):
    """
    The federated neural network does not expose Random Forest
    feature_importances_. Therefore we provide the patient's values
    and domain descriptions to Gemini instead of pretending that
    these are neural-network feature importance scores.
    """
    return "\n".join(
        f"- {FEATURE_DESCRIPTIONS[f]}: {combined_record[f]}"
        for f in FEATURE_NAMES
    )


# =====================================================
# Gemini Agent
# =====================================================
def generate_agentic_recommendation(
    combined_record,
    lifestyle_record,
    risk_score,
    risk_label
):
    factor_lines = calculate_factor_context(combined_record)

    lifestyle_line = (
        f"Steps: {lifestyle_record.get('steps', 'N/A') if lifestyle_record else 'N/A'} daily; "
        f"sleep: {lifestyle_record.get('sleep_hours', 'N/A') if lifestyle_record else 'N/A'} hours; "
        f"resting HR: {lifestyle_record.get('resting_hr', 'N/A') if lifestyle_record else 'N/A'}; "
        f"diet quality: {lifestyle_record.get('diet_quality', 'N/A') if lifestyle_record else 'N/A'}."
    )

    prompt = f"""You are a clinical decision-support AI assisting with heart disease
risk communication.

You are NOT a doctor and must not diagnose disease. Explain the model prediction
and provide general, non-diagnostic lifestyle guidance.

Patient age: {combined_record['age']}
Model source: Federated global model trained with FedAvg across 5 simulated hospitals.
Predicted heart disease risk: {risk_score:.1%} ({risk_label} risk)

Patient clinical values:
{factor_lines}

Patient lifestyle data:
{lifestyle_line}

Return exactly these sections:

ANALYZE:
2-3 sentences explaining the most relevant clinical/lifestyle factors visible
in this patient's data and how they relate to the model's risk result.

RECOMMEND:
Exactly 3 specific, actionable, non-diagnostic lifestyle recommendations.
Tie each recommendation to a relevant patient factor.

EXPLAIN:
1-2 sentences explaining why these recommendations were selected.

Keep the response under 200 words.
Do not diagnose the patient.
End by advising consultation with a licensed physician for medical decisions.
"""

    chat = client.chats.create(model=GEMINI_MODEL)
    response = chat.send_message(prompt)

    return response.text


# =====================================================
# Complete integrated pipeline
# =====================================================
def run_agent_for_patient(patient_id):
    print(
        f"\n===== FEDERATED + DIGITAL TWIN + AGENTIC AI "
        f"FOR PATIENT {patient_id} =====\n"
    )

    twin = get_twin_state(patient_id)
    patient = twin["patient"]
    clinical = twin["latest_clinical"]
    lifestyle = twin["latest_lifestyle"]

    if clinical is None:
        print("No clinical record found for this patient. Add one first.")
        return

    combined_record = {**patient, **clinical}

    print("Loading federated global model...")
    model, scaler = load_federated_model()

    print("Predicting risk from Digital Twin...")
    risk_score, risk_label = predict_federated_risk(
        model,
        scaler,
        combined_record
    )

    print(f"Federated model risk: {risk_score:.1%} ({risk_label})")

    add_prediction(patient_id, risk_score, risk_label)
    print("Saved federated prediction to digital twin.")

    print("\nGenerating personalized recommendation via Gemini...")
    recommendation_text = generate_agentic_recommendation(
        combined_record,
        lifestyle,
        risk_score,
        risk_label
    )

    print("\n===== AGENT OUTPUT =====")
    print(recommendation_text)

    recommendation_section = recommendation_text
    explanation_section = "See full agent output above."

    if "RECOMMEND:" in recommendation_text:
        recommendation_section = recommendation_text.split(
            "RECOMMEND:", 1
        )[1]

        if "EXPLAIN:" in recommendation_section:
            recommendation_section, explanation_section = (
                recommendation_section.split("EXPLAIN:", 1)
            )

    add_recommendation(
        patient_id,
        recommendation_text=recommendation_section.strip(),
        explanation_text=explanation_section.strip()
    )

    print("\nSaved recommendation to digital twin.")
    print("\n===== INTEGRATED PIPELINE COMPLETE =====")


if __name__ == "__main__":
    run_agent_for_patient(patient_id=1)