"""
Phase 3: Digital Twin - SQLite Schema and Management
---------------------------------------------------------
Purpose: Represent each patient as a continuously-updating "digital twin."
Every new piece of data (a checkup, a wearable sync, a lifestyle log) is
appended as a new record with a timestamp - nothing overwrites history,
so the twin can show trends over time, not just a single snapshot.

Tables:
  patients          - static identity info (set once)
  clinical_records  - clinical/medical data over time (maps to the 13
                       heart disease model features)
  lifestyle_records  - wearable + lifestyle data over time
  predictions        - model's risk output over time
  recommendations     - agentic AI's advice + explanation over time

Run this file directly to create the database and insert a demo patient
with a few simulated updates, so you can see the twin "evolve."
"""

import sqlite3
from datetime import datetime, timedelta
import json

DB_PATH = "digital_twin.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            sex INTEGER,  -- 1 = male, 0 = female (matches dataset encoding)
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS clinical_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            cp INTEGER, trestbps REAL, chol REAL, fbs INTEGER,
            restecg INTEGER, thalach REAL, exang INTEGER,
            oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS lifestyle_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            steps INTEGER,
            sleep_hours REAL,
            resting_hr REAL,
            diet_quality TEXT,  -- e.g. "poor", "average", "good"
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            risk_score REAL,   -- 0.0 to 1.0
            risk_label TEXT,   -- "low", "moderate", "high"
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            recommendation_text TEXT,
            explanation_text TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


# -----------------------------
# Insert functions
# -----------------------------
def add_patient(name, age, sex):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patients (name, age, sex, created_at) VALUES (?, ?, ?, ?)",
        (name, age, sex, datetime.now().isoformat())
    )
    conn.commit()
    patient_id = cur.lastrowid
    conn.close()
    return patient_id


def add_clinical_record(patient_id, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal, timestamp=None):
    conn = get_connection()
    cur = conn.cursor()
    ts = timestamp or datetime.now().isoformat()
    cur.execute("""
        INSERT INTO clinical_records
        (patient_id, timestamp, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, ts, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal))
    conn.commit()
    conn.close()


def add_lifestyle_record(patient_id, steps, sleep_hours, resting_hr, diet_quality, timestamp=None):
    conn = get_connection()
    cur = conn.cursor()
    ts = timestamp or datetime.now().isoformat()
    cur.execute("""
        INSERT INTO lifestyle_records (patient_id, timestamp, steps, sleep_hours, resting_hr, diet_quality)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_id, ts, steps, sleep_hours, resting_hr, diet_quality))
    conn.commit()
    conn.close()


def add_prediction(patient_id, risk_score, risk_label, timestamp=None):
    conn = get_connection()
    cur = conn.cursor()
    ts = timestamp or datetime.now().isoformat()
    cur.execute("""
        INSERT INTO predictions (patient_id, timestamp, risk_score, risk_label)
        VALUES (?, ?, ?, ?)
    """, (patient_id, ts, risk_score, risk_label))
    conn.commit()
    conn.close()


def add_recommendation(patient_id, recommendation_text, explanation_text, timestamp=None):
    conn = get_connection()
    cur = conn.cursor()
    ts = timestamp or datetime.now().isoformat()
    cur.execute("""
        INSERT INTO recommendations (patient_id, timestamp, recommendation_text, explanation_text)
        VALUES (?, ?, ?, ?)
    """, (patient_id, ts, recommendation_text, explanation_text))
    conn.commit()
    conn.close()


# -----------------------------
# Retrieval: the actual "digital twin" view
# -----------------------------
def get_twin_state(patient_id):
    """
    Returns the CURRENT state of a patient's digital twin:
    latest clinical record, latest lifestyle record, latest prediction,
    latest recommendation, plus full history of each for trend analysis.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    patient = dict(cur.fetchone())

    cur.execute("SELECT * FROM clinical_records WHERE patient_id = ? ORDER BY timestamp", (patient_id,))
    clinical_history = [dict(row) for row in cur.fetchall()]

    cur.execute("SELECT * FROM lifestyle_records WHERE patient_id = ? ORDER BY timestamp", (patient_id,))
    lifestyle_history = [dict(row) for row in cur.fetchall()]

    cur.execute("SELECT * FROM predictions WHERE patient_id = ? ORDER BY timestamp", (patient_id,))
    prediction_history = [dict(row) for row in cur.fetchall()]

    cur.execute("SELECT * FROM recommendations WHERE patient_id = ? ORDER BY timestamp", (patient_id,))
    recommendation_history = [dict(row) for row in cur.fetchall()]

    conn.close()

    twin = {
        "patient": patient,
        "latest_clinical": clinical_history[-1] if clinical_history else None,
        "latest_lifestyle": lifestyle_history[-1] if lifestyle_history else None,
        "latest_prediction": prediction_history[-1] if prediction_history else None,
        "latest_recommendation": recommendation_history[-1] if recommendation_history else None,
        "clinical_history": clinical_history,
        "lifestyle_history": lifestyle_history,
        "prediction_history": prediction_history,
        "recommendation_history": recommendation_history,
    }
    return twin


# -----------------------------
# Demo: create a patient and simulate 3 updates over time
# -----------------------------
if __name__ == "__main__":
    init_db()

    print("\n===== CREATING DEMO PATIENT =====")
    pid = add_patient(name="Ravi Kumar", age=54, sex=1)
    print(f"Created patient with ID: {pid}")

    # Simulate 3 checkups spaced a month apart, with health slowly changing
    base_time = datetime.now() - timedelta(days=60)

    print("\n===== SIMULATING 3 MONTHLY UPDATES =====")
    checkups = [
        {"cp": 2, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 1, "thalach": 150,
         "exang": 0, "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 2, "steps": 3000,
         "sleep_hours": 5.5, "resting_hr": 82, "diet_quality": "poor"},
        {"cp": 2, "trestbps": 135, "chol": 245, "fbs": 0, "restecg": 1, "thalach": 155,
         "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2, "steps": 4500,
         "sleep_hours": 6.2, "resting_hr": 78, "diet_quality": "average"},
        {"cp": 1, "trestbps": 128, "chol": 220, "fbs": 0, "restecg": 0, "thalach": 162,
         "exang": 0, "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 2, "steps": 6800,
         "sleep_hours": 7.0, "resting_hr": 72, "diet_quality": "good"},
    ]

    for i, c in enumerate(checkups):
        ts = (base_time + timedelta(days=30 * i)).isoformat()
        add_clinical_record(pid, c["cp"], c["trestbps"], c["chol"], c["fbs"], c["restecg"],
                              c["thalach"], c["exang"], c["oldpeak"], c["slope"], c["ca"], c["thal"], timestamp=ts)
        add_lifestyle_record(pid, c["steps"], c["sleep_hours"], c["resting_hr"], c["diet_quality"], timestamp=ts)
        print(f"  Month {i+1}: chol={c['chol']}, resting_hr={c['resting_hr']}, diet={c['diet_quality']}")

    print("\n===== CURRENT TWIN STATE =====")
    twin = get_twin_state(pid)
    print(json.dumps(twin, indent=2, default=str))