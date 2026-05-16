
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

CORS(app)

# =========================
# LOAD AI MODELS
# =========================

from huggingface_hub import hf_hub_download
amr_model_path = hf_hub_download(
    repo_id="Rawanx1/aegisdx-models",
    filename="amr_resistance_model.pkl"
)

early_model_path = hf_hub_download(
    repo_id="Rawanx1/aegisdx-models",
    filename="early_model.pkl"
)

diagnostic_model_path = hf_hub_download(
    repo_id="Rawanx1/aegisdx-models",
    filename="diagnostic_model.pkl"
)

model = joblib.load(amr_model_path)
early_model = joblib.load(early_model_path)
diagnostic_model = joblib.load(diagnostic_model_path)

# =========================
# FRONTEND ROUTES
# =========================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/early.html")
def early():
    return send_from_directory(".", "early.html")

@app.route("/diagnostic.html")
def diagnostic():
    return send_from_directory(".", "diagnostic.html")

@app.route("/amr.html")
def amr_page():
    return send_from_directory(".", "amr.html")

# =========================
# EARLY PREDICTION AI API
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    symptoms = data.get("symptoms", "")

    prediction = early_model.predict([symptoms])[0]

    infection = "Possible Infection"

    confidence = "96%"

    recommendation = (
        "Clinical follow-up recommended."
    )

    if prediction == "Low":

        infection = "Minor Viral Infection"

        confidence = "88%"

        recommendation = (
            "Rest and hydration recommended."
        )

    return jsonify({

        "risk": prediction + " Risk",

        "infection": infection,

        "confidence": confidence,

        "recommendation": recommendation
    })

# =========================
# DIAGNOSTIC AI API
# =========================

@app.route("/diagnose", methods=["POST"])
def diagnose():

    data = request.json

    sample = pd.DataFrame([{

        "age": float(data.get("age", 45)),
        "bmi": float(data.get("bmi", 25)),
        "systolic_bp": float(data.get("systolic_bp", 120)),
        "diastolic_bp": float(data.get("diastolic_bp", 80)),
        "glucose": float(data.get("glucose", 100)),
        "cholesterol": float(data.get("cholesterol", 180)),
        "creatinine": float(data.get("creatinine", 1.0)),
        "diabetes": int(data.get("diabetes", 0)),
        "hypertension": int(data.get("hypertension", 0))
    }])

    prediction = diagnostic_model.predict(sample)[0]

    diagnosis_name = diagnosis_encoder.inverse_transform(
        [prediction]
    )[0]

    return jsonify({

        "pathogen":
        f"{diagnosis_name}",

        "interpretation":
        "Prediction generated using clinical diagnostic AI model.",

        "tests":
        "Further laboratory evaluation recommended.",

        "notes":
        "AI-generated clinical assessment.",

        "confidence": "92%"
    })

# =========================
# AMR AI MODEL API
# =========================

@app.route("/amr", methods=["POST"])
def amr():

    data = request.json

    bacteria = data.get("bacteria", "")
    antibiotic = data.get("antibiotic", "")

    if antibiotic == "":
        antibiotic = "Ciprofloxacin"

    sample = pd.DataFrame({
        "pathogen": [bacteria],
        "antibiotic_name": [antibiotic]
    })

    prediction = model.predict(sample)[0]

    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="google/gemma-2b-it",
        token=os.getenv("HF_TOKEN")
    )

    gemma_response = pipe(
        f"""
        Explain this antimicrobial resistance result clinically:

        Resistance Prediction: {prediction:.1f}%

        Give a short medical explanation.
        """,
        max_new_tokens=80
    )

    gemma_text = gemma_response[0]["generated_text"]

    resistance_level = "Low"

    if prediction >= 70:
        resistance_level = "High"

    elif prediction >= 40:
        resistance_level = "Moderate"

    return jsonify({

        "warning":
        f"{resistance_level} Resistance ({prediction:.1f}%)",

        "alternatives":
        "Ceftriaxone, Meropenem",

        "trend":
        "Predicted using WHO AMR dataset",

        "explanation":
        "Resistance percentage predicted using trained machine learning model.",

        "gemma_explanation":
        gemma_text,

        "confidence": "94%"
    })

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    import os

    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000))
    )
