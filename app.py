from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('diabetes_rf_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')

# Define columns as in training
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

app = FastAPI()

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: str
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(data: DiabetesInput):
    input_dict = data.dict()

    # Clean string inputs like "not mentioned"
    for k, v in input_dict.items():
        if isinstance(v, str) and v.strip().lower() in ['no', 'not mentioned', 'n/a', 'unknown', '', 'none']:
            input_dict[k] = np.nan

    df = pd.DataFrame([input_dict])
    df = df[feature_columns]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Scale
    scaled = scaler.transform(df)

    # Predict
    probs = model.predict_proba(scaled)[0]
    pred_class = int(probs[1] >= 0.6)
    confidence = probs[1] if pred_class == 1 else probs[0]

    return {
        "prediction": "Diabetic" if pred_class else "Not Diabetic",
        "confidence": round(confidence * 100, 2)
    }
