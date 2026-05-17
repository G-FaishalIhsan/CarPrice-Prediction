import json
from pathlib import Path

import os
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent

MODEL_FILE = BASE_DIR / "car_price_model.joblib"
CONFIG_FILE = BASE_DIR / "feature_config.json"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_FILE}\n"
        "Jalankan train_model.py atau download model dari Google Colab terlebih dahulu."
    )

model = joblib.load(MODEL_FILE)

with open(CONFIG_FILE, encoding="utf-8") as f:
    config = json.load(f)

RMSE = config["rmse"]
R2 = config["r2_score"]
VEHICLE_TYPES = config.get("vehicle_types", ["Car", "Passenger"])

app = FastAPI(title="CarPrice Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://carprice-prediction-ds.netlify.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

def quality_label(r2: float) -> str:
    if r2 >= 0.80: return "Baik"
    elif r2 >= 0.60: return "Cukup Baik / Moderat"
    return "Perlu Ditingkatkan"

class CarSpec(BaseModel):
    Engine_size: float
    Horsepower: float
    Wheelbase: float
    Width: float
    Length: float
    Curb_weight: float
    Fuel_capacity: float
    Fuel_efficiency: float
    Vehicle_type: str

    class Config:
        json_schema_extra = {
            "example": {
                "Engine_size": 2.3,
                "Horsepower": 150,
                "Wheelbase": 107,
                "Width": 70,
                "Length": 185,
                "Curb_weight": 3.1,
                "Fuel_capacity": 16,
                "Fuel_efficiency": 27,
                "Vehicle_type": "Passenger"
            }
        }

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "CarPrice Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Linear Regression (sklearn Pipeline)",
        "model_file": MODEL_FILE.name,
        "rmse": round(RMSE, 4),
        "mae": config.get("mae", None),
        "r2_score": round(R2, 4),
        "model_quality": quality_label(R2),
        "vehicle_types": VEHICLE_TYPES,
        "numeric_features": config.get("numeric_features", []),
        "categorical_features": config.get("categorical_features", []),
    }

@app.post("/predict")
def predict(spec: CarSpec):
    if spec.Vehicle_type not in VEHICLE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Vehicle_type harus salah satu dari: {VEHICLE_TYPES}"
        )
    input_df = pd.DataFrame([{
        "Engine_size":    spec.Engine_size,
        "Horsepower":     spec.Horsepower,
        "Wheelbase":      spec.Wheelbase,
        "Width":          spec.Width,
        "Length":         spec.Length,
        "Curb_weight":    spec.Curb_weight,
        "Fuel_capacity":  spec.Fuel_capacity,
        "Fuel_efficiency":spec.Fuel_efficiency,
        "Vehicle_type":   spec.Vehicle_type,
    }])

    pred = float(model.predict(input_df)[0])
    pred = max(0.0, pred) 

    return {
        "predicted_price_thousand_usd": round(pred, 3),
        "predicted_price_usd": round(pred * 1000, 2),
        "estimated_lower_usd": round(max(0, pred - RMSE) * 1000, 2),
        "estimated_upper_usd": round((pred + RMSE) * 1000, 2),
        "r2_score": round(R2, 4),
        "rmse_thousand_usd": round(RMSE, 4),
        "model_quality": quality_label(R2),
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)