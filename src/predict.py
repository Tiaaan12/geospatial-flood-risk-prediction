

from pathlib import Path
import pandas as pd
import numpy as np
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_single(latitude, longitude, duration, rainfall, elevation, slope):


    # create the dataframe with base features
    data = pd.DataFrame([{
        "Latitude": latitude,
        "Longitude": longitude,
        "duration": duration,
        "Rainfall": rainfall,
        "Elevation": elevation,
        "Slope": slope
    }])
