

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
    
    # do the same feature engineering
    data["rainfall_elevation"] = data["Rainfall"] / (data["Elevation"] + 1)
    data["terrain_risk"] = data["Rainfall"] / (data["Elevation"] - data["Elevation"].min() + 10)
    data["rain_slope"] = data["Rainfall"] * data["Slope"]


    for col in ["rainfall_elevation", "terrain_risk", "rain_slope"]:
        min_val = data[col].min()
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        data[col] = np.log1p(data[col] + shift)
        
    feature_order = [
        "Latitude",
        "Longitude",
        "duration",
        "Rainfall",
        "Elevation",
        "Slope",
        "rainfall_elevation",
        "terrain_risk",
        "rain_slope"
    ]


    data = data[feature_order]
    
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]


    return prediction, probability


