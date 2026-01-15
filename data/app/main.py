from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from app.schemas import PlayerInput,PredictionResponse
from model.train_and_save_model import model

app = FastAPI(title="Football Market Value Predictor")

@app.get("/")
def health_check():
    return {"status": "API is running"}
@app.post("/predict", response_model=PredictionResponse)
def predict_value_tier(player: PlayerInput):
    data = pd.DataFrame([player.model_dump()])

    data = data.replace([np.inf,-np.inf], np.nan)
    data[["offensive_efficiency","progressive_actions"]] = (data[["offensive_efficiency","progressive_actions"]].fillna(0))

    prediction = model.predict(data)[0]

    return {"predicted_value_tier" : prediction}
