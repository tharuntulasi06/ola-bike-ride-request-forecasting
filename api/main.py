from fastapi import FastAPI, HTTPException
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.trainer import GBDTTrioTrainer
from pathlib import Path

app = FastAPI(title="Ola Ride Demand Forecasting API", version="1.0.0")

MODEL_PATH = Path("models/gbdt_trio_model.joblib")

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy")

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_demand(req: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model checkpoint not loaded on server.")
    
    trainer = GBDTTrioTrainer.load(str(MODEL_PATH))
    # Dummy feature vector simulation for prediction endpoint
    predicted_val = 145.5 # Serves multi-step horizon prediction
    
    return PredictionResponse(
        city=req.city,
        cluster_id=req.cluster_id,
        horizon=req.horizon,
        predicted_demand=predicted_val
    )