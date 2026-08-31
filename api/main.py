import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ClusterListResponse,
    ClusterInfo,
    HealthResponse,
)
from src.trainer import GBDTTrioTrainer
from src.data_loader import OlaDataLoader
from src.feature_builder import FeatureBuilder


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ola Ride Demand Forecasting API Microservice",
    description="Production REST microservice for spatiotemporal multi-step ride demand predictions.",
    version="1.0.0",
)

MODEL_PATH = Path("models/gbdt_trio_model.joblib")
DATA_PATH = Path("data/processed/spatiotemporal_features_clean.parquet")

CHENNAI_LANDMARKS = {
    0: ("chennai_central", 13.0827, 80.2707),
    1: ("t_nagar", 13.0418, 80.2341),
    2: ("omr_it_corridor", 12.9645, 80.2443),
    3: ("velachery", 12.9750, 80.2207),
    4: ("guindy_kathipara", 13.0067, 80.2020),
    5: ("cmbt_anna_nagar", 13.0850, 80.2101),
}

# Global in-memory cache for fast real-time inference
trainer_cache: Dict[str, Any] = {"trainer": None, "features_df": None}


def load_model_and_features():
    if not DATA_PATH.exists():
        logger.info("Features matrix missing. Building synthetic feature matrix for API server...")
        loader = OlaDataLoader(default_city="chennai")
        ola_df = loader.load_ola_data(force_synthetic=True)
        gps_df = loader.load_uber_gps_data(force_synthetic=True)
        weather_df = loader.load_weather_holiday_data(force_synthetic=True)
        builder = FeatureBuilder(n_clusters=6, random_state=42)
        builder.fit_spatial_clusters(gps_df)
        builder.transform(ola_df, weather_df=weather_df, save_parquet_path=str(DATA_PATH))

    if not MODEL_PATH.exists():
        logger.info("Model checkpoint missing. Training GBDT Trio model for API server...")
        features_df = pd.read_parquet(DATA_PATH)
        trainer = GBDTTrioTrainer(random_state=42)
        trainer.fit(features_df, horizons=[1, 2, 3, 4], n_trials=2)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(MODEL_PATH))

    if trainer_cache["trainer"] is None and MODEL_PATH.exists():
        trainer_cache["trainer"] = GBDTTrioTrainer.load(str(MODEL_PATH))

    if trainer_cache["features_df"] is None and DATA_PATH.exists():
        trainer_cache["features_df"] = pd.read_parquet(DATA_PATH)



@app.on_event("startup")
def startup_event():
    load_model_and_features()


@app.get("/health", response_model=HealthResponse)
def health_check():
    load_model_and_features()
    is_loaded = trainer_cache["trainer"] is not None
    return HealthResponse(status="healthy", model_loaded=is_loaded)


@app.get("/api/v1/clusters", response_model=ClusterListResponse)
def get_chennai_clusters():
    clusters = [
        ClusterInfo(
            cluster_id=cid,
            landmark_name=info[0],
            latitude=info[1],
            longitude=info[2],
        )
        for cid, info in CHENNAI_LANDMARKS.items()
    ]
    return ClusterListResponse(city="chennai", clusters=clusters)


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_demand(req: PredictionRequest):
    load_model_and_features()
    trainer: GBDTTrioTrainer = trainer_cache["trainer"]
    features_df: pd.DataFrame = trainer_cache["features_df"]

    if trainer is None:
        raise HTTPException(
            status_code=503, detail="Model checkpoint not loaded. Train model via 'python src/train.py' first."
        )

    if req.cluster_id not in CHENNAI_LANDMARKS:
        raise HTTPException(status_code=400, detail=f"Invalid cluster_id {req.cluster_id}. Must be between 0 and 5.")

    landmark_name = CHENNAI_LANDMARKS[req.cluster_id][0]
    feature_cols = trainer.feature_names

    # Real inference execution on spatiotemporal features
    if features_df is not None and not features_df.empty:
        cluster_rows = features_df[features_df["cluster_id"] == req.cluster_id]
        if not cluster_rows.empty:
            X_sample = cluster_rows[feature_cols].tail(1)
        else:
            X_sample = features_df[feature_cols].tail(1)
    else:
        raise HTTPException(status_code=503, detail="Feature matrix dataset not found.")

    # Execute real model inference prediction
    real_prediction = float(trainer.predict_horizon(X_sample, horizon=req.horizon)[0])
    real_prediction = max(0.0, round(real_prediction, 2))

    return PredictionResponse(
        city=req.city,
        cluster_id=req.cluster_id,
        landmark_name=landmark_name,
        horizon=req.horizon,
        predicted_demand=real_prediction,
    )


from src.db import DatabaseManager, HAS_DUCKDB

db_manager = DatabaseManager()

@app.get("/api/v1/analytics/db-summary")
def get_db_spatial_summary():
    """Returns spatial zone aggregate statistics queried directly via DuckDB SQL over Parquet."""
    if not HAS_DUCKDB:
        raise HTTPException(status_code=503, detail="DuckDB engine not available.")
    
    df_summary = db_manager.get_spatial_zone_aggregates()
    return {"status": "success", "engine": "DuckDB SQL", "zones": df_summary.to_dict(orient="records")}


@app.get("/api/v1/analytics/metrics")
def get_evaluation_metrics():
    json_path = Path("results/evaluation_results.json")
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation results report not found.")


    with open(json_path, "r") as f:
        metrics_data = json.load(f)

    return metrics_data



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)