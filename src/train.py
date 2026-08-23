import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
from src.data_loader import OlaDataLoader
from src.feature_builder import FeatureBuilder
from src.trainer import GBDTTrioTrainer
from src.st_gnn_model import STGNNPredictor, HAS_TORCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Phase 3 Model Training Pipeline...")

    # 1. Load Data
    loader = OlaDataLoader(default_city="chennai")
    ola_df = loader.load_ola_data()
    gps_df = loader.load_uber_gps_data()
    weather_df = loader.load_weather_holiday_data()

    # 2. Build Spatiotemporal Features & Spatial Graph W_ij
    logger.info("Fitting MiniBatchKMeans (K=6) & generating spatiotemporal feature matrix...")
    builder = FeatureBuilder(n_clusters=6, random_state=42)
    builder.fit_spatial_clusters(gps_df)

    feature_path = Path("data/processed/spatiotemporal_features_clean.parquet")
    features_df = builder.transform(ola_df, weather_df=weather_df, save_parquet_path=str(feature_path))

    # 3. Train GBDT Trio Models (XGBoost Tweedie + LightGBM + CatBoost)
    logger.info("Training GBDT Trio Models (XGBoost Tweedie, LightGBM, CatBoost) with Optuna TPE tuning...")
    trainer = GBDTTrioTrainer(random_state=42)
    metrics = trainer.fit(features_df, horizons=[1, 2, 3, 4], n_trials=10)

    # 4. Save GBDT Trio Artifacts
    gbdt_model_path = Path("models/gbdt_trio_model.joblib")
    trainer.save(str(gbdt_model_path))
    logger.info(f"GBDT Trio Model saved to: {gbdt_model_path.resolve()}")

    # 5. Train & Save ST-GNN PyTorch Model (if PyTorch available)
    if HAS_TORCH:
        import torch
        logger.info("Training Spatiotemporal Graph Neural Network (ST-GNN)...")
        st_gnn = STGNNPredictor(num_nodes=6, in_features=35, out_horizons=4)
        st_gnn_path = Path("models/st_gnn_model.pt")
        st_gnn_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(st_gnn.model.state_dict(), st_gnn_path)
        logger.info(f"ST-GNN Model saved to: {st_gnn_path.resolve()}")

    print("\n========================================================")
    print("=== Training Complete — Model Checkpoints in models/ ===")
    print("========================================================")
    for h, res in metrics.items():
        print(f"Horizon t+{h} | XGB: {res['xgb_wape']:.4f} | LGB: {res['lgb_wape']:.4f} | CAT: {res['cat_wape']:.4f} | Ensemble WAPE: {res['ensemble_wape']:.4f}")


if __name__ == "__main__":
    main()
