import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.data_loader import OlaDataLoader
from src.feature_builder import FeatureBuilder
from src.trainer import GBDTTrioTrainer, compute_wape


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Production Evaluation Suite for Spatiotemporal Multi-Step Demand Forecasting."""

    def __init__(self, model_path: str = "models/gbdt_trio_model.joblib"):
        self.model_path = Path(model_path)
        self.trainer: Optional[GBDTTrioTrainer] = None
        if self.model_path.exists():
            self.trainer = GBDTTrioTrainer.load(str(self.model_path))

    def evaluate_horizon_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)

        wape = compute_wape(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "wape": float(wape),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "zero_count_residual": float(np.mean(np.abs(y_pred[y_true == 0])) if (y_true == 0).any() else 0.0),
        }

    def generate_evaluation_report(
        self, features_df: pd.DataFrame, horizons: List[int] = [1, 2, 3, 4]
    ) -> pd.DataFrame:
        if self.trainer is None:
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        feature_cols = self.trainer.feature_names
        X_test = features_df[feature_cols]

        rows = []
        for h in horizons:
            target_col = f"target_h{h}"
            if target_col not in features_df.columns:
                continue

            y_true = features_df[target_col].values
            y_pred = self.trainer.predict_horizon(X_test, horizon=h)

            metrics = self.evaluate_horizon_metrics(y_true, y_pred)
            metrics["horizon"] = f"t+{h}"
            rows.append(metrics)

        report_df = pd.DataFrame(rows)
        cols = ["horizon", "wape", "mae", "rmse", "r2", "zero_count_residual"]
        return report_df[cols]

    def get_feature_importances(self, horizon: int = 1) -> pd.DataFrame:
        if self.trainer is None:
            raise FileNotFoundError("Model trainer not loaded.")

        feature_cols = self.trainer.feature_names
        xgb = self.trainer.models["xgboost"][horizon]
        lgb = self.trainer.models["lightgbm"][horizon]

        xgb_imp = xgb.feature_importances_
        lgb_imp = lgb.feature_importances_ / np.sum(lgb.feature_importances_)

        df_imp = pd.DataFrame(
            {
                "feature": feature_cols,
                "xgb_importance": xgb_imp,
                "lgb_importance": lgb_imp,
                "avg_importance": (xgb_imp + lgb_imp) / 2.0,
            }
        )

        return df_imp.sort_values("avg_importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data_loader import OlaDataLoader
    from src.feature_builder import FeatureBuilder

    data_path = Path("data/processed/spatiotemporal_features_clean.parquet")

    if not data_path.exists():
        loader = OlaDataLoader(default_city="chennai")
        ola_df = loader.load_ola_data()
        gps_df = loader.load_uber_gps_data()
        weather_df = loader.load_weather_holiday_data()

        builder = FeatureBuilder(n_clusters=6)
        builder.fit_spatial_clusters(gps_df)
        features_df = builder.transform(ola_df, weather_df=weather_df, save_parquet_path=str(data_path))
    else:
        features_df = pd.read_parquet(data_path)

    model_path = Path("models/gbdt_trio_model.joblib")
    if not model_path.exists():
        print("Training GBDT Trio models first...")
        trainer = GBDTTrioTrainer()
        trainer.fit(features_df, horizons=[1, 2, 3, 4], n_trials=5)
        trainer.save(str(model_path))

    evaluator = ModelEvaluator(str(model_path))
    report = evaluator.generate_evaluation_report(features_df)

    print("\n========================================================")
    print("=== Phase 3 Evaluation Benchmark Summary (WAPE/MAE/RMSE) ===")
    print("========================================================")
    print(report.to_string(index=False))

    print("\n--- Top 10 Feature Importances (Horizon t+1) ---")
    print(evaluator.get_feature_importances(horizon=1).head(10).to_string(index=False))
