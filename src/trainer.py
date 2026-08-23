import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Weighted Absolute Percentage Error (WAPE). Zero-safe count evaluation metric."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum(np.abs(y_true))
    if denom == 0.0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


class GBDTTrioTrainer:
    """Production Multi-Horizon Gradient Boosting Trio Trainer (XGBoost, LightGBM, CatBoost)."""

    FEATURE_EXCLUSIONS = [
        "datetime",
        "city",
        "cnt",
        "casual",
        "registered",
        "target_h1",
        "target_h2",
        "target_h3",
        "target_h4",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Dict[int, Any]] = {
            "xgboost": {},
            "lightgbm": {},
            "catboost": {},
        }
        self.feature_names: List[str] = []
        self.ensemble_weights: Dict[int, Dict[str, float]] = {}

    def extract_feature_columns(self, df: pd.DataFrame) -> List[str]:
        num_cols = list(df.select_dtypes(include=[np.number, "category"]).columns)
        return [c for c in num_cols if c not in self.FEATURE_EXCLUSIONS and not c.startswith("target_")]


    def train_single_horizon(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        horizon: int = 1,
        n_trials: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.feature_names = list(X_train.columns)

        def objective(trial: optuna.Trial) -> float:
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 150),
                "learning_rate": trial.suggest_float("xgb_lr", 0.02, 0.15, log=True),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
                "objective": "reg:tweedie",
                "tweedie_variance_power": trial.suggest_float("xgb_tweedie_power", 1.1, 1.9),
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            xgb = XGBRegressor(**xgb_params)
            xgb.fit(X_train, y_train)
            pred_xgb = np.clip(xgb.predict(X_val), 0.0, None)
            return compute_wape(y_val.values, pred_xgb)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_xgb_params = {
            "n_estimators": study.best_params["xgb_n_estimators"],
            "learning_rate": study.best_params["xgb_lr"],
            "max_depth": study.best_params["xgb_max_depth"],
            "subsample": study.best_params["xgb_subsample"],
            "colsample_bytree": study.best_params["xgb_colsample"],
            "objective": "reg:tweedie",
            "tweedie_variance_power": study.best_params["xgb_tweedie_power"],
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        xgb_model = XGBRegressor(**best_xgb_params)
        xgb_model.fit(X_train, y_train)

        lgb_model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)

        cat_model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.08,
            depth=6,
            random_seed=self.random_state,
            verbose=0,
        )
        cat_model.fit(X_train, y_train)

        p_xgb = np.clip(xgb_model.predict(X_val), 0.0, None)
        p_lgb = np.clip(lgb_model.predict(X_val), 0.0, None)
        p_cat = np.clip(cat_model.predict(X_val), 0.0, None)

        w_xgb = compute_wape(y_val.values, p_xgb)
        w_lgb = compute_wape(y_val.values, p_lgb)
        w_cat = compute_wape(y_val.values, p_cat)

        inv_sum = (1.0 / (w_xgb + 1e-6)) + (1.0 / (w_lgb + 1e-6)) + (1.0 / (w_cat + 1e-6))
        weights = {
            "xgboost": (1.0 / (w_xgb + 1e-6)) / inv_sum,
            "lightgbm": (1.0 / (w_lgb + 1e-6)) / inv_sum,
            "catboost": (1.0 / (w_cat + 1e-6)) / inv_sum,
        }

        self.models["xgboost"][horizon] = xgb_model
        self.models["lightgbm"][horizon] = lgb_model
        self.models["catboost"][horizon] = cat_model
        self.ensemble_weights[horizon] = weights

        horizon_models = {
            "xgboost": xgb_model,
            "lightgbm": lgb_model,
            "catboost": cat_model,
        }

        return horizon_models, weights

    def fit(
        self,
        features_df: pd.DataFrame,
        horizons: List[int] = [1, 2, 3, 4],
        val_ratio: float = 0.2,
        n_trials: int = 10,
    ) -> Dict[int, Dict[str, float]]:
        features_df = features_df.sort_values("datetime").reset_index(drop=True)
        feature_cols = self.extract_feature_columns(features_df)

        n_samples = len(features_df)
        split_idx = int(n_samples * (1.0 - val_ratio))

        train_df = features_df.iloc[:split_idx]
        val_df = features_df.iloc[split_idx:]

        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]

        results = {}
        for h in horizons:
            target_col = f"target_h{h}"
            if target_col not in features_df.columns:
                continue

            y_train = train_df[target_col]
            y_val = val_df[target_col]

            _, weights = self.train_single_horizon(
                X_train, y_train, X_val, y_val, horizon=h, n_trials=n_trials
            )

            p_xgb = np.clip(self.models["xgboost"][h].predict(X_val), 0.0, None)
            p_lgb = np.clip(self.models["lightgbm"][h].predict(X_val), 0.0, None)
            p_cat = np.clip(self.models["catboost"][h].predict(X_val), 0.0, None)

            p_ens = weights["xgboost"] * p_xgb + weights["lightgbm"] * p_lgb + weights["catboost"] * p_cat
            ens_wape = compute_wape(y_val.values, p_ens)

            results[h] = {
                "xgb_wape": compute_wape(y_val.values, p_xgb),
                "lgb_wape": compute_wape(y_val.values, p_lgb),
                "cat_wape": compute_wape(y_val.values, p_cat),
                "ensemble_wape": ens_wape,
            }

        return results

    def predict_horizon(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        p_xgb = np.clip(self.models["xgboost"][horizon].predict(X), 0.0, None)
        p_lgb = np.clip(self.models["lightgbm"][horizon].predict(X), 0.0, None)
        p_cat = np.clip(self.models["catboost"][horizon].predict(X), 0.0, None)

        w = self.ensemble_weights[horizon]
        return w["xgboost"] * p_xgb + w["lightgbm"] * p_lgb + w["catboost"] * p_cat

    def save(self, save_path: str = "models/gbdt_trio_model.joblib") -> None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "models": self.models,
            "feature_names": self.feature_names,
            "ensemble_weights": self.ensemble_weights,
            "random_state": self.random_state,
        }
        joblib.dump(artifact, out_file)
        logger.info(f"Model artifacts saved successfully to {out_file}")

    @classmethod
    def load(cls, load_path: str = "models/gbdt_trio_model.joblib") -> "GBDTTrioTrainer":
        artifact = joblib.load(load_path)
        trainer = cls(random_state=artifact["random_state"])
        trainer.models = artifact["models"]
        trainer.feature_names = artifact["feature_names"]
        trainer.ensemble_weights = artifact["ensemble_weights"]
        return trainer


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data_loader import OlaDataLoader
    from src.feature_builder import FeatureBuilder

    loader = OlaDataLoader(default_city="chennai")
    ola_df = loader.load_ola_data()
    gps_df = loader.load_uber_gps_data()
    weather_df = loader.load_weather_holiday_data()

    builder = FeatureBuilder(n_clusters=6)
    builder.fit_spatial_clusters(gps_df)
    features_df = builder.transform(ola_df, weather_df=weather_df, save_parquet_path="data/processed/spatiotemporal_features_clean.parquet")

    trainer = GBDTTrioTrainer()
    metrics = trainer.fit(features_df, horizons=[1, 2, 3, 4], n_trials=5)

    print("\n=== Phase 3 Model Training Metrics (WAPE) ===")
    for h, res in metrics.items():
        print(f"Horizon t+{h} | XGB: {res['xgb_wape']:.4f} | LGB: {res['lgb_wape']:.4f} | CAT: {res['cat_wape']:.4f} | Ensemble: {res['ensemble_wape']:.4f}")

    trainer.save("models/gbdt_trio_model.joblib")
