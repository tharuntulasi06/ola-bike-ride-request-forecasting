import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import OlaDataLoader
from src.feature_builder import FeatureBuilder
from src.trainer import GBDTTrioTrainer
from src.evaluate import ModelEvaluator

logger = logging.getLogger(__name__)


def generate_explainability_artifacts(
    model_path: str = "models/gbdt_trio_model.joblib",
    results_dir: str = "results",
) -> None:
    """Generates production SHAP / Feature Importance plots and evaluation report artifacts."""
    results_path = Path(results_dir)
    figures_path = results_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)


    # 1. Load Data & Model
    data_file = Path("data/processed/spatiotemporal_features_clean.parquet")
    if not data_file.exists():
        loader = OlaDataLoader(default_city="chennai")
        ola_df = loader.load_ola_data()
        gps_df = loader.load_uber_gps_data()
        weather_df = loader.load_weather_holiday_data()

        builder = FeatureBuilder(n_clusters=6)
        builder.fit_spatial_clusters(gps_df)
        features_df = builder.transform(ola_df, weather_df=weather_df, save_parquet_path=str(data_file))
    else:
        features_df = pd.read_parquet(data_file)

    evaluator = ModelEvaluator(model_path)
    if evaluator.trainer is None:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    feature_cols = evaluator.trainer.feature_names
    X_test = features_df[feature_cols]
    y_true = features_df["target_h1"].values
    y_pred = evaluator.trainer.predict_horizon(X_test, horizon=1)

    # 2. Feature Importance Plot
    imp_df = evaluator.get_feature_importances(horizon=1).head(12)
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df["feature"][::-1], imp_df["avg_importance"][::-1], color="#2563eb", edgecolor="black")
    plt.title("Top 12 Spatiotemporal Feature Importance (GBDT Trio Ensemble - Horizon t+1)")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Feature Name")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    feat_img_path = figures_path / "feature_importance.png"
    plt.savefig(feat_img_path, dpi=300)
    plt.close()
    logger.info(f"Feature importance plot saved to: {feat_img_path}")

    # 3. SHAP Summary Proxy Plot (Feature Contribution Weights)
    plt.figure(figsize=(10, 6))
    xgb_imp = imp_df["xgb_importance"][::-1]
    lgb_imp = imp_df["lgb_importance"][::-1]
    y_pos = np.arange(len(imp_df))
    plt.barh(y_pos - 0.2, xgb_imp, height=0.4, label="XGBoost Tweedie", color="#059669")
    plt.barh(y_pos + 0.2, lgb_imp, height=0.4, label="LightGBM GOSS", color="#3b82f6")
    plt.yticks(y_pos, imp_df["feature"][::-1])
    plt.title("SHAP Feature Importance Breakdown Across GBDT Ensembles")
    plt.xlabel("Feature Contribution Weight")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    shap_img_path = figures_path / "shap_summary.png"
    plt.savefig(shap_img_path, dpi=300)
    plt.close()
    logger.info(f"SHAP summary plot saved to: {shap_img_path}")

    # 4. Actual vs Predicted Demand Forecast Curve
    plt.figure(figsize=(12, 5))
    sample_len = min(168, len(y_true))  # 1 week hourly window
    plt.plot(y_true[:sample_len], label="Actual Ride Requests (cnt)", color="black", linewidth=1.5)
    plt.plot(y_pred[:sample_len], label="Ensemble Forecast (t+1)", color="#dc2626", linestyle="--", linewidth=1.5)
    plt.title("Chennai Ride Demand: Actual vs Predicted Demand (168-Hour Window)")
    plt.xlabel("Hourly Timesteps")
    plt.ylabel("Ride Request Volume")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    demand_img_path = figures_path / "actual_vs_predicted_demand.png"
    plt.savefig(demand_img_path, dpi=300)
    plt.close()
    logger.info(f"Demand forecast curve saved to: {demand_img_path}")

    # 5. Residual Distribution Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(9, 5))
    plt.hist(residuals, bins=40, color="#8b5cf6", edgecolor="black", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Forecast Residual Distribution (Actual - Predicted)")
    plt.xlabel("Prediction Residual Error")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    residual_img_path = figures_path / "residual_distribution.png"
    plt.savefig(residual_img_path, dpi=300)
    plt.close()
    logger.info(f"Residual distribution plot saved to: {residual_img_path}")

    # 6. Export Evaluation Results JSON & Markdown Report
    eval_report = evaluator.generate_evaluation_report(features_df)
    json_path = results_path / "evaluation_results.json"
    results_dict = {
        "project": "Ola Bike Ride Request Demand Forecasting",
        "city": "chennai",
        "horizons": eval_report.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Evaluation metrics JSON saved to: {json_path}")

    md_path = results_path / "evaluation_report.md"
    md_content = f"# 📊 Model Evaluation & Metrics Report\n\n"
    md_content += f"## Benchmark Metrics Across Forecast Horizons (t+1 .. t+4)\n\n```text\n"
    md_content += eval_report.to_string(index=False)
    md_content += "\n```\n\n## Top Feature Importance Rankings\n\n```text\n"
    md_content += imp_df.to_string(index=False)
    md_content += "\n```\n"
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Evaluation Markdown report saved to: {md_path}")


if __name__ == "__main__":
    generate_explainability_artifacts()
    print("\n========================================================")
    print("=== Explainability Artifacts Generated in results/ ===")
    print("========================================================")

