import sys
import unittest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.data_loader import OlaDataLoader
from src.feature_builder import FeatureBuilder
from src.trainer import GBDTTrioTrainer, compute_wape
from src.evaluate import ModelEvaluator


class TestTrainerAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.loader = OlaDataLoader(data_dir=self.tmpdir.name)
        self.ola_df = self.loader.load_ola_data(city="chennai", force_synthetic=True)
        self.gps_df = self.loader.load_uber_gps_data(city="chennai", force_synthetic=True)
        self.weather_df = self.loader.load_weather_holiday_data(force_synthetic=True)

        self.builder = FeatureBuilder(n_clusters=4, random_state=42)
        self.builder.fit_spatial_clusters(self.gps_df)
        self.features_df = self.builder.transform(self.ola_df, weather_df=self.weather_df, drop_na=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_compute_wape(self):
        y_true = np.array([10.0, 20.0, 30.0, 0.0])
        y_pred = np.array([12.0, 18.0, 30.0, 0.0])
        wape = compute_wape(y_true, y_pred)
        # sum abs diff = 2 + 2 + 0 + 0 = 4; sum true = 60; wape = 4 / 60 = 0.0667
        self.assertAlmostEqual(wape, 4.0 / 60.0, places=4)

    def test_fit_and_predict_gbdt_trio(self):
        trainer = GBDTTrioTrainer(random_state=42)
        metrics = trainer.fit(self.features_df, horizons=[1, 2], val_ratio=0.2, n_trials=2)

        self.assertIn(1, metrics)
        self.assertIn(2, metrics)
        self.assertIn("ensemble_wape", metrics[1])
        self.assertLessEqual(metrics[1]["ensemble_wape"], 1.0)

        # Test predict horizon output shape
        feature_cols = trainer.feature_names
        X_sample = self.features_df[feature_cols].head(10)
        preds = trainer.predict_horizon(X_sample, horizon=1)
        self.assertEqual(len(preds), 10)
        self.assertTrue((preds >= 0.0).all())

    def test_model_save_and_load(self):
        trainer = GBDTTrioTrainer(random_state=42)
        trainer.fit(self.features_df, horizons=[1], val_ratio=0.2, n_trials=2)

        save_path = Path(self.tmpdir.name) / "test_model.joblib"
        trainer.save(str(save_path))
        self.assertTrue(save_path.exists())

        loaded_trainer = GBDTTrioTrainer.load(str(save_path))
        self.assertEqual(loaded_trainer.feature_names, trainer.feature_names)
        self.assertIn(1, loaded_trainer.models["xgboost"])

    def test_evaluator_report(self):
        trainer = GBDTTrioTrainer(random_state=42)
        trainer.fit(self.features_df, horizons=[1], val_ratio=0.2, n_trials=2)

        model_path = Path(self.tmpdir.name) / "eval_model.joblib"
        trainer.save(str(model_path))

        evaluator = ModelEvaluator(str(model_path))
        report = evaluator.generate_evaluation_report(self.features_df, horizons=[1])

        self.assertIsInstance(report, pd.DataFrame)
        self.assertEqual(len(report), 1)
        self.assertIn("wape", report.columns)
        self.assertIn("zero_count_residual", report.columns)


if __name__ == "__main__":
    unittest.main()
