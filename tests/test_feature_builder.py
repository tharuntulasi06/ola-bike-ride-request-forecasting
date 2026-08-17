import sys
import unittest
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.data_loader import OlaDataLoader
from src.feature_builder import FeatureBuilder, haversine_distance_km


class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.loader = OlaDataLoader(data_dir=self.tmpdir.name)
        self.ola_df = self.loader.load_ola_data(city="chennai", force_synthetic=True)
        self.gps_df = self.loader.load_uber_gps_data(city="chennai", force_synthetic=True)
        self.weather_df = self.loader.load_weather_holiday_data(force_synthetic=True)
        self.builder = FeatureBuilder(n_clusters=4, random_state=42)


    def tearDown(self):
        self.tmpdir.cleanup()

    def test_haversine_distance(self):
        # Distance between Chennai Central (13.0827, 80.2707) and Guindy (13.0067, 80.2020) ~ 11.2 km
        dist = haversine_distance_km(13.0827, 80.2707, 13.0067, 80.2020)
        self.assertGreater(dist, 9.0)
        self.assertLess(dist, 14.0)

    def test_fit_spatial_clusters_and_adjacency(self):
        self.builder.fit_spatial_clusters(self.gps_df)
        self.assertIsNotNone(self.builder.kmeans)
        self.assertEqual(len(self.builder.cluster_centroids), 4)
        
        W = self.builder.adjacency_matrix
        self.assertEqual(W.shape, (4, 4))
        # Diagonal should be 0 (no self-loops in adjacency)
        self.assertTrue((np.diag(W) == 0.0).all())

    def test_cyclical_time_features(self):
        df_time = self.builder.build_cyclical_time_features(self.ola_df.head(24))
        self.assertIn("hour_sin", df_time.columns)
        self.assertIn("hour_cos", df_time.columns)
        self.assertIn("dow_sin", df_time.columns)
        self.assertIn("month_sin", df_time.columns)
        
        # Sine/Cosine bounds checking [-1.0, 1.0]
        self.assertTrue((df_time["hour_sin"] >= -1.0).all() and (df_time["hour_sin"] <= 1.0).all())
        self.assertTrue((df_time["hour_cos"] >= -1.0).all() and (df_time["hour_cos"] <= 1.0).all())

    def test_autocorrelation_lags(self):
        df_clustered = self.builder.assign_spatial_clusters(self.ola_df.head(500))
        df_lags = self.builder.build_autocorrelation_lags(df_clustered, lags=[1, 24])
        self.assertIn("lag_1h", df_lags.columns)
        self.assertIn("lag_24h", df_lags.columns)

    def test_transform_pipeline_e2e(self):
        self.builder.fit_spatial_clusters(self.gps_df)
        feature_matrix = self.builder.transform(
            self.ola_df.head(1000),
            weather_df=self.weather_df.head(1000),
            drop_na=True
        )

        self.assertIsInstance(feature_matrix, pd.DataFrame)
        self.assertGreater(len(feature_matrix), 0)
        self.assertIn("cluster_id", feature_matrix.columns)
        self.assertIn("target_h1", feature_matrix.columns)
        self.assertIn("target_h4", feature_matrix.columns)
        self.assertIn("temp_roll_mean_3h", feature_matrix.columns)
        self.assertFalse(feature_matrix.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
