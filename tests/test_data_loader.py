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

class TestOlaDataLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.loader = OlaDataLoader(data_dir=self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_synthetic_ola_generation(self):
        df = self.loader.generate_synthetic_ola_data(n_hours=100)
        self.assertEqual(len(df), 100)
        self.assertIn("datetime", df.columns)
        self.assertIn("cnt", df.columns)
        self.assertTrue((df["cnt"] >= 0).all())
        self.assertTrue((df["casual"] >= 0).all())
        self.assertTrue((df["registered"] >= 0).all())

    def test_synthetic_chennai_gps_generation(self):
        df = self.loader.generate_synthetic_gps_data(n_samples=500, city="chennai")
        self.assertEqual(len(df), 500)
        self.assertIn("Lat", df.columns)
        self.assertIn("Lon", df.columns)
        self.assertIn("Date/Time", df.columns)
        # Verify Chennai lat/lon bounds (~12.9°N - 13.2°N, 80.15°E - 80.35°E)
        self.assertTrue((df["Lat"] >= 12.8).all() and (df["Lat"] <= 13.2).all())
        self.assertTrue((df["Lon"] >= 80.15).all() and (df["Lon"] <= 80.35).all())



    def test_ola_preprocessing_and_resampling(self):
        dates = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 04:00:00"])
        raw_df = pd.DataFrame({
            "datetime": dates,
            "season": [1, 1, 1],
            "weather_situation": [1, 1, 2],
            "temp": [20.0, 22.0, 28.0],
            "cnt": [50, 60, 100]
        })

        clean_df = self.loader.preprocess_ola_data(raw_df)

        self.assertEqual(len(clean_df), 5)
        self.assertEqual(clean_df["datetime"].min(), pd.Timestamp("2024-01-01 00:00:00"))
        self.assertEqual(clean_df["datetime"].max(), pd.Timestamp("2024-01-01 04:00:00"))

        temp_02 = clean_df.loc[clean_df["datetime"] == "2024-01-01 02:00:00", "temp"].values[0]
        temp_03 = clean_df.loc[clean_df["datetime"] == "2024-01-01 03:00:00", "temp"].values[0]
        self.assertAlmostEqual(temp_02, 24.0, places=2)
        self.assertAlmostEqual(temp_03, 26.0, places=2)

        cnt_02 = clean_df.loc[clean_df["datetime"] == "2024-01-01 02:00:00", "cnt"].values[0]
        self.assertEqual(cnt_02, 0)

    def test_load_ola_data_e2e(self):
        df = self.loader.load_ola_data(force_synthetic=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 8760)
        summary = self.loader.get_data_summary(df)
        self.assertEqual(summary["null_count"], 0)
        self.assertEqual(summary["total_records"], 8760)

    def test_weather_holiday_generation_and_loading(self):
        df = self.loader.load_weather_holiday_data(force_synthetic=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 8760)
        self.assertIn("rain_1h", df.columns)
        self.assertIn("is_holiday", df.columns)
        self.assertIn("is_weekend", df.columns)
        self.assertTrue(df["is_holiday"].isin([0, 1]).all())
        self.assertTrue(df["is_weekend"].isin([0, 1]).all())

    def test_parse_raw_gps_coordinates(self):
        # Raw data with non-standard column names, out-of-bounds coords, and invalid values
        raw_gps = pd.DataFrame({
            "pickup_latitude": [13.0827, 13.0418, 55.0000, 0.0, np.nan, 12.9645],
            "pickup_longitude": [80.2707, 80.2341, -10.0000, 0.0, 80.2443, 80.2443],
            "timestamp": [
                "2024-04-01 08:30:00",
                "2024-04-01 09:15:00",
                "2024-04-01 10:00:00",
                "2024-04-01 10:30:00",
                "2024-04-01 11:00:00",
                "2024-04-01 11:30:00"
            ]
        })

        parsed = self.loader.parse_raw_gps_coordinates(raw_gps, city="chennai", filter_bounds=True, tag_nearest_hotspot=True)
        
        # Valid Chennai coordinates: 3 rows (indices 0, 1, 5)
        self.assertEqual(len(parsed), 3)
        self.assertIn("Lat", parsed.columns)
        self.assertIn("Lon", parsed.columns)
        self.assertIn("datetime", parsed.columns)
        self.assertIn("zone_landmark", parsed.columns)
        self.assertEqual(parsed.iloc[0]["zone_landmark"], "chennai_central")
        self.assertEqual(parsed.iloc[1]["zone_landmark"], "t_nagar")
        self.assertEqual(parsed.iloc[2]["zone_landmark"], "omr_it_corridor")

    def test_create_zonal_demand_matrix(self):
        # Raw trips for 2 zones across 2 discrete hours
        raw_trips = pd.DataFrame({
            "tpep_pickup_datetime": [
                "2024-04-01 08:10:00",
                "2024-04-01 08:45:00",
                "2024-04-01 09:15:00"
            ],
            "PULocationID": [1, 1, 2]
        })

        # Test long format with zero-fill
        zonal_long = self.loader.create_zonal_demand_matrix(raw_trips, fill_zero=True, as_pivot=False)
        self.assertEqual(len(zonal_long), 4)  # 2 hours x 2 zones = 4
        self.assertIn("trip_count", zonal_long.columns)
        
        # Check zone 2 at 08:00 (should be 0)
        h8_z2 = zonal_long[(zonal_long["pickup_hour"] == "2024-04-01 08:00:00") & (zonal_long["PULocationID"] == 2)]["trip_count"].values[0]
        self.assertEqual(h8_z2, 0)

        # Check zone 1 at 08:00 (should be 2)
        h8_z1 = zonal_long[(zonal_long["pickup_hour"] == "2024-04-01 08:00:00") & (zonal_long["PULocationID"] == 1)]["trip_count"].values[0]
        self.assertEqual(h8_z1, 2)

        # Test pivot format
        zonal_pivot = self.loader.create_zonal_demand_matrix(raw_trips, fill_zero=True, as_pivot=True)
        self.assertEqual(zonal_pivot.shape, (2, 2))  # 2 hours x 2 zones
        self.assertEqual(zonal_pivot.loc["2024-04-01 08:00:00", 1], 2)
        self.assertEqual(zonal_pivot.loc["2024-04-01 08:00:00", 2], 0)

    def test_nyc_tlc_generation_and_loading(self):
        df = self.loader.load_nyc_tlc_data(force_synthetic=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("pickup_hour", df.columns)
        self.assertIn("PULocationID", df.columns)
        self.assertIn("trip_count", df.columns)
        self.assertTrue((df["trip_count"] >= 0).all())

if __name__ == "__main__":
    unittest.main()


