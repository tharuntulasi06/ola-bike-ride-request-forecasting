import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.db import DatabaseManager, HAS_DUCKDB


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager()

    def test_duckdb_availability(self):
        self.assertTrue(HAS_DUCKDB, "DuckDB should be installed in environment.")
        self.assertIsNotNone(self.db.conn)

    def test_query_spatial_zone_aggregates(self):
        data_path = Path("data/processed/spatiotemporal_features_clean.parquet")
        if data_path.exists():
            df_aggs = self.db.get_spatial_zone_aggregates()
            self.assertIsInstance(df_aggs, pd.DataFrame)
            if not df_aggs.empty:
                self.assertIn("cluster_id", df_aggs.columns)
                self.assertIn("mean_hourly_demand", df_aggs.columns)


if __name__ == "__main__":
    unittest.main()
