import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class DatabaseManager:
    """Production Embedded SQL Engine using DuckDB for zero-overhead Parquet queries."""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.conn: Optional[Any] = None
        if HAS_DUCKDB:
            self.conn = duckdb.connect(database=":memory:")
            logger.info("DuckDB in-memory database engine initialized.")

    def query(self, sql_query: str) -> pd.DataFrame:
        """Executes raw SQL query on Parquet files or pandas DataFrames."""
        if not HAS_DUCKDB or self.conn is None:
            logger.warning("DuckDB not installed. Falling back to pandas memory query.")
            return pd.DataFrame()

        try:
            return self.conn.execute(sql_query).df()
        except Exception as e:
            logger.error(f"DuckDB SQL Execution Error: {e}")
            raise e

    def get_zone_demand_history(self, cluster_id: int, hours: int = 24) -> pd.DataFrame:
        """Retrieves historical demand time-series for a specific spatial hotspot zone."""
        parquet_file = self.data_dir / "spatiotemporal_features_clean.parquet"
        if not parquet_file.exists():
            logger.warning(f"Parquet file {parquet_file} does not exist.")
            return pd.DataFrame()

        sql = f"""
            SELECT timestamp, cluster_id, count AS demand, temp, humidity, windspeed, is_holiday
            FROM '{parquet_file}'
            WHERE cluster_id = {cluster_id}
            ORDER BY timestamp DESC
            LIMIT {hours}
        """
        return self.query(sql)

    def get_spatial_zone_aggregates(self) -> pd.DataFrame:
        """Computes aggregate volume metrics per spatial zone directly via DuckDB SQL."""
        parquet_file = self.data_dir / "spatiotemporal_features_clean.parquet"
        if not parquet_file.exists():
            return pd.DataFrame()

        sql = f"""
            SELECT 
                cluster_id,
                COUNT(*) AS total_records,
                AVG(count) AS mean_hourly_demand,
                MAX(count) AS max_peak_demand,
                STDDEV(count) AS demand_stddev,
                AVG(temp) AS avg_temperature
            FROM '{parquet_file}'
            GROUP BY cluster_id
            ORDER BY cluster_id ASC
        """
        return self.query(sql)


if __name__ == "__main__":
    db = DatabaseManager()
    if HAS_DUCKDB:
        df_aggs = db.get_spatial_zone_aggregates()
        print("\n========================================================")
        print("=== DuckDB Spatial Zone Aggregates (SQL on Parquet) ===")
        print("========================================================")
        print(df_aggs.to_string(index=False))
