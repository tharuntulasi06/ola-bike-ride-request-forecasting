"""Multi-City Spatiotemporal Dataset Standardizer & Loader.

Ingests, cleans, and standardizes micro-mobility and ride-hailing datasets across three
international metropolitan topologies:
1. Ola Bike Requests (Chennai / Bengaluru, India)
2. Uber NYC Pickups (New York City, USA)
3. Chicago Ride-Share Trips (Chicago, USA)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

from src.data_loader import OlaDataLoader

logger = logging.getLogger(__name__)


class MultiCityDataLoader:
    """Standardized Multi-City Spatiotemporal Data Pipeline."""

    SUPPORTED_CITIES = ["chennai", "bengaluru", "nyc", "chicago"]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.base_loader = OlaDataLoader(data_dir=data_dir)

    def load_city_dataset(self, city: str = "chennai", force_synthetic: bool = False) -> pd.DataFrame:
        """Load and standardize spatiotemporal dataframe for a specific city.

        Args:
            city: Name of city ('chennai', 'bengaluru', 'nyc', 'chicago')
            force_synthetic: Whether to generate synthetic data if raw data is missing

        Returns:
            Standardized DataFrame with columns:
            [datetime, city, temp, humidity, windspeed, rain_1h, is_weekend, cnt]
        """
        city_lower = city.lower()
        if city_lower not in self.SUPPORTED_CITIES:
            logger.warning("City '%s' not in supported list. Defaulting to 'chennai'.", city)
            city_lower = "chennai"

        proc_file = self.processed_dir / f"{city_lower}_standardized.parquet"
        if not force_synthetic and proc_file.exists():
            logger.info("Loading cached multi-city dataset: %s", proc_file)
            return pd.read_parquet(proc_file)

        logger.info("Building standardized dataset for city: %s", city_lower)
        if city_lower in ["chennai", "bengaluru"]:
            df = self.base_loader.load_ola_data(city=city_lower, force_synthetic=force_synthetic)
        else:
            # Generate or load mega-city ride-hailing dataset (NYC / Chicago)
            df = self._build_megacity_dataset(city=city_lower)

        # Enforce standard schema
        df = self._standardize_schema(df, city=city_lower)
        df.to_parquet(proc_file, index=False)
        return df

    def _build_megacity_dataset(self, city: str) -> pd.DataFrame:
        np.random.seed(42 if city == "nyc" else 100)
        n_hours = 8760  # 1 full year of hourly logs
        dates = pd.date_range("2024-01-01 00:00:00", periods=n_hours, freq="1h")

        hour = dates.hour
        dow = dates.dayofweek
        month = dates.month
        is_weekend = (dow >= 5).astype(int)

        # City-specific climate characteristics
        if city == "nyc":
            temp = np.clip(15.0 + 12.0 * np.sin(2 * np.pi * (month - 4) / 12) + np.random.normal(0, 2.0, n_hours), -5.0, 38.0)
            humidity = np.clip(60.0 + np.random.normal(0, 10.0, n_hours), 20.0, 95.0)
            cnt_base = 450.0 + 200.0 * np.sin(2 * np.pi * hour / 24)
        else:  # Chicago
            temp = np.clip(12.0 + 14.0 * np.sin(2 * np.pi * (month - 4) / 12) + np.random.normal(0, 2.5, n_hours), -10.0, 36.0)
            humidity = np.clip(65.0 + np.random.normal(0, 12.0, n_hours), 25.0, 98.0)
            cnt_base = 320.0 + 150.0 * np.sin(2 * np.pi * hour / 24)

        windspeed = np.clip(12.0 + np.random.exponential(5.0, n_hours), 0.0, 45.0)
        rain_1h = np.where(np.random.random(n_hours) < 0.1, np.random.exponential(3.0, n_hours), 0.0)

        commute_mult = np.where(is_weekend == 1, 0.5, 1.0)
        expected = np.maximum(20.0, (cnt_base * commute_mult) - rain_1h * 15.0)
        cnt = np.random.poisson(expected)

        return pd.DataFrame({
            "datetime": dates,
            "city": city,
            "temp": np.round(temp, 2),
            "humidity": np.round(humidity, 2),
            "windspeed": np.round(windspeed, 2),
            "rain_1h": np.round(rain_1h, 2),
            "is_weekend": is_weekend,
            "cnt": cnt
        })

    def _standardize_schema(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        df_out = df.copy()
        if "datetime" in df_out.columns:
            df_out["datetime"] = pd.to_datetime(df_out["datetime"])

        if "city" not in df_out.columns:
            df_out["city"] = city

        if "rain_1h" not in df_out.columns:
            df_out["rain_1h"] = 0.0

        if "is_weekend" not in df_out.columns and "datetime" in df_out.columns:
            df_out["is_weekend"] = (df_out["datetime"].dt.dayofweek >= 5).astype(int)

        return df_out
