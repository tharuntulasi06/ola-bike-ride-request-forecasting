from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import joblib

logger = logging.getLogger(__name__)


def haversine_distance_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


class FeatureBuilder:
    """Spatiotemporal feature extraction and spatial graph construction engine."""

    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[MiniBatchKMeans] = None
        self.cluster_centroids: Optional[np.ndarray] = None
        self.adjacency_matrix: Optional[np.ndarray] = None

    def fit_spatial_clusters(self, gps_df: pd.DataFrame, save_path: Optional[str] = None) -> "FeatureBuilder":
        coords = gps_df[["Lat", "Lon"]].dropna().values
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=4096,
            random_state=self.random_state,
            n_init="auto"
        )
        self.kmeans.fit(coords)
        self.cluster_centroids = self.kmeans.cluster_centers_
        self.adjacency_matrix = self._compute_spatial_adjacency(self.cluster_centroids)

        if save_path:
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"kmeans": self.kmeans, "centroids": self.cluster_centroids, "adj_matrix": self.adjacency_matrix}, out_file)

        return self

    def _compute_spatial_adjacency(self, centroids: np.ndarray, sigma: float = 5.0, kappa: float = 15.0) -> np.ndarray:
        K = len(centroids)
        W = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                if i != j:
                    dist = haversine_distance_km(centroids[i, 0], centroids[i, 1], centroids[j, 0], centroids[j, 1])
                    if dist <= kappa:
                        W[i, j] = np.exp(-((dist / sigma) ** 2))
        return W

    def assign_spatial_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Lat" in df.columns and "Lon" in df.columns and self.kmeans is not None:
            df["cluster_id"] = self.kmeans.predict(df[["Lat", "Lon"]].values)
        elif "cluster_id" not in df.columns:
            records = []
            for k in range(self.n_clusters):
                sub = df.copy()
                sub["cluster_id"] = k
                records.append(sub)
            df = pd.concat(records, ignore_index=True)

        return df

    def build_cyclical_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        dt = pd.to_datetime(df["datetime"])

        hour = dt.dt.hour
        dow = dt.dt.dayofweek
        month = dt.dt.month

        df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_peak_hour"] = hour.isin([8, 9, 17, 18, 19]).astype(int)

        return df

    def build_autocorrelation_lags(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 24, 48, 168]) -> pd.DataFrame:
        df = df.copy().sort_values(["cluster_id", "datetime"]).reset_index(drop=True)
        grouped = df.groupby("cluster_id")["cnt"]

        for lag in lags:
            df[f"lag_{lag}h"] = grouped.shift(lag)

        return df

    def build_rolling_weather_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 24]) -> pd.DataFrame:
        df = df.copy().sort_values(["cluster_id", "datetime"]).reset_index(drop=True)

        if "atemp" in df.columns and "temp" in df.columns:
            df["temp_feels_diff"] = df["atemp"] - df["temp"]

        for w in windows:
            rolled = df.groupby("cluster_id")["temp"].shift(1).rolling(window=w)
            df[f"temp_roll_mean_{w}h"] = rolled.mean()
            df[f"temp_roll_std_{w}h"] = rolled.std().fillna(0.0)

            if "humidity" in df.columns:
                df[f"humidity_roll_mean_{w}h"] = df.groupby("cluster_id")["humidity"].shift(1).rolling(window=w).mean()

        if "rain_1h" in df.columns:
            df["rain_peak_interaction"] = df["rain_1h"] * df["is_peak_hour"]

        return df

    def build_multi_step_targets(self, df: pd.DataFrame, horizons: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        df = df.copy().sort_values(["cluster_id", "datetime"]).reset_index(drop=True)
        grouped = df.groupby("cluster_id")["cnt"]

        for h in horizons:
            df[f"target_h{h}"] = grouped.shift(-h)

        return df

    def transform(
        self,
        ola_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        drop_na: bool = True
    ) -> pd.DataFrame:
        df = ola_df.copy()

        if weather_df is not None:
            weather_df = weather_df.copy()
            weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            overlap_cols = [c for c in weather_df.columns if c in df.columns and c != "datetime"]
            weather_df = weather_df.drop(columns=overlap_cols)
            df = pd.merge(df, weather_df, on="datetime", how="left")

        df = self.assign_spatial_clusters(df)
        df = self.build_cyclical_time_features(df)
        df = self.build_autocorrelation_lags(df)
        df = self.build_rolling_weather_features(df)
        df = self.build_multi_step_targets(df)

        if drop_na:
            df = df.dropna().reset_index(drop=True)

        return df


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.data_loader import OlaDataLoader

    loader = OlaDataLoader(default_city="chennai")
    ola_df = loader.load_ola_data()
    gps_df = loader.load_uber_gps_data()
    weather_df = loader.load_weather_holiday_data()

    builder = FeatureBuilder(n_clusters=6)
    builder.fit_spatial_clusters(gps_df)
    features_df = builder.transform(ola_df, weather_df=weather_df)

    print("\n--- Spatiotemporal Feature Matrix Summary ---")
    print(f"Shape: {features_df.shape}")
    print(f"Sample Columns: {list(features_df.columns[:10])}")
