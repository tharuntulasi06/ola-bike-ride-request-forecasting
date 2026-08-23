import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt kagglehub import for dynamic Kaggle dataset downloading
try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False


class OlaDataLoader:
    """Production data pipeline for spatiotemporal micro-mobility demand forecasting."""

    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

    CHENNAI_HOTSPOTS = {
        "chennai_central": (13.0827, 80.2707),
        "t_nagar": (13.0418, 80.2341),
        "omr_it_corridor": (12.9645, 80.2443),
        "velachery": (12.9750, 80.2207),
        "guindy_kathipara": (13.0067, 80.2020),
        "cmbt_anna_nagar": (13.0850, 80.2101),
    }

    NYC_HOTSPOTS = {
        "midtown": (40.7549, -73.9840),
        "financial_district": (40.7075, -74.0089),
        "williamsburg": (40.7081, -73.9571),
        "jfk_airport": (40.6413, -73.7781),
        "uew_harlem": (40.8075, -73.9465),
    }

    CITY_BOUNDS = {
        "chennai": (12.8, 13.2, 80.15, 80.35),
        "nyc": (40.5, 40.9, -74.25, -73.70),
    }

    KAGGLE_HANDLES = {
        "ola": "palvinder2006/ola-bike-ride-request",
        "uber_gps": "fivethirtyeight/uber-pickups-in-new-york-city",
        "nyc_tlc": "anaghbar81/tlc-trip-record-data",
        "weather": "muthuj7/weather-dataset",
    }

    def __init__(self, data_dir: Optional[str] = None, default_city: str = "chennai"):
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self.default_city = default_city.lower()
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_from_kaggle(self, dataset_handle: str) -> Optional[pd.DataFrame]:
        """Dynamically fetches public Kaggle dataset directly via kagglehub API."""
        if not HAS_KAGGLEHUB:
            logger.warning("kagglehub package is not installed. Falling back to local/synthetic pipeline.")
            return None

        try:
            logger.info(f"Fetching dataset directly from Kaggle API handle: '{dataset_handle}'...")
            download_path = Path(kagglehub.dataset_download(dataset_handle))
            data_files = list(download_path.glob("*.parquet")) + list(download_path.glob("*.csv"))
            if data_files:
                target_file = data_files[0]
                logger.info(f"Successfully downloaded {target_file.name} from Kaggle cache.")
                if target_file.suffix == ".parquet":
                    return pd.read_parquet(target_file)
                return pd.read_csv(target_file)
        except Exception as e:
            logger.warning(f"Kaggle API fetch for '{dataset_handle}' encountered an issue: {e}. Falling back...")
        return None

    def generate_synthetic_ola_data(
        self, n_hours: int = 8760, city: str = "chennai", output_filename: str = "ola_bike_requests.parquet"
    ) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range("2024-01-01 00:00:00", periods=n_hours, freq="1h")

        hour = dates.hour
        dow = dates.dayofweek
        month = dates.month

        season = np.select([month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])], [1, 2, 3], default=4)

        if city.lower() == "chennai":
            temp_base = 29.0 + 5.0 * np.sin(2 * np.pi * (month - 3) / 12)
            temp_diurnal = 4.0 * np.sin(2 * np.pi * (hour - 8) / 24)
            temp = np.clip(temp_base + temp_diurnal + np.random.normal(0, 1.2, size=n_hours), 18.0, 43.0)
            atemp = temp + np.random.normal(1.5, 1.0, size=n_hours)
            humidity = np.clip(72.0 - 10.0 * np.sin(2 * np.pi * (hour - 8) / 24) + np.random.normal(0, 5, n_hours), 30.0, 98.0)
            rain_prob = np.where(month.isin([10, 11]), 0.28, np.where(humidity > 80, 0.12, 0.04))
        else:
            temp_base = 22.0 + 8.0 * np.sin(2 * np.pi * (month - 3) / 12)
            temp_diurnal = 5.0 * np.sin(2 * np.pi * (hour - 8) / 24)
            temp = np.clip(temp_base + temp_diurnal + np.random.normal(0, 1.5, size=n_hours), 5.0, 38.0)
            atemp = temp + np.random.normal(0.5, 1.0, size=n_hours)
            humidity = np.clip(60.0 - 15.0 * np.sin(2 * np.pi * (hour - 8) / 24) + np.random.normal(0, 5, n_hours), 15.0, 95.0)
            rain_prob = np.where(humidity > 80, 0.35, 0.08)

        windspeed = np.clip(10.0 + np.random.exponential(6.0, size=n_hours), 0.0, 50.0)
        weather_situation = np.where(
            np.random.binomial(1, rain_prob) == 0,
            np.random.choice([1, 2], size=n_hours, p=[0.7, 0.3]),
            np.random.choice([3, 4], size=n_hours, p=[0.8, 0.2]),
        )

        morning = 190.0 * np.exp(-((hour - 8.5) ** 2) / 2.5)
        evening = 230.0 * np.exp(-((hour - 17.5) ** 2) / 3.0)
        offpeak = 30.0 + 15.0 * np.sin(2 * np.pi * hour / 24)

        is_weekend = (dow >= 5).astype(int)
        commute_mult = np.where(is_weekend == 1, 0.45, 1.0)
        leisure = np.where(is_weekend == 1, 85.0 * np.exp(-((hour - 14) ** 2) / 10.0), 0.0)
        weather_mult = np.select([weather_situation == 1, weather_situation == 2, weather_situation == 3], [1.1, 1.0, 0.60], default=0.20)

        expected = (offpeak + (morning + evening) * commute_mult + leisure) * weather_mult
        cnt = np.random.poisson(np.clip(expected, 2.0, None))

        casual_ratio = np.clip(np.where(is_weekend == 1, 0.35, 0.15) + np.random.normal(0, 0.05, n_hours), 0.05, 0.6)
        casual = np.round(cnt * casual_ratio).astype(int)
        registered = np.maximum(0, cnt - casual)

        df = pd.DataFrame(
            {
                "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
                "city": city.lower(),
                "season": season,
                "weather_situation": weather_situation,
                "temp": np.round(temp, 2),
                "atemp": np.round(atemp, 2),
                "humidity": np.round(humidity, 2),
                "windspeed": np.round(windspeed, 2),
                "casual": casual,
                "registered": registered,
                "cnt": cnt,
            }
        )

        out_path = self.raw_dir / output_filename
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return df

    def generate_synthetic_gps_data(
        self, n_samples: int = 50000, city: str = "chennai", output_filename: str = "gps_pickups.parquet"
    ) -> pd.DataFrame:
        np.random.seed(42)
        start = pd.Timestamp("2024-04-01 00:00:00")
        timestamps = start + pd.to_timedelta(np.random.randint(0, 30 * 24 * 3600, size=n_samples), unit="s")

        hotspots = self.CHENNAI_HOTSPOTS if city.lower() == "chennai" else self.NYC_HOTSPOTS
        centers = list(hotspots.values())
        names = list(hotspots.keys())

        chosen = np.random.choice(len(centers), size=n_samples)
        lats = [np.random.normal(centers[i][0], 0.012) for i in chosen]
        lons = [np.random.normal(centers[i][1], 0.015) for i in chosen]
        landmarks = [names[i] for i in chosen]

        df = pd.DataFrame(
            {
                "Date/Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
                "city": city.lower(),
                "Lat": np.round(lats, 6),
                "Lon": np.round(lons, 6),
                "zone_landmark": landmarks,
                "Base": np.random.choice(["OLA_BIKE_01", "OLA_BIKE_02", "RAPIDO_01"], size=n_samples),
            }
        )

        out_path = self.raw_dir / output_filename
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return df

    def preprocess_ola_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        if "datetime" not in df.columns and "date/time" in df.columns:
            df.rename(columns={"date/time": "datetime"}, inplace=True)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

        grid = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="1h")
        df = df.set_index("datetime").reindex(grid)
        df.index.name = "datetime"
        df = df.reset_index()

        for col in ["temp", "atemp", "humidity", "windspeed"]:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").bfill().ffill()

        for col in ["season", "weather_situation"]:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().astype(int)

        for col in ["casual", "registered", "cnt"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).clip(lower=0).astype(int)

        if "cnt" not in df.columns and "casual" in df.columns and "registered" in df.columns:
            df["cnt"] = df["casual"] + df["registered"]

        if "city" in df.columns:
            df["city"] = df["city"].ffill().bfill().fillna(self.default_city)

        return df

    def load_ola_data(self, file_path: Optional[str] = None, city: str = "chennai", force_synthetic: bool = False) -> pd.DataFrame:
        proc_parquet = self.processed_dir / "ola_bike_requests_clean.parquet"
        proc_csv = self.processed_dir / "ola_bike_requests_clean.csv"

        if not force_synthetic:
            if proc_parquet.exists():
                return pd.read_parquet(proc_parquet)
            elif proc_csv.exists():
                return pd.read_csv(proc_csv)

        raw_df = None
        if not force_synthetic:
            raw_df = self.fetch_from_kaggle(self.KAGGLE_HANDLES["ola"])

        if raw_df is None:
            raw_df = self.generate_synthetic_ola_data(city=city, output_filename="ola_bike_requests.parquet")

        clean_df = self.preprocess_ola_data(raw_df)
        clean_df.to_parquet(proc_parquet, index=False)
        clean_df.to_csv(proc_csv, index=False)
        return clean_df

    def load_uber_gps_data(self, file_path: Optional[str] = None, city: str = "chennai", force_synthetic: bool = False) -> pd.DataFrame:
        proc_parquet = self.processed_dir / "uber_gps_pickups_clean.parquet"
        proc_csv = self.processed_dir / "uber_gps_pickups_clean.csv"

        if not force_synthetic:
            if proc_parquet.exists():
                return pd.read_parquet(proc_parquet)
            elif proc_csv.exists():
                return pd.read_csv(proc_csv)

        raw_df = None
        if not force_synthetic:
            raw_df = self.fetch_from_kaggle(self.KAGGLE_HANDLES["uber_gps"])

        if raw_df is None:
            raw_df = self.generate_synthetic_gps_data(city=city, output_filename="uber_nyc_pickups.parquet")

        dt_col = "Date/Time" if "Date/Time" in raw_df.columns else "datetime"
        raw_df["datetime"] = pd.to_datetime(raw_df[dt_col])
        raw_df = raw_df.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)

        if city.lower() in self.CITY_BOUNDS:
            min_lat, max_lat, min_lon, max_lon = self.CITY_BOUNDS[city.lower()]
            mask = (raw_df["Lat"] >= min_lat) & (raw_df["Lat"] <= max_lat) & (raw_df["Lon"] >= min_lon) & (raw_df["Lon"] <= max_lon)
            raw_df = raw_df[mask].reset_index(drop=True)

        raw_df.to_parquet(proc_parquet, index=False)
        raw_df.to_csv(proc_csv, index=False)
        return raw_df

    def parse_raw_gps_coordinates(
        self, df: pd.DataFrame, city: str = "chennai", filter_bounds: bool = True, tag_nearest_hotspot: bool = True
    ) -> pd.DataFrame:
        df = df.copy()
        dt_col = "Date/Time" if "Date/Time" in df.columns else ("datetime" if "datetime" in df.columns else df.columns[0])
        df["datetime"] = pd.to_datetime(df[dt_col])

        lat_cols = [c for c in df.columns if any(k in c.lower() for k in ["lat", "latitude"])]
        lon_cols = [c for c in df.columns if any(k in c.lower() for k in ["lon", "lng", "longitude"])]

        if not lat_cols or not lon_cols:
            raise KeyError("Could not identify Lat/Lon columns in input DataFrame.")

        df.rename(columns={lat_cols[0]: "Lat", lon_cols[0]: "Lon"}, inplace=True)
        df = df.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)

        if filter_bounds and city.lower() in self.CITY_BOUNDS:
            min_lat, max_lat, min_lon, max_lon = self.CITY_BOUNDS[city.lower()]
            mask = (df["Lat"] >= min_lat) & (df["Lat"] <= max_lat) & (df["Lon"] >= min_lon) & (df["Lon"] <= max_lon)
            df = df[mask].reset_index(drop=True)

        if tag_nearest_hotspot:
            hotspots = self.CHENNAI_HOTSPOTS if city.lower() == "chennai" else self.NYC_HOTSPOTS
            names = list(hotspots.keys())
            coords = np.array(list(hotspots.values()))

            lats, lons = df["Lat"].values, df["Lon"].values
            dist_sq = (lats[:, None] - coords[:, 0]) ** 2 + (lons[:, None] - coords[:, 1]) ** 2
            nearest_idx = np.argmin(dist_sq, axis=1)
            df["zone_landmark"] = [names[i] for i in nearest_idx]

        return df

    def create_zonal_demand_matrix(
        self, df: pd.DataFrame, fill_zero: bool = True, as_pivot: bool = False
    ) -> pd.DataFrame:
        df = df.copy()
        dt_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
        df["pickup_hour"] = pd.to_datetime(df[dt_col]).dt.floor("1h")

        counts = df.groupby(["pickup_hour", "PULocationID"]).size().reset_index(name="trip_count")

        if fill_zero:
            hours = pd.date_range(counts["pickup_hour"].min(), counts["pickup_hour"].max(), freq="1h")
            zones = counts["PULocationID"].unique()
            full_idx = pd.MultiIndex.from_product([hours, zones], names=["pickup_hour", "PULocationID"])
            counts = counts.set_index(["pickup_hour", "PULocationID"]).reindex(full_idx, fill_value=0).reset_index()

        if as_pivot:
            counts["pickup_hour"] = counts["pickup_hour"].astype(str)
            return counts.pivot(index="pickup_hour", columns="PULocationID", values="trip_count").fillna(0).astype(int)

        counts["pickup_hour"] = counts["pickup_hour"].astype(str)
        return counts

    def generate_synthetic_weather_holiday_data(
        self, n_hours: int = 8760, output_filename: str = "weather_holiday_features.parquet"
    ) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range("2024-01-01 00:00:00", periods=n_hours, freq="1h")
        is_weekend = (dates.dayofweek >= 5).astype(int)

        holidays = pd.to_datetime(
            ["2024-01-01", "2024-01-26", "2024-03-25", "2024-04-11", "2024-05-01", "2024-08-15", "2024-10-02", "2024-10-12", "2024-11-01", "2024-12-25"]
        ).date
        is_holiday = np.isin(dates.date, holidays).astype(int)

        rain_mask = np.random.binomial(1, 0.08, size=n_hours)
        rain_1h = np.where(rain_mask == 1, np.random.exponential(3.5, size=n_hours), 0.0)

        visibility = np.where(rain_1h > 5.0, np.random.uniform(500, 3000, size=n_hours), np.random.uniform(8000, 10000, size=n_hours))
        pressure = 1013.25 + np.random.normal(0, 5.0, size=n_hours)

        df = pd.DataFrame(
            {
                "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
                "rain_1h": np.round(rain_1h, 2),
                "visibility": np.round(visibility, 1),
                "pressure": np.round(pressure, 1),
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
            }
        )

        out_path = self.raw_dir / output_filename
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return df

    def generate_synthetic_nyc_tlc_data(self, n_samples: int = 20000, output_filename: str = "nyc_tlc_trips.parquet") -> pd.DataFrame:
        np.random.seed(42)
        start = pd.Timestamp("2024-04-01 00:00:00")
        timestamps = start + pd.to_timedelta(np.random.randint(0, 30 * 24 * 3600, size=n_samples), unit="s")

        pu = np.random.randint(1, 264, size=n_samples)
        do = np.random.randint(1, 264, size=n_samples)
        distances = np.round(np.random.exponential(3.5, size=n_samples) + 0.5, 2)
        fares = np.clip(np.round(2.5 + distances * 3.0 + np.random.normal(0, 2.0, n_samples), 2), 3.0, 150.0)

        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
                "PULocationID": pu,
                "DOLocationID": do,
                "trip_distance": distances,
                "passenger_count": np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.03]),
                "fare_amount": fares,
                "congestion_surcharge": np.random.choice([0.0, 2.75], size=n_samples, p=[0.3, 0.7]),
            }
        )

        out_path = self.raw_dir / output_filename
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return df

    def load_weather_holiday_data(self, file_path: Optional[str] = None, force_synthetic: bool = False) -> pd.DataFrame:
        proc_parquet = self.processed_dir / "weather_holiday_clean.parquet"
        proc_csv = self.processed_dir / "weather_holiday_clean.csv"

        if not force_synthetic:
            if proc_parquet.exists():
                return pd.read_parquet(proc_parquet)
            elif proc_csv.exists():
                return pd.read_csv(proc_csv)

        raw_df = None
        if not force_synthetic:
            raw_df = self.fetch_from_kaggle(self.KAGGLE_HANDLES["weather"])

        if raw_df is None:
            raw_df = self.generate_synthetic_weather_holiday_data(output_filename="weather_holiday_features.parquet")

        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
        raw_df.to_parquet(proc_parquet, index=False)
        raw_df.to_csv(proc_csv, index=False)
        return raw_df

    def load_nyc_tlc_data(self, file_path: Optional[str] = None, force_synthetic: bool = False) -> pd.DataFrame:
        proc_parquet = self.processed_dir / "nyc_tlc_benchmark_clean.parquet"
        proc_csv = self.processed_dir / "nyc_tlc_benchmark_clean.csv"

        if not force_synthetic:
            if proc_parquet.exists():
                return pd.read_parquet(proc_parquet)
            elif proc_csv.exists():
                return pd.read_csv(proc_csv)

        raw_df = None
        if not force_synthetic:
            raw_df = self.fetch_from_kaggle(self.KAGGLE_HANDLES["nyc_tlc"])

        if raw_df is None:
            raw_df = self.generate_synthetic_nyc_tlc_data(output_filename="nyc_tlc_trips.parquet")

        raw_df["tpep_pickup_datetime"] = pd.to_datetime(raw_df["tpep_pickup_datetime"])
        raw_df["pickup_hour"] = raw_df["tpep_pickup_datetime"].dt.floor("1h")

        zonal_demand = raw_df.groupby(["pickup_hour", "PULocationID"]).size().reset_index(name="trip_count")
        zonal_demand.to_parquet(proc_parquet, index=False)
        zonal_demand.to_csv(proc_csv, index=False)
        return zonal_demand

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        time_col = "datetime" if "datetime" in df.columns else ("pickup_hour" if "pickup_hour" in df.columns else None)
        summary = {
            "total_records": len(df),
            "start_time": str(df[time_col].min()) if time_col else None,
            "end_time": str(df[time_col].max()) if time_col else None,
            "null_count": int(df.isnull().sum().sum()),
            "columns": list(df.columns),
        }
        if "cnt" in df.columns:
            summary["target_stats"] = {
                "mean": float(df["cnt"].mean()),
                "std": float(df["cnt"].std()),
                "min": int(df["cnt"].min()),
                "max": int(df["cnt"].max()),
                "zero_count": int((df["cnt"] == 0).sum()),
            }
        return summary


if __name__ == "__main__":
    loader = OlaDataLoader(default_city="chennai")
    df_ola = loader.load_ola_data(city="chennai", force_synthetic=True)
    df_gps = loader.load_uber_gps_data(city="chennai", force_synthetic=True)
    df_weather = loader.load_weather_holiday_data(force_synthetic=True)
    df_tlc = loader.load_nyc_tlc_data(force_synthetic=True)

    print("\n--- Ola Demand Summary (Chennai) ---")
    print(loader.get_data_summary(df_ola))
    print("\n--- GPS Pickups Sample ---")
    print(df_gps[["Date/Time", "Lat", "Lon", "zone_landmark"]].head())
