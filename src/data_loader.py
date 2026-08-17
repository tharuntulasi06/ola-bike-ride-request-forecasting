import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OlaDataLoader:
    """
    Data Ingestion and Preprocessing Pipeline for Ola Bike Ride Request Demand Forecasting.
    Supports a modular, city-aware architecture with Chennai as the primary case study
    and cross-city fallback capabilities (NYC/General).
    """

    DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

    # Define Chennai hotspot centroids (Lat, Lon)
    CHENNAI_HOTSPOTS = {
        "chennai_central": (13.0827, 80.2707),
        "t_nagar": (13.0418, 80.2341),
        "omr_it_corridor": (12.9645, 80.2443),
        "velachery": (12.9750, 80.2207),
        "guindy_kathipara": (13.0067, 80.2020),
        "cmbt_anna_nagar": (13.0850, 80.2101)
    }

    # Define NYC hotspot centroids (Lat, Lon)
    NYC_HOTSPOTS = {
        "midtown": (40.7549, -73.9840),
        "financial_district": (40.7075, -74.0089),
        "williamsburg": (40.7081, -73.9571),
        "jfk_airport": (40.6413, -73.7781),
        "uew_harlem": (40.8075, -73.9465)
    }

    # Bounding boxes for spatial coordinate validation
    CITY_BOUNDING_BOXES = {
        "chennai": {"lat_min": 12.80, "lat_max": 13.25, "lon_min": 80.10, "lon_max": 80.35},
        "nyc": {"lat_min": 40.50, "lat_max": 40.95, "lon_min": -74.25, "lon_max": -73.70},
    }

    def __init__(self, data_dir: Optional[str] = None, default_city: str = "chennai"):
        """
        Initialize DataLoader with raw and processed data directory paths and default city context.
        """
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self.default_city = default_city.lower()
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they do not exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def parse_raw_gps_coordinates(
        self,
        df: pd.DataFrame,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        time_col: Optional[str] = None,
        city: Optional[str] = None,
        filter_bounds: bool = True,
        tag_nearest_hotspot: bool = True
    ) -> pd.DataFrame:
        """
        Parses, cleans, and standardizes raw GPS spatial trip logs with dynamic schema resolution.
        
        Parameters:
            df: Input raw DataFrame with spatial coordinates and timestamps.
            lat_col: Explicit name of latitude column (or auto-detected).
            lon_col: Explicit name of longitude column (or auto-detected).
            time_col: Explicit name of timestamp column (or auto-detected).
            city: City identifier for bounding box and hotspot matching (default: self.default_city).
            filter_bounds: If True, filters out coordinates outside the city bounding box.
            tag_nearest_hotspot: If True, tags each record with its nearest landmark hotspot centroid.
            
        Returns:
            Clean standardized DataFrame with columns ['datetime', 'lat', 'lon', 'city', ...].
        """
        df = df.copy()
        target_city = (city or self.default_city).lower()
        cols_lower = {col.lower(): col for col in df.columns}

        # 1. Resolve Latitude Column
        if lat_col and lat_col in df.columns:
            actual_lat = lat_col
        else:
            lat_candidates = ["lat", "latitude", "pickup_lat", "pickup_latitude", "pu_lat", "y"]
            actual_lat = next((cols_lower[c] for c in lat_candidates if c in cols_lower), None)
            if not actual_lat:
                raise ValueError(f"Unable to auto-detect latitude column. Available: {list(df.columns)}")

        # 2. Resolve Longitude Column
        if lon_col and lon_col in df.columns:
            actual_lon = lon_col
        else:
            lon_candidates = ["lon", "lng", "longitude", "pickup_lon", "pickup_longitude", "pu_lon", "x"]
            actual_lon = next((cols_lower[c] for c in lon_candidates if c in cols_lower), None)
            if not actual_lon:
                raise ValueError(f"Unable to auto-detect longitude column. Available: {list(df.columns)}")

        # 3. Resolve Timestamp Column
        if time_col and time_col in df.columns:
            actual_time = time_col
        else:
            time_candidates = ["datetime", "date/time", "timestamp", "tpep_pickup_datetime", "pickup_datetime", "time", "date"]
            actual_time = next((cols_lower[c] for c in time_candidates if c in cols_lower), None)

        # 4. Standardize column names
        rename_map = {actual_lat: "Lat", actual_lon: "Lon"}
        if actual_time:
            rename_map[actual_time] = "datetime"
        df.rename(columns=rename_map, inplace=True)

        # 5. Clean numeric coordinates
        df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
        df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
        df = df.dropna(subset=["Lat", "Lon"])
        # Remove (0,0) or invalid lat/lon
        df = df[(df["Lat"].abs() > 0.001) & (df["Lon"].abs() > 0.001) & (df["Lat"].abs() <= 90.0) & (df["Lon"].abs() <= 180.0)]

        # 6. Parse timestamps
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            df["Date/Time"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # 7. Apply City Bounding Box Filtering
        if filter_bounds and target_city in self.CITY_BOUNDING_BOXES:
            bbox = self.CITY_BOUNDING_BOXES[target_city]
            initial_count = len(df)
            df = df[
                (df["Lat"] >= bbox["lat_min"]) & (df["Lat"] <= bbox["lat_max"]) &
                (df["Lon"] >= bbox["lon_min"]) & (df["Lon"] <= bbox["lon_max"])
            ]
            logger.info(f"Bounding box filtered {initial_count - len(df)} out-of-bounds GPS records for city='{target_city}'.")

        # 8. Tag Nearest Hotspot Landmark
        hotspots = self.CHENNAI_HOTSPOTS if target_city == "chennai" else (self.NYC_HOTSPOTS if target_city == "nyc" else None)
        if tag_nearest_hotspot and hotspots:
            hs_names = list(hotspots.keys())
            hs_coords = np.array(list(hotspots.values()))  # Shape: (K, 2) [lat, lon]
            pts = df[["Lat", "Lon"]].values  # Shape: (N, 2)
            if len(pts) > 0:
                # Euclidean distance in lat/lon space for fast hotspot assignment
                dists = np.sqrt(((pts[:, None, :] - hs_coords[None, :, :]) ** 2).sum(axis=2))
                nearest_idx = np.argmin(dists, axis=1)
                df["zone_landmark"] = [hs_names[i] for i in nearest_idx]

        if "city" not in df.columns:
            df["city"] = target_city

        df = df.reset_index(drop=True)
        logger.info(f"Parsed and validated {len(df)} GPS trip records for city='{target_city}'.")
        return df

    def generate_synthetic_ola_data(self, n_hours: int = 8760, city: str = "chennai", output_filename: str = "ola_bike_requests.csv") -> pd.DataFrame:
        """
        Generates realistic synthetic hourly Ola bike ride request data.
        Tailored to city climate profiles (e.g. Chennai monsoon rain & heat index).
        """
        logger.info(f"Generating {n_hours} hours of synthetic Ola ride request data for city='{city}'...")

        np.random.seed(42)
        start_date = pd.Timestamp("2024-01-01 00:00:00")
        dates = pd.date_range(start=start_date, periods=n_hours, freq="1h")

        hour = dates.hour
        dayofweek = dates.dayofweek
        month = dates.month

        season = np.where(month.isin([3, 4, 5]), 1,
                 np.where(month.isin([6, 7, 8]), 2,
                 np.where(month.isin([9, 10, 11]), 3, 4)))

        if city.lower() == "chennai":
            # Chennai climate: Severe summer heat in Apr-May, Northeast monsoon heavy rain in Oct-Nov
            temp_base = 29.0 + 5.0 * np.sin(2 * np.pi * (month - 3) / 12)
            temp_diurnal = 4.0 * np.sin(2 * np.pi * (hour - 8) / 24)
            temp = np.clip(temp_base + temp_diurnal + np.random.normal(0, 1.2, size=n_hours), 18.0, 43.0)
            atemp = temp + np.random.normal(1.5, 1.0, size=n_hours) # High humidity feels-like
            humidity = np.clip(72.0 - 10.0 * np.sin(2 * np.pi * (hour - 8) / 24) + np.random.normal(0, 5, n_hours), 30.0, 98.0)
            
            # Monsoon rain probability spike in Oct-Nov (Northeast monsoon)
            rain_prob = np.where(month.isin([10, 11]), 0.28, np.where(humidity > 80, 0.12, 0.04))
        else:
            # Moderate climate profile
            temp_base = 22.0 + 8.0 * np.sin(2 * np.pi * (month - 3) / 12)
            temp_diurnal = 5.0 * np.sin(2 * np.pi * (hour - 8) / 24)
            temp = np.clip(temp_base + temp_diurnal + np.random.normal(0, 1.5, size=n_hours), 5.0, 38.0)
            atemp = temp + np.random.normal(0.5, 1.0, size=n_hours)
            humidity = np.clip(60.0 - 15.0 * np.sin(2 * np.pi * (hour - 8) / 24) + np.random.normal(0, 5, n_hours), 15.0, 95.0)
            rain_prob = np.where(humidity > 80, 0.35, 0.08)

        windspeed = np.clip(10.0 + np.random.exponential(scale=5.0, size=n_hours), 0.0, 50.0)

        is_rain = np.random.binomial(1, rain_prob)
        weather_situation = np.where(is_rain == 0,
                             np.random.choice([1, 2], size=n_hours, p=[0.7, 0.3]),
                             np.random.choice([3, 4], size=n_hours, p=[0.8, 0.2]))

        # Hourly commute curves (8:30 AM & 5:30 PM)
        morning_peak = 190.0 * np.exp(-((hour - 8.5) ** 2) / 2.5)
        evening_peak = 230.0 * np.exp(-((hour - 17.5) ** 2) / 3.0)
        offpeak_base = 30.0 + 15.0 * np.sin(2 * np.pi * hour / 24)

        is_weekend = (dayofweek >= 5).astype(int)
        commute_mult = np.where(is_weekend == 1, 0.45, 1.0)
        weekend_leisure = np.where(is_weekend == 1, 85.0 * np.exp(-((hour - 14) ** 2) / 10.0), 0.0)

        weather_mult = np.where(weather_situation == 1, 1.1,
                        np.where(weather_situation == 2, 1.0,
                        np.where(weather_situation == 3, 0.60, 0.20)))

        expected_cnt = (offpeak_base + (morning_peak + evening_peak) * commute_mult + weekend_leisure) * weather_mult
        cnt = np.random.poisson(lam=np.clip(expected_cnt, 2.0, None))

        casual_ratio = np.clip(np.where(is_weekend == 1, 0.35, 0.15) + np.random.normal(0, 0.05, n_hours), 0.05, 0.6)
        casual = np.round(cnt * casual_ratio).astype(int)
        registered = np.maximum(0, cnt - casual)

        df = pd.DataFrame({
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
            "cnt": cnt
        })

        output_path = self.raw_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic dataset saved to {output_path} ({df.shape[0]} rows)")
        return df

    def generate_synthetic_gps_data(self, n_samples: int = 50000, city: str = "chennai", output_filename: str = "gps_pickups.csv") -> pd.DataFrame:
        """
        Generates realistic synthetic raw GPS trip logs centered on target city hotspots (e.g., Chennai or NYC).
        """
        logger.info(f"Generating {n_samples} synthetic GPS trip records for city='{city}'...")
        np.random.seed(42)

        start_date = pd.Timestamp("2024-04-01 00:00:00")
        random_seconds = np.random.randint(0, 30 * 24 * 3600, size=n_samples)
        timestamps = start_date + pd.to_timedelta(random_seconds, unit="s")

        hotspot_dict = self.CHENNAI_HOTSPOTS if city.lower() == "chennai" else self.NYC_HOTSPOTS
        cluster_centers = list(hotspot_dict.values())
        cluster_names = list(hotspot_dict.keys())
        
        n_clusters = len(cluster_centers)
        probs = [1.0 / n_clusters] * n_clusters
        chosen_indices = np.random.choice(n_clusters, size=n_samples, p=probs)

        lats, lons, zone_labels = [], [], []
        for idx in chosen_indices:
            c_lat, c_lon = cluster_centers[idx]
            lats.append(np.random.normal(c_lat, 0.012))
            lons.append(np.random.normal(c_lon, 0.015))
            zone_labels.append(cluster_names[idx])

        df = pd.DataFrame({
            "Date/Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "city": city.lower(),
            "Lat": np.round(lats, 6),
            "Lon": np.round(lons, 6),
            "zone_landmark": zone_labels,
            "Base": np.random.choice(["OLA_BIKE_01", "OLA_BIKE_02", "OLA_AUTO_01", "RAPIDO_01"], size=n_samples)
        })

        output_path = self.raw_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic GPS dataset saved to {output_path} ({df.shape[0]} rows)")
        return df

    def preprocess_ola_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses and cleans the raw Ola ride request dataset.
        """
        logger.info("Executing preprocessing and temporal resampling pipeline...")
        df = df.copy()

        df.columns = [col.strip().lower() for col in df.columns]

        if "datetime" not in df.columns and "date/time" in df.columns:
            df.rename(columns={"date/time": "datetime"}, inplace=True)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

        full_time_index = pd.date_range(start=df["datetime"].min(), end=df["datetime"].max(), freq="1h")
        df = df.set_index("datetime").reindex(full_time_index)
        df.index.name = "datetime"
        df = df.reset_index()

        num_cols = ["temp", "atemp", "humidity", "windspeed"]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").bfill().ffill()

        cat_cols = ["season", "weather_situation"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().astype(int)

        count_cols = ["casual", "registered", "cnt"]
        for col in count_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).clip(lower=0).astype(int)

        if "cnt" not in df.columns and "casual" in df.columns and "registered" in df.columns:
            df["cnt"] = df["casual"] + df["registered"]

        if "city" in df.columns:
            df["city"] = df["city"].ffill().bfill().fillna(self.default_city)

        logger.info(f"Preprocessing completed. Final continuous hourly dataset shape: {df.shape}")
        return df

    def load_ola_data(self, file_path: Optional[str] = None, city: str = "chennai", force_synthetic: bool = False) -> pd.DataFrame:
        """
        Loads Ola ride request dataset for specified city.
        """
        raw_path = Path(file_path) if file_path else self.raw_dir / "ola_bike_requests.csv"

        if force_synthetic or not raw_path.exists():
            raw_df = self.generate_synthetic_ola_data(city=city, output_filename=raw_path.name)
        else:
            raw_df = pd.read_csv(raw_path)

        clean_df = self.preprocess_ola_data(raw_df)

        processed_path = self.processed_dir / "ola_bike_requests_clean.csv"
        clean_df.to_csv(processed_path, index=False)
        logger.info(f"Cleaned dataset saved to {processed_path}")

        return clean_df

    def load_uber_gps_data(self, file_path: Optional[str] = None, city: str = "chennai", force_synthetic: bool = False) -> pd.DataFrame:
        """
        Loads raw GPS pickup logs centered on target city hotspots (default Chennai).
        Parses and standardizes coordinates and timestamps via parse_raw_gps_coordinates.
        """
        raw_path = Path(file_path) if file_path else self.raw_dir / "uber_nyc_pickups.csv"

        if force_synthetic or not raw_path.exists():
            raw_df = self.generate_synthetic_gps_data(city=city, output_filename=raw_path.name)
        else:
            raw_df = pd.read_csv(raw_path)

        parsed_df = self.parse_raw_gps_coordinates(raw_df, city=city)

        processed_path = self.processed_dir / "uber_gps_pickups_clean.csv"
        parsed_df.to_csv(processed_path, index=False)
        logger.info(f"Cleaned GPS dataset saved to {processed_path} ({parsed_df.shape[0]} records)")

        return parsed_df

    def generate_synthetic_weather_holiday_data(self, n_hours: int = 8760, output_filename: str = "weather_holiday_features.csv") -> pd.DataFrame:
        """
        Generates realistic auxiliary OpenWeatherMap & Holiday exogenous feature logs.
        """
        logger.info(f"Generating {n_hours} hours of OpenWeather & Holiday exogenous features...")
        np.random.seed(42)
        start_date = pd.Timestamp("2024-01-01 00:00:00")
        dates = pd.date_range(start=start_date, periods=n_hours, freq="1h")

        dayofweek = dates.dayofweek
        is_weekend = (dayofweek >= 5).astype(int)

        # Public holidays (approx ~10 holiday dates per year)
        holiday_dates = pd.to_datetime([
            "2024-01-01", "2024-01-26", "2024-03-25", "2024-04-11", "2024-05-01",
            "2024-08-15", "2024-10-02", "2024-10-12", "2024-11-01", "2024-12-25"
        ]).date
        is_holiday = np.isin(dates.date, holiday_dates).astype(int)

        # Rain precipitation depth (mm)
        rain_prob = 0.08
        rain_mask = np.random.binomial(1, rain_prob, size=n_hours)
        rain_1h = np.where(rain_mask == 1, np.random.exponential(scale=3.5, size=n_hours), 0.0)

        # Visibility (meters) & Barometric pressure (hPa)
        visibility = np.where(rain_1h > 5.0, np.random.uniform(500, 3000, size=n_hours), np.random.uniform(8000, 10000, size=n_hours))
        pressure = 1013.25 + np.random.normal(0, 5.0, size=n_hours)

        df = pd.DataFrame({
            "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "rain_1h": np.round(rain_1h, 2),
            "visibility": np.round(visibility, 1),
            "pressure": np.round(pressure, 1),
            "is_holiday": is_holiday,
            "is_weekend": is_weekend
        })

        output_path = self.raw_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"Weather/Holiday features saved to {output_path}")
        return df

    def generate_synthetic_nyc_tlc_data(self, n_samples: int = 20000, output_filename: str = "nyc_tlc_trips.csv") -> pd.DataFrame:
        """
        Generates synthetic NYC TLC Ride-Hailing benchmark dataset (263 Taxi Zones).
        """
        logger.info(f"Generating {n_samples} records for NYC TLC cross-city benchmark...")
        np.random.seed(42)
        start_date = pd.Timestamp("2024-04-01 00:00:00")
        random_seconds = np.random.randint(0, 30 * 24 * 3600, size=n_samples)
        timestamps = start_date + pd.to_timedelta(random_seconds, unit="s")

        pu_locations = np.random.randint(1, 264, size=n_samples)
        do_locations = np.random.randint(1, 264, size=n_samples)
        trip_distances = np.round(np.random.exponential(scale=3.5, size=n_samples) + 0.5, 2)
        fare_amounts = np.round(2.5 + trip_distances * 3.0 + np.random.normal(0, 2.0, n_samples), 2)
        fare_amounts = np.clip(fare_amounts, 3.0, 150.0)

        df = pd.DataFrame({
            "tpep_pickup_datetime": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "PULocationID": pu_locations,
            "DOLocationID": do_locations,
            "trip_distance": trip_distances,
            "passenger_count": np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.03]),
            "fare_amount": fare_amounts,
            "congestion_surcharge": np.random.choice([0.0, 2.75], size=n_samples, p=[0.3, 0.7])
        })

        output_path = self.raw_dir / output_filename
        df.to_csv(output_path, index=False)
        logger.info(f"NYC TLC benchmark dataset saved to {output_path}")
        return df

    def create_zonal_demand_matrix(
        self,
        tlc_df: pd.DataFrame,
        time_col: str = "tpep_pickup_datetime",
        zone_col: str = "PULocationID",
        fill_zero: bool = True,
        as_pivot: bool = False,
        total_zones: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Aggregates raw TLC ride-hailing trips into a continuous hourly zonal demand matrix.
        
        Parameters:
            tlc_df: Raw trip DataFrame containing timestamps and zone IDs.
            time_col: Name of pickup datetime column.
            zone_col: Name of pickup location zone ID column.
            fill_zero: If True, fills missing hour-zone combinations with 0 demand.
            as_pivot: If True, returns a wide format matrix (index=pickup_hour, columns=zones).
            total_zones: Expected number of discrete zones (e.g. 263 for NYC).
            
        Returns:
            Hourly zonal demand DataFrame (long or wide format).
        """
        df = tlc_df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df["pickup_hour"] = df[time_col].dt.floor("1h")

        # Aggregate counts per (pickup_hour, zone)
        zonal_agg = df.groupby(["pickup_hour", zone_col]).size().reset_index(name="trip_count")
        zonal_agg.rename(columns={zone_col: "PULocationID"}, inplace=True)

        if fill_zero:
            all_hours = pd.date_range(start=df["pickup_hour"].min(), end=df["pickup_hour"].max(), freq="1h")
            if total_zones:
                all_zones = np.arange(1, total_zones + 1)
            else:
                all_zones = np.sort(df[zone_col].unique())

            full_grid = pd.MultiIndex.from_product([all_hours, all_zones], names=["pickup_hour", "PULocationID"]).to_frame().reset_index(drop=True)
            zonal_agg = pd.merge(full_grid, zonal_agg, on=["pickup_hour", "PULocationID"], how="left")
            zonal_agg["trip_count"] = zonal_agg["trip_count"].fillna(0).astype(int)

        if as_pivot:
            pivot_df = zonal_agg.pivot(index="pickup_hour", columns="PULocationID", values="trip_count").fillna(0).astype(int)
            return pivot_df

        return zonal_agg

    def load_weather_holiday_data(self, file_path: Optional[str] = None, force_synthetic: bool = False) -> pd.DataFrame:
        """
        Loads OpenWeatherMap and Holiday exogenous features.
        """
        raw_path = Path(file_path) if file_path else self.raw_dir / "weather_holiday_features.csv"
        if force_synthetic or not raw_path.exists():
            raw_df = self.generate_synthetic_weather_holiday_data(output_filename=raw_path.name)
        else:
            raw_df = pd.read_csv(raw_path)

        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
        processed_path = self.processed_dir / "weather_holiday_clean.csv"
        raw_df.to_csv(processed_path, index=False)
        logger.info(f"Cleaned Weather/Holiday features saved to {processed_path}")
        return raw_df

    def load_nyc_tlc_data(self, file_path: Optional[str] = None, force_synthetic: bool = False, as_pivot: bool = False) -> pd.DataFrame:
        """
        Loads NYC TLC ride-hailing benchmark dataset and aggregates into hourly zonal demand.
        """
        raw_path = Path(file_path) if file_path else self.raw_dir / "nyc_tlc_trips.csv"
        if force_synthetic or not raw_path.exists():
            raw_df = self.generate_synthetic_nyc_tlc_data(output_filename=raw_path.name)
        else:
            raw_df = pd.read_csv(raw_path)

        zonal_demand = self.create_zonal_demand_matrix(raw_df, fill_zero=True, as_pivot=as_pivot)

        processed_path = self.processed_dir / "nyc_tlc_benchmark_clean.csv"
        zonal_demand.to_csv(processed_path, index=as_pivot)
        logger.info(f"Cleaned NYC TLC benchmark zonal demand saved to {processed_path} (shape: {zonal_demand.shape})")
        return zonal_demand

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates summary statistics and data health metrics for the dataset.
        """
        summary = {
            "total_records": int(len(df)),
            "start_time": str(df["datetime"].min()) if "datetime" in df.columns else (str(df["pickup_hour"].min()) if "pickup_hour" in df.columns else None),
            "end_time": str(df["datetime"].max()) if "datetime" in df.columns else (str(df["pickup_hour"].max()) if "pickup_hour" in df.columns else None),
            "null_count": int(df.isnull().sum().sum()),
            "columns": list(df.columns)
        }
        if "cnt" in df.columns:
            summary["target_stats"] = {
                "mean": float(df["cnt"].mean()),
                "std": float(df["cnt"].std()),
                "min": int(df["cnt"].min()),
                "max": int(df["cnt"].max()),
                "zero_count": int((df["cnt"] == 0).sum())
            }
        elif "trip_count" in df.columns:
            summary["target_stats"] = {
                "mean": float(df["trip_count"].mean()),
                "std": float(df["trip_count"].std()),
                "min": int(df["trip_count"].min()),
                "max": int(df["trip_count"].max()),
                "zero_count": int((df["trip_count"] == 0).sum())
            }
        return summary


if __name__ == "__main__":
    loader = OlaDataLoader(default_city="chennai")
    df_ola = loader.load_ola_data(city="chennai", force_synthetic=True)
    df_gps = loader.load_uber_gps_data(city="chennai", force_synthetic=True)
    df_weather = loader.load_weather_holiday_data(force_synthetic=True)
    df_tlc = loader.load_nyc_tlc_data(force_synthetic=True)
    print("\n--- Ola Bike Request Summary (Chennai) ---")
    print(loader.get_data_summary(df_ola))
    print("\n--- GPS Trip Summary (Chennai Hotspots) ---")
    print(df_gps[['Date/Time', 'Lat', 'Lon', 'zone_landmark']].head())
    print("\n--- Weather & Holiday Summary ---")
    print(loader.get_data_summary(df_weather))
    print("\n--- NYC TLC Zonal Demand Summary ---")
    print(loader.get_data_summary(df_tlc))


