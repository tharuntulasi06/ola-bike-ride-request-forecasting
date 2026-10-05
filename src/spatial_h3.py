"""Uber H3 Hexagonal Hierarchical Spatial Indexing Module.

Provides spatial partitioning via Uber H3 hexagonal grids (Res 7, 8, 9)
as an advanced benchmark alternative to K-Means centroids.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3-py package not installed. Spatial H3 module running in fallback mode.")


class H3SpatialIndexer:
    """Uber H3 Spatial Hexagonal Hierarchy Manager."""

    def __init__(self, resolution: int = 8):
        self.resolution = resolution

    def latlng_to_h3(self, lat: float, lng: float) -> str:
        """Convert Latitude and Longitude to H3 index.

        Args:
            lat: Latitude coordinate
            lng: Longitude coordinate

        Returns:
            H3 string index code
        """
        if H3_AVAILABLE:
            try:
                return h3.latlng_to_cell(lat, lng, self.resolution)
            except AttributeError:
                return h3.geo_to_h3(lat, lng, self.resolution)
        else:
            # Fallback grid hashing if h3-py is not installed
            grid_lat = round(lat, 2)
            grid_lng = round(lng, 2)
            return f"h3_res{self.resolution}_{grid_lat}_{grid_lng}"

    def assign_h3_clusters(self, df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
        """Assign H3 spatial cell identifiers to dataframe rows.

        Args:
            df: DataFrame containing latitude and longitude columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column

        Returns:
            DataFrame with added 'h3_cell' and 'cluster_id' integer mapping
        """
        df_out = df.copy()

        if lat_col not in df_out.columns or lon_col not in df_out.columns:
            logger.warning("Latitude/Longitude columns not found. Returning original dataframe.")
            df_out["h3_cell"] = "h3_cell_0"
            df_out["cluster_id"] = 0
            return df_out

        df_out["h3_cell"] = [
            self.latlng_to_h3(row[lat_col], row[lon_col])
            for _, row in df_out.iterrows()
        ]

        # Map string H3 cells to discrete cluster IDs
        unique_cells = df_out["h3_cell"].unique()
        cell_map = {cell: i for i, cell in enumerate(unique_cells)}
        df_out["cluster_id"] = df_out["h3_cell"].map(cell_map)

        logger.info("Assigned %d unique H3 cells at resolution %d", len(unique_cells), self.resolution)
        return df_out

    def compute_h3_distance_matrix(self, df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> np.ndarray:
        """Compute spatial Euclidean distance matrix between H3 cell centroids.

        Args:
            df: DataFrame with cluster_id, lat, lon

        Returns:
            Distance matrix of shape [N_clusters, N_clusters]
        """
        if "cluster_id" not in df.columns:
            df = self.assign_h3_clusters(df, lat_col, lon_col)

        centroids = df.groupby("cluster_id")[[lat_col, lon_col]].mean().reset_index()
        n_clusters = len(centroids)
        dist_matrix = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                lat1, lon1 = centroids.loc[i, [lat_col, lon_col]]
                lat2, lon2 = centroids.loc[j, [lat_col, lon_col]]
                # Approx distance in km
                dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0
                dist_matrix[i, j] = dist

        return dist_matrix
