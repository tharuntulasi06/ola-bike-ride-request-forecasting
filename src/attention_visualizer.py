"""Spatial Graph Attention Weight Visualizer & GeoJSON Exporter.

Extracts dynamic spatial attention weights alpha_ij(t) from the WG-STGAT model
and exports spatiotemporal topology graphs to GeoJSON for the Next.js frontend dashboard.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.wg_stgat_model import WG_STGAT_Model

logger = logging.getLogger(__name__)


class GraphAttentionVisualizer:
    """Extracts and formats WG-STGAT Graph Attention weights into GeoJSON."""

    def __init__(self, model: WG_STGAT_Model):
        self.model = model

    def extract_attention_matrix(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        weather: torch.Tensor
    ) -> np.ndarray:
        """Extract spatial attention matrix alpha_ij from WG-STGAT layer.

        Args:
            x: Input feature tensor [B, N, T, F]
            adj: Spatial adjacency matrix [N, N]
            weather: Exogenous weather vector [B, weather_dim]

        Returns:
            Extracted attention matrix of shape [N, N]
        """
        self.model.eval()
        with torch.no_grad():
            B, N, T, _ = x.size()
            h = self.model.fc_in(x)
            h_last = h[:, :, -1, :]  # [B, N, H_dim]

            # Forward pass through GAT to compute scores
            gat = self.model.gat
            h_src = gat.lin_src(h_last).view(B, N, gat.heads, gat.out_channels)
            h_dst = gat.lin_dst(h_last).view(B, N, gat.heads, gat.out_channels)

            src_expanded = h_src.unsqueeze(2).expand(B, N, N, gat.heads, gat.out_channels)
            dst_expanded = h_dst.unsqueeze(1).expand(B, N, N, gat.heads, gat.out_channels)
            cat_features = torch.cat([src_expanded, dst_expanded], dim=-1)

            scores = torch.sum(cat_features * gat.att.unsqueeze(0).unsqueeze(0), dim=-1)
            scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.2)

            adj_mask = adj.unsqueeze(0).unsqueeze(-1)
            scores = scores.masked_fill(adj_mask == 0, -1e9)
            alpha = torch.nn.functional.softmax(scores, dim=2)  # [B, N, N, H]

            # Average over batch and multi-attention heads
            alpha_mean = alpha.mean(dim=(0, 3)).cpu().numpy()

        return alpha_mean

    def export_geojson_topology(
        self,
        centroids: np.ndarray,
        alpha_matrix: np.ndarray,
        output_file: str = "results/attention_topology.json"
    ) -> Dict[str, Any]:
        """Export spatial cluster nodes and graph attention edges to GeoJSON format.

        Args:
            centroids: Cluster centroid coordinates array of shape [N, 2] (lat, lon)
            alpha_matrix: Attention weights matrix of shape [N, N]
            output_file: Target JSON file path

        Returns:
            GeoJSON dictionary payload
        """
        N = len(centroids)
        features = []

        # 1. Add Cluster Centroid Node Features
        for i in range(N):
            lat, lon = centroids[i]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)]
                },
                "properties": {
                    "node_id": i,
                    "type": "cluster_centroid",
                    "name": f"Cluster Hotspot {i+1}"
                }
            })

        # 2. Add Directed Edge Spillover Features
        for i in range(N):
            for j in range(N):
                if i != j and alpha_matrix[i, j] > 0.05:
                    lat1, lon1 = centroids[i]
                    lat2, lon2 = centroids[j]
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [float(lon1), float(lat1)],
                                [float(lon2), float(lat2)]
                            ]
                        },
                        "properties": {
                            "source": i,
                            "target": j,
                            "weight": float(alpha_matrix[i, j]),
                            "type": "spillover_attention_edge"
                        }
                    })

        geojson_payload = {
            "type": "FeatureCollection",
            "metadata": {
                "num_nodes": N,
                "model": "WG-STGAT",
                "version": "1.0.0"
            },
            "features": features
        }

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(geojson_payload, f, indent=2)

        logger.info("Exported GeoJSON graph attention topology to: %s", out_path)
        return geojson_payload
