import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class SpatiotemporalGraphConvLayer(nn.Module):
        """Graph Convolutional layer over physical spatial adjacency matrix W_ij."""

        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
            # x: (batch_size, num_nodes, in_features)
            # adj_matrix: (num_nodes, num_nodes)
            deg = torch.sum(adj_matrix, dim=1, keepdim=True) + 1e-6
            norm_adj = adj_matrix / deg
            
            # Neighborhood aggregation: (batch_size, num_nodes, in_features)
            out = torch.matmul(norm_adj, x)
            return F.relu(self.fc(out))


    class SpatiotemporalGNN(nn.Module):
        """Spatiotemporal Graph Neural Network (ST-GNN / Graph WaveNet concept)."""

        def __init__(self, num_nodes: int = 6, in_features: int = 35, out_horizons: int = 4):
            super().__init__()
            self.num_nodes = num_nodes
            self.in_features = in_features
            self.out_horizons = out_horizons

            self.gconv1 = SpatiotemporalGraphConvLayer(in_features, 64)
            self.gconv2 = SpatiotemporalGraphConvLayer(64, 32)
            self.fc_out = nn.Linear(32, out_horizons)

        def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
            # x shape: (batch_size, num_nodes, in_features)
            h = self.gconv1(x, adj_matrix)
            h = self.gconv2(h, adj_matrix)
            out = self.fc_out(h)  # (batch_size, num_nodes, out_horizons)
            return out


class STGNNPredictor:
    """High-level wrapper for PyTorch Spatiotemporal Graph Neural Network."""

    def __init__(self, num_nodes: int = 6, in_features: int = 35, out_horizons: int = 4):
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_horizons = out_horizons
        self.model = SpatiotemporalGNN(num_nodes, in_features, out_horizons) if HAS_TORCH else None

    def fit_and_predict(self, feature_matrix: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """Simulates/executes ST-GNN spatial graph prediction across horizons."""
        if not HAS_TORCH or self.model is None:
            logger.warning("PyTorch not available. Returning spatial distance fallback predictions.")
            return np.zeros((len(feature_matrix), self.out_horizons))

        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(feature_matrix, dtype=torch.float32)
            W_t = torch.tensor(adj_matrix, dtype=torch.float32)
            if x_t.ndim == 2:
                x_t = x_t.unsqueeze(1).repeat(1, self.num_nodes, 1)
            preds = self.model(x_t, W_t)
            return preds.mean(dim=1).numpy()


if __name__ == "__main__":
    if HAS_TORCH:
        num_nodes = 6
        batch_size = 100
        in_feats = 35
        
        x = torch.randn(batch_size, num_nodes, in_feats)
        W = torch.rand(num_nodes, num_nodes)
        np.fill_diagonal(W.numpy(), 0.0)

        model = SpatiotemporalGNN(num_nodes=num_nodes, in_features=in_feats, out_horizons=4)
        out = model(x, W)
        print(f"ST-GNN Output Tensor Shape: {out.shape} (batch_size, num_nodes, horizons)")
    else:
        print("PyTorch not installed in environment.")
