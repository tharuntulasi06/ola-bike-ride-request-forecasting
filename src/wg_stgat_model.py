"""Weather-Gated Spatiotemporal Graph Attention Network (WG-STGAT).

Research-grade PyTorch / PyTorch Geometric module combining spatial graph attention,
exogenous weather gating, and gated temporal dilated convolutions.
"""

import math
import logging
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeatherGatedSpatialGraphAttention(nn.Module):
    """Spatial Graph Attention layer gated by exogenous weather features.

    Dynamically modulates spatial edge weights alpha_ij based on real-time
    meteorological severity (rain_1h, temp, windspeed).
    """

    def __init__(self, in_channels: int, out_channels: int, weather_dim: int = 4, heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weather_dim = weather_dim
        self.heads = heads

        # Projection matrices
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_weather = nn.Linear(weather_dim, heads, bias=False)

        # Attention vector
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_weather.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        weather: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features tensor of shape [B, N, F_in]
            adj: Adjacency matrix of shape [N, N] or [B, N, N]
            weather: Exogenous weather vector of shape [B, weather_dim] or [B, N, weather_dim]

        Returns:
            Output feature tensor of shape [B, N, heads * out_channels]
        """
        B, N, _ = x.size()

        # Project features: [B, N, H, F_out]
        h_src = self.lin_src(x).view(B, N, self.heads, self.out_channels)
        h_dst = self.lin_dst(x).view(B, N, self.heads, self.out_channels)

        # Weather gating score: [B, H]
        if weather.dim() == 2:
            w_gate = torch.sigmoid(self.lin_weather(weather))  # [B, H]
            w_gate = w_gate.unsqueeze(1).unsqueeze(3)  # [B, 1, H, 1]
        else:
            w_gate = torch.sigmoid(self.lin_weather(weather)).unsqueeze(3)  # [B, N, H, 1]

        # Apply weather gating to node embeddings
        h_src = h_src * (1.0 + w_gate)

        # Compute pairwise attention scores: [B, N, N, H]
        # h_src_i + h_dst_j concatenation approach
        src_expanded = h_src.unsqueeze(2).expand(B, N, N, self.heads, self.out_channels)
        dst_expanded = h_dst.unsqueeze(1).expand(B, N, N, self.heads, self.out_channels)
        cat_features = torch.cat([src_expanded, dst_expanded], dim=-1)  # [B, N, N, H, 2*out_channels]

        # Dot product with attention vector: [B, N, N, H]
        scores = torch.sum(cat_features * self.att.unsqueeze(0).unsqueeze(0), dim=-1)
        scores = F.leaky_relu(scores, negative_slope=0.2)

        # Mask with spatial adjacency: if adj is 0, mask score
        if adj.dim() == 2:
            adj_mask = adj.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        else:
            adj_mask = adj.unsqueeze(-1)  # [B, N, N, 1]

        scores = scores.masked_fill(adj_mask == 0, -1e9)
        alpha = F.softmax(scores, dim=2)  # Normalize across spatial neighbors j

        # Aggregate neighbor features: [B, N, H, F_out]
        out = torch.einsum("bnjh,bnjhf->bnhf", alpha, dst_expanded)
        out = out.reshape(B, N, self.heads * self.out_channels) + self.bias
        return F.elu(out)


class GatedTemporalConv(nn.Module):
    """Gated 1D Dilated Temporal Convolution for temporal pattern extraction."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )
        self.conv_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B * N, C_in, T]
        Returns:
            Output tensor of shape [B * N, C_out, T]
        """
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        out = filter_out * gate_out
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class WG_STGAT_Model(nn.Module):
    """Weather-Gated Spatiotemporal Graph Attention Network (WG-STGAT).

    Full architecture combining weather-gated spatial graph attention and
    gated temporal convolutions for multi-step spatiotemporal demand forecasting.
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 1,
        hidden_dim: int = 64,
        out_horizon: int = 4,
        weather_dim: int = 4,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_horizon = out_horizon
        self.weather_dim = weather_dim

        # Temporal Embedding & Feature Projection
        self.fc_in = nn.Linear(in_dim, hidden_dim)

        # Spatial Graph Attention Layer
        self.gat = WeatherGatedSpatialGraphAttention(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            weather_dim=weather_dim,
            heads=heads
        )

        # Gated Temporal Convolutions
        self.tconv1 = GatedTemporalConv(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tconv2 = GatedTemporalConv(hidden_dim, hidden_dim, kernel_size=3, dilation=2)

        # Output Regression Head
        self.dropout = nn.Dropout(dropout)
        self.fc_out1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out2 = nn.Linear(hidden_dim // 2, out_horizon)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        weather: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input node demand sequence of shape [B, N, T, F_in] or [B, N, T]
            adj: Adjacency matrix of shape [N, N]
            weather: Weather feature vector of shape [B, weather_dim]

        Returns:
            Multi-step forecast predictions of shape [B, N, out_horizon]
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # [B, N, T, 1]

        B, N, T, F_in = x.size()

        # Project input features: [B, N, T, H_dim]
        h = self.fc_in(x)

        # Apply Spatial Graph Attention at the last temporal step
        h_last = h[:, :, -1, :]  # [B, N, H_dim]
        h_spatial = self.gat(h_last, adj, weather)  # [B, N, H_dim]

        # Apply Temporal Convolutions over time steps
        h_temp = h.view(B * N, F_in if F_in == self.hidden_dim else T, -1).transpose(1, 2)
        if h_temp.size(1) != self.hidden_dim:
            h_temp = self.fc_in(x.view(B * N, T, F_in)).transpose(1, 2)

        h_temp = self.tconv1(h_temp)
        h_temp = self.tconv2(h_temp)
        h_temp_last = h_temp[:, :, -1].view(B, N, self.hidden_dim)

        # Fuse Spatial & Temporal representations
        h_fused = F.relu(h_spatial + h_temp_last)
        h_fused = self.dropout(h_fused)

        # Predict multi-step horizons: [B, N, out_horizon]
        out = F.relu(self.fc_out1(h_fused))
        preds = self.fc_out2(out)  # [B, N, out_horizon]
        return preds
