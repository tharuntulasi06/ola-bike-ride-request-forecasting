"""Unit tests verifying all Research-Level Transformation Modules.

Tests WG-STGAT model, Conformal Predictor, H3 Spatial Indexer, Diebold-Mariano tests,
and Ablation runners.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from src.wg_stgat_model import WG_STGAT_Model
from src.uncertainty import ConformalPredictor, pinball_loss
from src.spatial_h3 import H3SpatialIndexer
from src.stats_tests import diebold_mariano_test, wilcoxon_test
from src.benchmarks import MultiModelBenchmarkSuite


def test_wg_stgat_forward_pass():
    """Test WG-STGAT model forward pass dimensions."""
    B, N, T, F_in = 4, 6, 24, 1
    horizon = 4
    model = WG_STGAT_Model(num_nodes=N, in_dim=F_in, hidden_dim=32, out_horizon=horizon)

    x = torch.randn(B, N, T, F_in)
    adj = torch.eye(N)
    weather = torch.randn(B, 4)

    out = model(x, adj, weather)
    assert out.shape == (B, N, horizon)


def test_conformal_predictor():
    """Test Conformal Predictor calibration and interval bounds."""
    predictor = ConformalPredictor(alpha=0.05)
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 38.0, 52.0])

    q_hat = predictor.calibrate(y_true, y_pred)
    assert q_hat > 0

    lower, upper = predictor.predict_interval(y_pred)
    assert len(lower) == len(y_pred)
    assert len(upper) == len(y_pred)
    assert np.all(lower <= y_pred)
    assert np.all(upper >= y_pred)


def test_h3_spatial_indexer():
    """Test Uber H3 spatial index assignment."""
    indexer = H3SpatialIndexer(resolution=8)
    df = pd.DataFrame({
        "lat": [13.0827, 13.0830, 13.0900],
        "lon": [80.2707, 80.2710, 80.2800]
    })
    df_out = indexer.assign_h3_clusters(df)
    assert "h3_cell" in df_out.columns
    assert "cluster_id" in df_out.columns
    assert len(df_out) == 3


def test_diebold_mariano_test():
    """Test Diebold-Mariano statistical significance test."""
    np.random.seed(42)
    e1 = np.random.normal(0, 1.0, 100)
    e2 = np.random.normal(0, 2.5, 100)

    dm_stat, p_val = diebold_mariano_test(e1, e2)
    assert isinstance(dm_stat, float)
    assert 0.0 <= p_val <= 1.0


def test_multimodel_benchmark_suite():
    """Test MultiModelBenchmarkSuite execution."""
    suite = MultiModelBenchmarkSuite()
    df = pd.DataFrame({
        "cluster_id": [0] * 50 + [1] * 50,
        "temp": np.random.uniform(25, 35, 100),
        "humidity": np.random.uniform(50, 90, 100),
        "cnt": np.random.randint(10, 100, 100)
    })
    results = suite.run_all_benchmarks(df, horizon=2)
    assert "Naive_Historical_Average" in results
    assert "Random_Forest" in results
    assert "Proposed_WG_STGAT" in results
