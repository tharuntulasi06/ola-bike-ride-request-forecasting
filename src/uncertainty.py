"""Uncertainty Quantification & Conformal Prediction Module.

Implements Quantile Loss (Pinball Loss) and Inductive Conformal Prediction (ICP)
to construct 95% calibrated confidence intervals [y_low, y_high] for demand risk management.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
    """Compute Pinball (Quantile) Loss.

    Args:
        y_true: Ground truth target tensor
        y_pred: Predicted quantile target tensor
        quantile: Quantile target in (0, 1), e.g., 0.05, 0.5, 0.95

    Returns:
        Scalar pinball loss tensor
    """
    errors = y_true - y_pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.mean(loss)


class QuantileLoss(nn.Module):
    """Multi-Quantile Loss layer for simultaneous lower, median, and upper quantile prediction."""

    def __init__(self, quantiles: Tuple[float, ...] = (0.05, 0.50, 0.95)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_true: torch.Tensor, y_preds: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            y_true: Ground truth tensor of shape [B, N, H]
            y_preds: Predicted quantiles tensor of shape [B, N, H, Q]

        Returns:
            Combined multi-quantile loss scalar
        """
        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            total_loss += pinball_loss(y_true, y_preds[..., i], quantile=q)
        return total_loss / len(self.quantiles)


class ConformalPredictor:
    """Inductive Conformal Prediction (ICP) for 95% prediction interval guarantees."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.is_calibrated = False

    def calibrate(self, y_calib_true: np.ndarray, y_calib_pred: np.ndarray) -> float:
        """Calibrate conformal non-conformity scores on validation set.

        Args:
            y_calib_true: Ground truth array of shape [N_samples]
            y_calib_pred: Model predictions array of shape [N_samples]

        Returns:
            Empirical quantile value q_hat
        """
        n = len(y_calib_true)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Compute absolute non-conformity scores
        scores = np.abs(y_calib_true - y_calib_pred)

        # Compute empirical quantile at level (1 - alpha) * (1 + 1/n)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(1.0, max(0.0, quantile_level))

        self.q_hat = float(np.quantile(scores, quantile_level))
        self.is_calibrated = True
        logger.info("Conformal predictor calibrated: alpha=%.2f, q_hat=%.4f (n=%d)", self.alpha, self.q_hat, n)
        return self.q_hat

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate upper and lower bounds for new predictions.

        Args:
            y_pred: Point prediction array of shape [N_samples]

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        if not self.is_calibrated or self.q_hat is None:
            raise RuntimeError("ConformalPredictor must be calibrated before predicting intervals.")

        lower_bound = np.maximum(0.0, y_pred - self.q_hat)
        upper_bound = y_pred + self.q_hat
        return lower_bound, upper_bound

    def evaluate_coverage(
        self,
        y_test_true: np.ndarray,
        y_test_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute Prediction Interval Coverage Probability (PICP) and Mean Width (MPIW).

        Args:
            y_test_true: Ground truth test targets
            y_test_pred: Point test predictions

        Returns:
            Dictionary containing PICP coverage, target coverage, and MPIW width
        """
        lower, upper = self.predict_interval(y_test_pred)
        covered = (y_test_true >= lower) & (y_test_true <= upper)
        picp = float(np.mean(covered))
        mpiw = float(np.mean(upper - lower))

        return {
            "picp_coverage": picp,
            "target_coverage": 1.0 - self.alpha,
            "mpiw_width": mpiw,
            "q_hat": self.q_hat if self.q_hat is not None else 0.0
        }
