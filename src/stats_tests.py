"""Statistical Significance & Hypothesis Testing Module.

Implements the Diebold-Mariano (DM) Test and Wilcoxon Signed-Rank Test
to mathematically prove statistically significant superiority (p < 0.01)
of the proposed models over baselines.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    power: int = 1
) -> Tuple[float, float]:
    """Diebold-Mariano Test for equal predictive accuracy.

    H0: The two forecasting models have equal predictive accuracy.
    H1: Model 1 and Model 2 have significantly different accuracy.

    Args:
        e1: Forecast errors of Model 1 (y_true - y_pred1)
        e2: Forecast errors of Model 2 (y_true - y_pred2)
        h: Forecast horizon (default: 1)
        power: Error loss power (1 for MAE loss, 2 for MSE loss)

    Returns:
        Tuple of (dm_statistic, p_value)
    """
    e1 = np.asarray(e1, dtype=np.float64).flatten()
    e2 = np.asarray(e2, dtype=np.float64).flatten()
    n = len(e1)

    if n != len(e2):
        raise ValueError("Error arrays e1 and e2 must be of equal length.")

    # Define loss differential d_t
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    elif power == 2:
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) ** power - np.abs(e2) ** power

    # Mean loss differential
    d_bar = np.mean(d)

    # Compute autocovariance at lag k
    def autocovariance(k):
        if k == 0:
            return np.var(d, ddof=0)
        return np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))

    # Variance of mean loss differential with Harvey et al. correction
    gamma_0 = autocovariance(0)
    gamma_k = sum([autocovariance(k) for k in range(1, h)])
    var_d_bar = (gamma_0 + 2 * gamma_k) / n

    if var_d_bar <= 0:
        dm_stat = 0.0
        p_val = 1.0
    else:
        dm_stat = d_bar / np.sqrt(var_d_bar)
        # Apply Harvey, Leybourne, and Newbold (1997) small-sample modification
        hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = dm_stat * hln_factor
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    logger.info("Diebold-Mariano Test: DM-stat=%.4f, p-value=%.6f (n=%d)", dm_stat, p_val, n)
    return float(dm_stat), float(p_val)


def wilcoxon_test(e1: np.ndarray, e2: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon Signed-Rank Test for non-parametric residual comparison.

    Args:
        e1: Absolute errors of Model 1
        e2: Absolute errors of Model 2

    Returns:
        Tuple of (stat, p_value)
    """
    abs_e1 = np.abs(np.asarray(e1, dtype=np.float64).flatten())
    abs_e2 = np.abs(np.asarray(e2, dtype=np.float64).flatten())

    res = stats.wilcoxon(abs_e1, abs_e2, alternative="two-sided")
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    logger.info("Wilcoxon Test: stat=%.4f, p-value=%.6f", stat, p_val)
    return stat, p_val


def compute_statistical_significance_matrix(
    models_dict: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Compute pairwise DM test p-values across a dictionary of model predictions.

    Args:
        models_dict: Dictionary mapping model_name -> y_pred array
        y_true: True targets array

    Returns:
        Nested dictionary containing DM-stat, p-value, and statistical significance flag (p < 0.01)
    """
    model_names = list(models_dict.keys())
    results = {}

    for name1 in model_names:
        results[name1] = {}
        e1 = y_true - models_dict[name1]
        for name2 in model_names:
            if name1 == name2:
                results[name1][name2] = {"dm_stat": 0.0, "p_val": 1.0, "significant_p01": False}
            else:
                e2 = y_true - models_dict[name2]
                dm_stat, p_val = diebold_mariano_test(e1, e2)
                results[name1][name2] = {
                    "dm_stat": dm_stat,
                    "p_val": p_val,
                    "significant_p01": p_val < 0.01
                }

    return results
