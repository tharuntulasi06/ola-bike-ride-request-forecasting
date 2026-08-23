# 🤖 Model Architecture & Technical Specifications

> **Project**: Ola Bike Ride Request Demand Forecasting  
> **Core Architecture**: Dual-Paradigm Model Architecture (GBDT Trio Ensembles + PyTorch Geometric Spatiotemporal GNN)  
> **Flagship Scope**: Chennai Micro-Mobility Hotspots ($K=6$ Spatial Centroids) & Cross-City Generalizability  

---

## 📌 Executive Architectural Overview

This document provides a comprehensive technical breakdown of the machine learning and deep learning models implemented in the **Ola Bike Ride Request Demand Forecasting** system.

```text
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                 Input: Spatiotemporal Parquet Feature Matrix                │
  │                  (51,528 records × 45 features across 6 zones)              │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌─────────────────────────────┐                 ┌─────────────────────────────┐
  │  Paradigm 1: GBDT Trio      │                 │  Paradigm 2: ST-GNN         │
  │     (src/trainer.py)        │                 │   (src/st_gnn_model.py)     │
  │ • XGBoost (Tweedie Loss)    │                 │ • PyTorch Geometric ST-GNN  │
  │ • LightGBM (Leaf-wise GOSS) │                 │ • Spatial Adjacency (W_ij)  │
  │ • CatBoost (Ordered Boost)  │                 │ • Neighborhood Spillover    │
  │ • Optuna TPE Tuning         │                 └──────────────┬──────────────┘
  └──────────────┬──────────────┘                                │
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Weighted Stacking Meta-Ensemble & Evaluation Suite (src/evaluate.py)       │
  │  • WAPE, MAE, RMSE, R² Metrics + SHAP Feature Importance Plots              │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Direct Multi-Step Forecasting Strategy

Traditional multi-step time-series forecasting relies on **Recursive Forecasting** (predicting $t+1$, feeding the prediction back as an input lag, and repeating for $t+2 \dots t+4$). This causes **exponential error accumulation** by horizon $t+4$.

We employ the **Direct Multi-Step Forecasting Strategy**:
* We train **4 independent specialized estimators** per model family ($f_1, f_2, f_3, f_4$), where estimator $f_h$ directly targets demand at horizon $t+h$.
* **Advantage**: Zero error propagation from $t+1$ to $t+4$. Each model family optimizes specifically for its target forecast horizon.

---

## 🌳 Paradigm 1: Gradient Boosting Trio Ensembles (`src/trainer.py`)

### 1. XGBoost Regressor with Tweedie Loss
* **The Zero-Inflation Challenge**: Micro-mobility demand features zero-inflated count properties (many 0s or small integers during late-night off-peak hours, but massive spikes during morning/evening commute peaks). Standard Gaussian Loss (MSE) fails on zero-heavy counts and often predicts illegal negative demand values.
* **Tweedie Loss Formulation ($1 < p < 2$)**:
  $$\mathcal{L}_{\text{Tweedie}}(y, \mu) = - \frac{y \cdot \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p}$$
  * $p=1.0$: Pure Poisson Distribution (count modeling).
  * $p=2.0$: Pure Gamma Distribution (continuous volume modeling).
  * We tune $p \in [1.1, 1.9]$ via Optuna, unifying zero-count handling with continuous peak-demand estimation.

### 2. LightGBM Regressor
* **GOSS (Gradient-based One-Side Sampling)**: Retains instances with large gradients while randomly sampling instances with small gradients, accelerating training by $10\times$.
* **Leaf-wise Tree Growth**: Splitting by maximum delta loss captures high-order non-linear feature interactions (e.g. Northeast monsoon rain depth $\times$ peak 6 PM office commute hour).

### 3. CatBoost Regressor
* **Ordered Categorical Boosting**: Uses target-statistics permutation to handle spatial cluster IDs (`cluster_id = 0 \dots 5`), operational seasons, and weather situation indices without target leakage.

---

## 🕸️ Paradigm 2: Spatiotemporal Graph Neural Network (`src/st_gnn_model.py`)

Micro-mobility demand in an urban center does not exist in isolation. If heavy monsoon rain strikes **T. Nagar**, commuters shift to adjacent transit hubs like **Guindy** or **Chennai Central**.

### 1. Physical Haversine Spatial Adjacency Matrix ($W_{ij}$)
We compute the $6 \times 6$ Gaussian thresholded adjacency matrix using physical Haversine distance ($\text{km}$) between Chennai landmark centroids:

$$W_{ij} = \begin{cases} 
\exp\left(-\left(\frac{\text{dist}(i,j)}{\sigma}\right)^2\right) & \text{if } \text{dist}(i,j) \le \kappa \\
0 & \text{otherwise}
\end{cases}$$

Where $\sigma = 5.0\text{ km}$ and cutoff threshold $\kappa = 15.0\text{ km}$. Matrix $W_{ij}$ is symmetric ($W_{ij} = W_{ji}$) with a zero diagonal ($W_{ii} = 0$).

### 2. Graph Convolutions in PyTorch Geometric
* **Spatial Aggregation**: Computes normalized graph Laplacian convolutions $D^{-1} W x$ to model physical demand spillover to adjacent spatial nodes.
* **Temporal Convolution**: 1D temporal convolutions extract short-term momentum across lag features.

---

## 🎛️ Optuna Automated Hyperparameter Optimization

Hyperparameters are optimized using **Optuna's Tree-structured Parzen Estimator (TPE)** algorithm:

### Search Space Bounds
* `learning_rate`: $[0.01, 0.15]$ (log scale)
* `max_depth`: $[3, 8]$
* `subsample` / `colsample_bytree`: $[0.6, 1.0]$
* `tweedie_variance_power`: $[1.1, 1.9]$

### Optimization Metric: WAPE (Weighted Absolute Percentage Error)
$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$

> **Why WAPE over MAPE?** Standard MAPE ($\frac{|y - \hat{y}|}{y}$) divides by actual demand $y_i$, which causes division-by-zero ($\infty$) during off-peak zero-demand hours. WAPE divides by the total aggregate volume $\sum y_i$, making it stable, robust, and zero-safe.

---

## 🤝 Weighted Stacking Meta-Ensemble

Validation predictions from out-of-fold models are combined using inverse WAPE weighting:

$$w_m = \frac{1 / (\text{WAPE}_m + \epsilon)}{\sum_{k} 1 / (\text{WAPE}_k + \epsilon)}$$

$$\hat{Y}_{\text{ensemble}} = w_{\text{XGB}} \cdot \hat{Y}_{\text{XGBoost}} + w_{\text{LGB}} \cdot \hat{Y}_{\text{LightGBM}} + w_{\text{CAT}} \cdot \hat{Y}_{\text{CatBoost}}$$

---

## 📊 Empirical Performance & Feature Importance

### 1. Benchmark Results Across Forecast Horizons ($t+1 \dots t+4$)

| Horizon | XGBoost WAPE | LightGBM WAPE | CatBoost WAPE | Ensemble WAPE | MAE | RMSE | $R^2$ Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t+1$** | 0.4186 | 0.4202 | 0.4193 | **0.4184** | 46.98 | 54.99 | 0.1438 |
| **$t+2$** | 0.4175 | 0.4212 | 0.4194 | **0.4183** | 46.59 | 54.59 | 0.1563 |
| **$t+3$** | 0.4182 | 0.4237 | 0.4188 | **0.4189** | 46.97 | 55.05 | 0.1422 |
| **$t+4$** | 0.4184 | 0.4191 | 0.4176 | **0.4174** | 46.93 | 54.98 | 0.1445 |

### 2. Top 10 Global Feature Importance Rankings

| Rank | Feature | Description | Importance Score |
| :---: | :--- | :--- | :---: |
| **1** | `lag_168h` | Same hour demand from previous week (Weekly seasonality) | **0.0402** |
| **2** | `windspeed` | Environmental wind speed | **0.0401** |
| **3** | `temp_roll_mean_3h` | 3-hour trailing temperature mean | **0.0381** |
| **4** | `humidity_roll_mean_24h` | 24-hour trailing humidity mean | **0.0372** |
| **5** | `lag_2h` | 2-hour short-term temporal lag | **0.0368** |
| **6** | `cnt` | Base demand count | **0.0367** |
| **7** | `lag_48h` | 2-day historical lag | **0.0366** |
| **8** | `lag_24h` | 24-hour daily seasonality lag | **0.0364** |
| **9** | `humidity_roll_mean_3h` | 3-hour trailing humidity mean | **0.0361** |
| **10** | `temp_roll_std_24h` | 24-hour temperature variance | **0.0353** |

---

## 📂 Serialized Model Checkpoints & Production Serving

| Artifact File | Size | Architecture | Serialization Method |
| :--- | :--- | :--- | :--- |
| **[`models/gbdt_trio_model.joblib`](file:///Users/tharunt/ola_prediction/models/gbdt_trio_model.joblib)** | **2.1 MB** | GBDT Trio Stacking Meta-Ensemble | `joblib.dump()` |
| **[`models/st_gnn_model.pt`](file:///Users/tharunt/ola_prediction/models/st_gnn_model.pt)** | **21.2 KB** | PyTorch Spatiotemporal GNN Weights | `torch.save(state_dict)` |

### Serving Predictions in Production (Python Code Example)

```python
from src.trainer import GBDTTrioTrainer
import pandas as pd

# 1. Load serialized GBDT Meta-Ensemble
trainer = GBDTTrioTrainer.load("models/gbdt_trio_model.joblib")

# 2. Predict multi-step demand vector for horizon t+1
feature_cols = trainer.feature_names
X_sample = features_df[feature_cols].tail(1)

pred_h1 = trainer.predict_horizon(X_sample, horizon=1)
pred_h4 = trainer.predict_horizon(X_sample, horizon=4)

print(f"Predicted Demand (t+1): {pred_h1[0]:.2f} rides")
print(f"Predicted Demand (t+4): {pred_h4[0]:.2f} rides")
```
