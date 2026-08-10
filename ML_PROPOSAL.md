# 🧠 Machine Learning Architecture & Design Proposal

## 📌 Document Overview
* **Project**: Ola Bike Ride Request Demand Forecasting
* **Domain**: Spatiotemporal Machine Learning & Micro-Mobility Analytics
* **Focus**: Machine Learning Framework Selection, Mathematical Formulation, Feature Engineering Pipeline, Ensemble Architecture, and Model Evaluation.

---

## 1. Problem Formulation

Let $K$ denote the total number of spatial clusters generated via `MiniBatchKMeans` spatial partitioning, where $k \in \{1, 2, \dots, K\}$. Let $t \in \{1, 2, \dots, T\}$ index continuous hourly temporal intervals.

The goal is to learn a mapping function $f: \mathbf{X}_{k, t} \to \mathbf{Y}_{k, t}$ that predicts the ride request volume $\mathbf{Y}_{k, t}$ over a multi-step forecasting horizon $H \in \{1, 2, 3, 4\}$ hours ahead:

$$\mathbf{Y}_{k, t} = \left[ y_{k, t+1}, y_{k, t+2}, y_{k, t+3}, y_{k, t+4} \right] \in \mathbb{R}_{\ge 0}^4$$

where $\mathbf{X}_{k, t} \in \mathbb{R}^d$ represents a $d$-dimensional feature vector computed up to time $t$.

---

## 2. Feature Engineering & Matrix Construction

The input feature matrix $\mathbf{X}_{k, t}$ incorporates four distinct feature families:

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       Input Feature Matrix X_(k,t)                      │
 ├───────────────────┬───────────────────┬────────────────┬────────────────┤
 │ Autocorrelation   │ Rolling Weather   │ Cyclical Time  │ Spatial Zone   │
 │   Lags (ACF/PACF) │   Statistics      │ Encodings      │ Embeddings     │
 └───────────────────┴───────────────────┴────────────────┴────────────────┘
```

### 2.1 Autocorrelation Lag Features (ACF / PACF Analysis)
Captures three critical temporal properties of urban human mobility:
* **Closeness (Short-Term Memory)**: $y_{k, t-1}, y_{k, t-2}, y_{k, t-3}$ (immediate past hours).
* **Daily Periodicity (Diurnal Pattern)**: $y_{k, t-24}, y_{k, t-25}, y_{k, t-48}$ (same hours yesterday and 2 days prior).
* **Weekly Trend (Seasonal Pattern)**: $y_{k, t-168}$ (same hour, same day last week).

### 2.2 Rolling Window Weather & Environmental Features
Computes non-leakage trailing statistics over windows $W \in \{3\text{h}, 6\text{h}, 24\text{h}\}$:
* **Temperature & Feels-Like**: $\mu_{temp}(W), \sigma_{temp}(W), \Delta_{temp} = temp_t - temp_{t-1}$
* **Relative Humidity & Windspeed**: $\mu_{humidity}(W), \mu_{windspeed}(W)$
* **Precipitation Severity**: Categorical weather severity $w_t \in \{1: \text{Clear}, 2: \text{Cloudy}, 3: \text{Light Rain}, 4: \text{Heavy Rain}\}$ and rain depth (mm).

### 2.3 Cyclical Temporal Encodings
Encodes periodic time cycles using trigonometric transformation to preserve continuity between $23:00$ and $00:00$:

$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$

$$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot d}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot d}{7}\right)$$

### 2.4 Spatial Hotspot Allocation (`MiniBatchKMeans`)
Groups raw pickup latitude and longitude coordinates $(\text{lat}_i, \text{lon}_i)$ into $K$ centroid zones:

$$\min_{S} \sum_{k=1}^{K} \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

---

## 3. Machine Learning Framework Comparison & Selection

We evaluate and compare four machine learning algorithms tailored for structured spatiotemporal tabular data:

| ML Framework | Primary Advantage | Loss Function / Objective | Operational Role |
| :--- | :--- | :--- | :--- |
| **XGBoost** (`xgboost`) | Exact tree splitting, regularization ($L_1/L_2$), robust feature importance | Tweedie ($1 < p < 2$) / Poisson | **Primary Regressor** |
| **LightGBM** (`lightgbm`) | Leaf-wise tree growth, GOSS, EFB, 10x-15x faster execution speed | L1 / Huber / Tweedie | **Fast Large-Scale Regressor** |
| **CatBoost** (`catboost`) | Ordered Boosting, native categorical encoding for spatial cluster IDs | RMSE / MAE | **Categorical Spatial Regressor** |
| **Random Forest** (`scikit-learn`) | Bagging ensemble, non-parametric baseline | MSE | **Non-Parametric Baseline** |

---

### 3.1 XGBoost Regressor (Extreme Gradient Boosting)
XGBoost minimizes a regularized objective function at step $m$:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{N} l\left(y_i, \hat{y}_i^{(m-1)} + f_m(\mathbf{x}_i)\right) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$

* **Tweedie Loss Objective**: Handles zero-inflated off-peak request counts:

$$l(y, \hat{y}) = -y \frac{e^{\hat{y}(1-p)}}{1-p} + \frac{e^{\hat{y}(2-p)}}{2-p}, \quad 1 < p < 2$$

---

### 3.2 LightGBM (Light Gradient Boosting Machine)
LightGBM uses **Gradient-based One-Side Sampling (GOSS)** to filter data instances with small gradients and **Exclusive Feature Bundling (EFB)** to bundle sparse features. This enables ultra-fast training over multi-million record datasets (like NYC TLC & Uber NYC logs).

---

### 3.3 CatBoost (Categorical Boosting)
CatBoost uses **Ordered Boosting** to combat target leakage occurring in traditional GBDTs when calculating target statistics for spatial cluster IDs. It builds symmetric (oblivious) trees to prevent overfitting during sudden weather transitions.

---

## 4. Multi-Step Horizon Strategy

To predict demand across 4 discrete hours ahead ($t+1, t+2, t+3, t+4$), we implement the **Direct Multi-Step Strategy**:

$$\hat{y}_{k, t+h} = f_h\left(\mathbf{X}_{k, t}\right), \quad \text{for } h \in \{1, 2, 3, 4\}$$

* **Why Direct Strategy?**: Unlike recursive multi-step forecasting (which feeds $t+1$ predictions back as inputs for $t+2$), the direct strategy trains 4 independent models $(f_1, f_2, f_3, f_4)$. This completely eliminates compounding prediction error drift over longer time horizons.

---

## 5. Weighted Stacking Meta-Ensemble

The final system output combines out-of-fold predictions from XGBoost, LightGBM, and CatBoost:

$$\hat{y}_{\text{final}, h} = w_{1, h} \cdot \hat{y}_{\text{XGB}, h} + w_{2, h} \cdot \hat{y}_{\text{LGB}, h} + w_{3, h} \cdot \hat{y}_{\text{Cat}}, h$$

Subject to the constraints:

$$\sum_{m=1}^{3} w_{m, h} = 1 \quad \text{and} \quad w_{m, h} \ge 0$$

Weights $\mathbf{w}$ are optimized using Nelder-Mead / L-BFGS-B optimization on the out-of-fold validation predictions to minimize overall WAPE.

---

## 6. Hyperparameter Optimization Framework (Optuna)

Hyperparameters for all three GBDT models are automatically tuned using **Optuna** with Tree-structured Parzen Estimator (TPE) sampling:

```python
# Search Space Definition
search_space = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
}
```

---

## 7. Model Evaluation Metrics & Interpretability

### 7.1 Performance Metrics
1. **Weighted Absolute Percentage Error (WAPE)** *(Primary Metric)*:
   $$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$
   *(Handles zero counts gracefully without division-by-zero errors present in standard MAPE).*

2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

3. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

4. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

### 7.2 Explainable AI (SHAP Interpretability)
We generate **SHAP (SHapley Additive exPlanations)** summary and dependence plots to measure feature importance:

$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left( f(S \cup \{i\}) - f(S) \right)$$

---

## 8. Summary of ML Architecture Deliverables

* 📦 `src/data_loader.py` — Ingestion & dataset unification.
* 📦 `src/feature_builder.py` — ACF/PACF lag generation, rolling weather stats, and cyclical encodings.
* 📦 `src/trainer.py` — Multi-step XGBoost, LightGBM, and CatBoost training with Optuna tuning.
* 📦 `src/evaluate.py` — WAPE, MAE, RMSE calculation, residual analysis, and SHAP plots.
