# 🔬 Research-Level Project Transformation Roadmap

## 📌 Vision & Objective
Transform the **Ola Bike Ride Request Demand Forecasting** codebase into a top-tier, publication-grade research project suitable for submission to high-impact journals (e.g., *IEEE Transactions on Intelligent Transportation Systems*, *Transportation Research Part C*) or premier conferences (*KDD*, *AAAI*, *IJCAI*).

---

## 🏛️ Research Gaps & Novel Contributions

Standard industrial demand forecasting pipelines focus on basic model accuracy. A publication-grade research paper requires addressing three fundamental research gaps:

1. **Static Spatial Boundaries vs. Dynamic Traffic Flows**:
   * *Gap*: Traditional $K$-means or grid clustering uses static Euclidean distances, ignoring physical road network constraints and dynamic travel times.
   * *Contribution*: Introduce **Adaptive Weather-Gated Spatiotemporal Graph Attention (WG-STGAT)** that dynamically adjusts inter-zone edge weights based on real-time precipitation and traffic velocity.

2. **Point Predictions vs. Operational Risk & Uncertainty**:
   * *Gap*: Standard point predictions ($\hat{y}$) do not inform fleet managers about prediction risk during volatile weather shifts.
   * *Contribution*: Incorporate **Conformal Prediction & Quantile Loss** to generate calibrated 95% confidence prediction intervals $[\hat{y}_{low}, \hat{y}_{high}]$, establishing a risk-aware bike rebalancing framework.

3. **Lack of Multi-City Spatiotemporal Generalizability**:
   * *Gap*: Most studies evaluate models on a single city dataset, risking overfitting to local geographic characteristics.
   * *Contribution*: Benchmark across **three multi-national datasets** (Ola India, Uber NYC, Chicago Ride-Share) with standardized spatial index mapping (Uber H3).

---

## 🗺️ Phase-by-Phase Execution Plan

```text
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 1: Novel Methodological & Uncertainty Enhancements          │
 │ • WG-STGAT Architecture (PyTorch Geometric)                     │
 │ • Conformal Prediction Intervals (Quantile Regression)           │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 2: Comprehensive Multi-Model & Multi-City Benchmarking     │
 │ • 10+ Baselines (ARIMA, Prophet, XGBoost, LightGBM, CatBoost,    │
 │   N-HiTS, PatchTST, ST-GCN, Graph WaveNet)                      │
 │ • Multi-City Datasets (Ola India, Uber NYC, Chicago Ride-Share)  │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 3: Ablation Studies & Statistical Significance             │
 │ • Spatial Indexing Ablation (K-Means vs. Uber H3 Res 7/8/9)      │
 │ • Extreme Weather Sensitivity & Horizon Scaling (t+1 to t+24)    │
 │ • Diebold-Mariano & Wilcoxon Signed-Rank Hypothesis Tests        │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 4: Explainable AI & Dynamic Spatiotemporal Visualizations  │
 │ • SHAP Global & Local Feature Attribution                        │
 │ • Graph Attention Heatmaps (Alpha_ij Spatiotemporal Spillover)   │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 5: IEEE / Elsevier LaTeX Manuscript Preparation             │
 │ • Double-Column IEEEtran / Elsevier LaTeX Paper (paper/main.tex) │
 │ • High-Resolution Vector Figures & BibTeX Reference Library      │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Detailed Milestone Specifications

### 📍 Phase 1: Novel Methodological & Uncertainty Enhancements
* **WG-STGAT Model Implementation (`src/wg_stgat_model.py`)**:
  * Build a custom PyTorch Geometric module combining spatial graph attention ($\mathbf{\alpha}_{ij}^{(t)}$) and gated temporal dilated convolutions.
  * Integrate an exogenous weather gating mechanism:
    $$\mathbf{W}_{ij}^{(t)} = \text{Softmax}_j\left( \text{LeakyReLU}\left( \mathbf{a}^T [\mathbf{W}_{ij} \,\|\, \text{weather}_t] \right) \right)$$
* **Uncertainty Quantification (`src/uncertainty.py`)**:
  * Implement Conformal Prediction and Pinball Loss (Quantile Regression at $\alpha \in \{0.05, 0.5, 0.95\}$) to yield distribution coverage guarantees.

---

### 📊 Phase 2: Comprehensive Benchmarking Suite
* **10+ Baseline Pipeline (`src/benchmarks.py`)**:
  1. *Statistical*: Auto-ARIMA, SARIMAX, Facebook Prophet.
  2. *Tree Ensembles*: XGBoost, LightGBM, CatBoost, Random Forest.
  3. *Deep Learning*: LSTM, ConvLSTM, N-HiTS (NeuralForecast), PatchTST, ST-GCN, Graph WaveNet, WG-STGAT.
* **Multi-City Unification (`src/multi_city_loader.py`)**:
  * Ingest and standardize:
    - **Ola Bike Ride Request Dataset** (India micro-mobility).
    - **Uber NYC Pickups Dataset** (High-density mega-city).
    - **Chicago Taxi & Ride-Share Dataset** (Midwest US urban grid).

---

### 🔬 Phase 3: Rigorous Ablation & Hypothesis Testing
* **Spatial Grid Ablation**:
  * Benchmark $K$-Means centroids vs. **Uber H3 Hexagonal Hierarchical Indexing** at Resolutions 7, 8, and 9.
* **Temporal Horizon Scaling**:
  * Evaluate forecast accuracy deterioration over expanded horizons: $t+1, t+2, t+4, t+6, t+12, t+24$ hours.
* **Extreme Weather Stress Test**:
  * Isolate test subset for severe weather events ($\text{precipitation} > 10\text{mm/hr}$) and compute conditional WAPE/RMSE lift.
* **Statistical Hypothesis Testing (`src/stats_tests.py`)**:
  * Compute **Diebold-Mariano Test** and **Wilcoxon Signed-Rank Test** $p$-values across model residual distributions to prove statistically significant superiority ($p < 0.01$).

---

### 🎨 Phase 4: Explainability & Interpretability
* **SHAP Feature Attribution (`src/explainability.py`)**:
  * Generate global SHAP summary bar charts, dependence plots, and force plots for individual peak-hour cluster forecasts.
* **Spatial Attention Spillover Visualizations**:
  * Export dynamic graph attention matrices $\mathbf{\alpha}_{ij}^{(t)}$ and render animated GeoJSON heatmap layers for the Next.js frontend dashboard.

---

### 📄 Phase 5: Publication Manuscript (`paper/`)
* **LaTeX Structure**:
  * `paper/main.tex` — Complete 10–12 page IEEE / Elsevier double-column LaTeX manuscript.
  * `paper/references.bib` — Comprehensive BibTeX file with 30+ peer-reviewed IEEE/Elsevier references (2021–2026).
  * `paper/figures/` — High-resolution vector PDF plots (WAPE comparison, H3 spatial grid, SHAP summary, DM test heatmaps).

---

## 📈 Summary Matrix of Research Contributions

| Metric / Dimension | Standard Coursework Project | Research-Level Paper (Proposed Plan) |
| :--- | :--- | :--- |
| **Model Scope** | Standard XGBoost / GBDT | **WG-STGAT + GBDT Stacking Ensemble** |
| **Predictions** | Point Predictions ($\hat{y}$) | **Point + 95% Conformal Prediction Bounds** |
| **Datasets** | Single Ola Dataset | **Multi-City (Ola India + Uber NYC + Chicago Taxi)** |
| **Spatial Indexing** | Simple $K$-Means | **Uber H3 Hexagonal Spatial Hierarchy (Res 7–9)** |
| **Baselines** | 1–2 Baselines (Random Forest) | **10+ Baselines (Statistical, Tree, Deep Transformers)** |
| **Statistical Rigor** | Basic Metrics (MAE, RMSE) | **Diebold-Mariano & Wilcoxon Signed-Rank Tests ($p < 0.01$)** |
| **Explainability** | None or basic plots | **SHAP Attributions + Dynamic Graph Attention Heatmaps** |
| **Deliverables** | Jupyter Notebooks / Readme | **IEEE double-column LaTeX manuscript & Reproducible Codebase** |
