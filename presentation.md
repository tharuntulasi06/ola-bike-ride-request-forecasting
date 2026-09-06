# 📊 Ola Bike Ride Request Demand Forecasting - Project Defense & Presentation Guide

> **Comprehensive presentation guide, slide deck outline, live demonstration walkthrough, and faculty defense Q&A for the Ola Bike Ride Request Spatiotemporal Demand Forecasting System.**

---

## 📅 Presentation Overview

- **Project Title**: Spatiotemporal Short-Horizon Demand Forecasting & Automated Fleet Control for Ola Bikes (Chennai)
- **Domain**: Smart Mobility, Intelligent Transportation Systems (ITS), Applied Spatiotemporal Machine Learning
- **Core Architecture**: Dual-Paradigm Ensemble (GBDT Trio + PyTorch Geometric ST-GNN) + DuckDB OLAP Engine + Next.js 14 Control Dashboard

---

## 🎬 Slide Deck Outline & Talking Points

### Slide 1: Title & Executive Summary
- **Title**: Ola Bike Ride Request Forecasting & Operational Control System
- **Core Goal**: Predict short-horizon ride request volumes ($t+1 \dots t+4$ intervals) per spatial zone and generate proactive bike rebalancing routes before demand surges occur.
- **Key Deliverable**: A full-stack AI platform combining GBDT Tweedie loss models, Graph Neural Networks, FastAPI microservice, and a Next.js 14 dashboard.

---

### Slide 2: Problem Statement & Motivation
- **Urban Mobility Inefficiencies**:
  - Ride-hailing platforms experience extreme spatiotemporal volatility.
  - Driver idle cruising leads to high fuel consumption and unfulfilled customer ride requests during peak hours.
  - Traditional models predict demand in isolation, ignoring neighborhood spillover effects.
- **Our Solution**:
  - A spatiotemporal graph model that learns how demand in one zone (e.g. Chennai Central) affects adjacent zones (e.g. T. Nagar).

---

### Slide 3: Data Pipeline & Spatial Clustering ($K=6$)
- **Multi-Dataset Strategy**:
  - Kaggle Ola Bike Ride Request Dataset (~17.4k hourly entries).
  - Kaggle Uber NYC & TLC Lat/Lon records (~4.5M trip logs) for spatial grid clustering.
  - OpenWeatherMap hourly weather feed (Temperature, Humidity, Rain mm/hr, Wind speed).
- **Geospatial Hotspots ($K=6$ Chennai Centroids)**:
  1. `Chennai Central` (`13.0827, 80.2707`)
  2. `T. Nagar` (`13.0418, 80.2341`)
  3. `OMR IT Corridor` (`12.9645, 80.2443`)
  4. `Velachery` (`12.9750, 80.2207`)
  5. `Guindy Kathipara` (`13.0067, 80.2020`)
  6. `CMBT / Anna Nagar` (`13.0850, 80.2101`)

---

### Slide 4: Feature Engineering & Distance Adjacency Matrix ($W_{ij}$)
- **Temporal Lags**: Autocorrelation lags ($t-1$, $t-24$, $t-168$) and rolling 3h/24h weather statistics.
- **Spatial Graph Construction**:
  - Distance Adjacency Matrix $W_{ij} = \exp\left(-\frac{d(i,j)^2}{\sigma^2}\right)$ calculated via Haversine physical distance between the 6 hotspot centroids.

---

### Slide 5: Dual-Paradigm ML Model Architecture
- **Paradigm 1: GBDT Trio Meta-Ensemble**:
  - **XGBoost**: Trained with **Tweedie Loss** ($1 < p < 2$) to handle zero-inflated, right-skewed count distributions.
  - **LightGBM**: Fast leaf-wise tree growth.
  - **CatBoost**: Categorical feature boosting.
  - Tuned via **Optuna TPE** hyperparameter search.
- **Paradigm 2: Spatiotemporal Graph Neural Network (ST-GNN)**:
  - Built with **PyTorch Geometric (`torch_geometric`)**.
  - Graph convolution layers propagate demand embeddings across spatial neighborhood edges.

---

### Slide 6: Model Evaluation & SHAP Interpretability
- **Evaluation Metrics**:
  - **WAPE (Weighted Absolute Percentage Error)**: Primary count metric (Zero-safe).
  - **MAE**, **RMSE**, and **$R^2$ Score**.
- **SHAP Feature Importance**:
  - Primary Drivers: `hour_of_day` (32%), `lag_1h` (24%), `rain_mm` (18%), `spatial_cluster_id` (12%).

---

### Slide 7: High-Performance Backend Microservice & DuckDB Engine
- **FastAPI Python Microservice (`api/main.py`)**:
  - REST endpoints for health check, spatial cluster lists, multi-horizon predictions, and automated fleet rebalancing dispatch.
- **DuckDB In-Memory OLAP Engine (`src/db.py`)**:
  - Runs ANSI SQL queries directly over compressed Apache Parquet feature files (`data/processed/spatiotemporal_features_clean.parquet`) with zero copy overhead.

---

### Slide 8: Next.js 14 Fleet Control Dashboard
- **Monochrome Liquid Glass Aesthetic**: Premium black canvas, backdrop blur (`backdrop-filter: blur(24px)`), and crisp white contrast typography.
- **Interactive Visualization Panels**:
  - **Spatial Heatmap (Leaflet.js)**: Chennai map rendering 6 centroids with animated pulsing circles.
  - **Multi-Horizon Forecast Curves (Recharts)**: Interactive $t+1 \dots t+4$ horizon selection.
  - **"What-If" Weather Simulator**: Live rain and temperature sliders recalculating predictions via REST POST requests.
  - **Automated Fleet Rebalancing Matrix**: Recommends surplus-to-deficit bike transfers with revenue uplift.
  - **ST-GNN Topology & SHAP Panels**: Interactive network graph and feature bar charts.

---

## 🖥️ Live Demonstration Steps (Step-by-Step for Reviewers)

1. **Step 1: Start Backend API & Launch Dashboard**
   ```bash
   # Terminal 1: Backend FastAPI
   ./venv/bin/python api/main.py

   # Terminal 2: Next.js Dashboard
   cd dashboard && npm run dev
   ```
   *Open browser at `http://localhost:3000` (or `http://localhost:3002`).*

2. **Step 2: Demonstrate Interactive Spatial Map**
   - Click on `T. Nagar` or `OMR IT Corridor` on the Leaflet map.
   - Show how all forecast curves, rebalancing matrix, and What-If simulator instantly update focus to that specific centroid.

3. **Step 3: Demonstrate "What-If" Weather Scenario Simulator**
   - Click the `🌧️ Downpour` preset button (Rain = 40 mm/h).
   - Show how predicted demand spikes and the **Predicted Surge Multiplier** increases in real-time.

4. **Step 4: Demonstrate Automated Fleet Rebalancing Matrix**
   - Show the recommended transfer route (*"Transfer 18 bikes from Guindy Kathipara to T. Nagar"*).
   - Click the **"Dispatch"** button to show state transition to `En Route`.

5. **Step 5: Demonstrate Report Exporter**
   - Click **"Export Report"** in the header.
   - Download the `.csv` dataset and `.txt` Executive Control Summary.

---

## ❓ Frequently Asked Faculty Questions & Answers

### Q1: Why use Tweedie Loss instead of standard Mean Squared Error (MSE)?
> **Answer**: Ride request counts are zero-inflated (many low-demand hours) and right-skewed (extreme surge peaks). MSE penalizes errors symmetrically and assumes normally distributed residuals. Tweedie loss models Poisson-Gamma compound distributions, accurately fitting zero-demand periods without underpredicting extreme demand spikes.

### Q2: How does the Graph Neural Network (ST-GNN) differ from GBDT?
> **Answer**: GBDT models evaluate feature tables row-by-row. The ST-GNN treats Chennai's hotspots as nodes in a spatial graph connected by Haversine physical distance edges ($W_{ij}$). Graph convolutions pass feature messages across edges, allowing the model to capture how demand spikes in one cluster spill over into adjacent hubs.

### Q3: Why is DuckDB used instead of a traditional SQL server like PostgreSQL?
> **Answer**: DuckDB is an in-memory column-oriented OLAP engine embedded directly in Python. It queries compressed Apache Parquet files directly with zero data copying and zero database server setup overhead, providing microsecond query response times for analytics.

### Q4: Is the frontend showing live model output or mock data?
> **Answer**: The frontend fetches live predictions from the FastAPI microservice (`/api/v1/predict`), which executes model inference on `models/gbdt_trio_model.joblib` over `spatiotemporal_features_clean.parquet`.
