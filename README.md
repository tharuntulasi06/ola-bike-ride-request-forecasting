# 🚴 Ola Bike Ride Request Demand Forecasting

> **Spatiotemporal Micro-Mobility Demand Forecasting using Geospatial Clustering (`MiniBatchKMeans`), Gradient Boosting Trio Ensembles (`XGBoost`, `LightGBM`, `CatBoost`), PyTorch Geometric Graph Neural Networks (`ST-GNN`), Embedded DuckDB SQL Engine, and a Full-Stack Operational Dashboard (`FastAPI` + `Next.js`).**

---

## 📌 Project Overview

Ride-sharing and micro-mobility platforms like **Ola** face severe operational inefficiencies due to high spatiotemporal volatility in ride request demand. Static or purely reactive driver positioning leads to extended customer wait times (ETA), excessive unpaid driver idle cruising, unfulfilled ride requests during peak rush hours, and suboptimal dynamic surge pricing.

This repository provides an end-to-end production-grade machine learning system to accurately forecast short-horizon ride request volumes ($t+1$ to $t+4$ hours) per spatial cluster. Fleet managers and automated dispatch algorithms can proactively rebalance idle bikes to predicted high-demand zones before surge spikes occur.

---

## ✨ Key System Features

* 🏛️ **Modular City-Aware Architecture (Chennai Case Study)**: Configurable city pipeline featuring **Chennai** as the primary flagship case study (OMR IT Corridor, Guindy Kathipara, T. Nagar, Chennai Central, Velachery, CMBT) with cross-city generalizability benchmarking.
* 📍 **Geospatial Hotspot Allocation**: Groups pickup coordinates into localized demand zones using `MiniBatchKMeans` spatial clustering.
* ⏰ **Autocorrelation & Temporal Engineering**: Extracts statistical lag features ($t-1$, $t-24$, $t-168$) using ACF/PACF analysis alongside cyclical sine/cosine time encodings.
* 🌤️ **Exogenous Weather & Holiday Interactions**: Incorporates rolling window temperature, "feels-like" temperature, humidity, windspeed, precipitation, and public holiday markers.
* ⚡ **Gradient Boosting Trio Ensemble**: Combines **XGBoost** (Tweedie loss $1 < p < 2$), **LightGBM** (leaf-wise speed), and **CatBoost** (ordered categorical boosting) into a weighted stacking meta-ensemble tuned via Optuna TPE.
* 🕸️ **Spatiotemporal Graph Neural Network**: Models physical distance spatial adjacency ($W_{ij}$) via **PyTorch Geometric (`torch_geometric`)** to capture neighborhood demand spillover.
* 🦆 **DuckDB In-Memory SQL Engine**: Executes high-speed zero-overhead ANSI SQL analytical queries directly on compressed Apache Parquet dataset files.
* 📊 **Segmented Evaluation & Explainability**: Benchmarks models using **WAPE (Weighted Absolute Percentage Error)**, MAE, RMSE, and $R^2$ scores, generating SHAP summary plots and residual histograms in `results/figures/`.
* 🖥️ **Full-Stack Production Architecture**: Features a high-performance **FastAPI** Python inference microservice and an interactive **Next.js 14 / React** Fleet Management Dashboard with `Leaflet.js` spatial heatmaps.

---

## 🏗️ System Architecture

```text
 ┌──────────────────────────────────────────────────────────────────┐
 │      Multi-Source Data Ingestion (Ola, Uber NYC, Weather API)    │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Step 1: Spatial Clustering & Feature Engineering                 │
 │ • MiniBatchKMeans spatial hotspot partitioning (K=6 Chennai)     │
 │ • Spatial Graph Distance Adjacency Matrix (W_ij)                 │
 │ • ACF/PACF Lags (t-1, t-24, t-168) & Rolling Weather Stats      │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Step 2: Dual-Paradigm Model Training & Benchmarking              │
 │ • GBDT Trio: XGBoost (Tweedie), LightGBM, CatBoost               │
 │ • ST-GNN: PyTorch Geometric Spatiotemporal Graph WaveNet         │
 │ • Optuna TPE Tuning & Weighted Stacking Meta-Ensemble            │
 └──────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Step 3: Embedded Database Analytics & Deployment                 │
 │ • DuckDB SQL Engine for instant Parquet analytics                │
 │ • FastAPI REST Endpoints (/api/v1/predict, /api/v1/clusters)     │
 │ • Next.js / React Fleet UI with Leaflet.js Spatial Heatmaps      │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Multi-Dataset Strategy & Data Engineering

| Dataset | Type / Source | Volume | Primary Role |
| :--- | :--- | :--- | :--- |
| **Ola Bike Ride Request** | Kaggle (`palvinder2006/ola-bike-ride-request`) | ~17,379 hourly records | Primary target demand forecasting ($t+1 \dots t+4$) & weather sensitivity |
| **Uber NYC GPS Pickups** | Kaggle (`fivethirtyeight/uber-pickups-in-new-york-city`) | ~4.5M raw trip logs | Granular Lat/Lon evaluation for `MiniBatchKMeans` spatial clustering |
| **NYC TLC Taxi & FHV** | Kaggle (`anaghbar81/tlc-trip-record-data`) | ~10M trip records | Cross-city scalability benchmark across 263 discrete taxi zones |
| **OpenWeatherMap & Holiday** | Kaggle (`muthuj7/weather-dataset`) | Hourly continuous feed | Exogenous precipitation, visibility, barometric pressure & holiday flags |

> ⚡ **Direct Kaggle API & Apache Parquet Storage**:
> Datasets are dynamically fetched at runtime via `kagglehub.dataset_download()` and processed into compressed **Apache Parquet (`.parquet`)** format (`pyarrow`). **Zero data files are tracked in Git**, maintaining a lightweight repository size (< 3 MB).

---

## 🛠️ Technology Stack

* **ML & DL Engine**: Python 3.10+, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`, `torch_geometric`, `duckdb`, `pandas`, `numpy`, `statsmodels`, `optuna`, `joblib`
* **Backend REST API**: `FastAPI`, `uvicorn`, `pydantic` v2, `httpx`
* **Frontend Fleet Dashboard**: `Next.js 14+` / `React 18+`, `TailwindCSS`, `Lucide Icons`
* **Geospatial & Charts**: `Leaflet.js` / `React-Leaflet`, `Mapbox GL`, `Recharts`, `Matplotlib`
* **DevOps & Testing**: Pytest, GitHub Actions CI/CD Workflow, Git, Docker

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Node.js 18+** installed.

```bash
git clone https://github.com/tharuntulasi06/ola-bike-ride-request-forecasting.git
cd ola-bike-ride-request-forecasting
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pipeline Execution & Training
```bash
# 1. Run Data Ingestion & Preprocessing
python src/data_loader.py

# 2. Run Spatial Clustering & Feature Engineering
python src/feature_builder.py

# 3. Train Models & Save Checkpoints to models/
python src/train.py

# 4. Generate Evaluation Metrics & SHAP Plots in results/
python src/explainability.py

# 5. Run DuckDB SQL Analytics Engine
python src/db.py
```

### 4. Running the Backend Inference API
```bash
# Start FastAPI server on http://localhost:8000
python -m uvicorn api.main:app --reload --port 8000
```

### 5. Running Next.js 14 Fleet Control Dashboard
```bash
# Start Next.js operational dashboard on http://localhost:3000
cd dashboard
npm install
npm run dev
```

### 6. Running Automated Test Suite
```bash
# Run all 23 unit tests
python -m pytest tests/ -v
```

---

## 📄 Key Project Deliverables

* 📊 **[presentation.md](presentation.md)** — Project Defense Guide, Slide Deck Outline, Live Demo Walkthrough & Faculty Q&A.
* 🧠 **[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)** — Dual-Paradigm Model Specifications, Tweedie Loss Math & Benchmarks.
* 🧠 **[ML_PROPOSAL.md](ML_PROPOSAL.md)** — Machine Learning Framework Architecture & Design Proposal.
* 📖 **[PROJECT_PROPOSAL.md](PROJECT_PROPOSAL.md)** — Comprehensive Academic Project Proposal & System Design.
* 📊 **[SLIDES_PPT.md](SLIDES_PPT.md)** — Faculty Presentation Slide Deck with Speaker Notes.
* 📋 **[TEAM_IMPLEMENTATION_PLAN.md](TEAM_IMPLEMENTATION_PLAN.md)** — Peer Contribution Plan & Technical Responsibilities.

---

## 📜 License

This project is licensed under the MIT License — see the `LICENSE` file for details.
