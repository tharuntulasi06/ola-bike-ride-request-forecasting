# 👥 Team Implementation & Contribution Plan (Team of Two)

> **Project**: Ola Bike Ride Request Demand Forecasting Using Geospatial Clustering and Multi-Step Time-Series Machine Learning  
> **Target Scope**: Modular City-Aware Architecture with Chennai Flagship Case Study & Full-Stack Deployment  
> **Team Structure**: 2 Members (Peer Collaboration Division)

---

## 🎯 Executive Team Division & Ownership Matrix

To ensure clean git workflow, zero code conflicts, and equal individual contribution for academic grading/viva defense, the responsibilities are split into two distinct engineering tracks:

| Role & Focus | Primary Responsibilities | Main Code & File Ownership |
| :--- | :--- | :--- |
| **Member 1: ML Engine & Backend Architect** *(Data & Tree Ensembles)* | Data ingestion, temporal resampling, GBDT Trio models (XGBoost, LightGBM, CatBoost), Optuna hyperparameter tuning, and FastAPI REST microservice. | `src/data_loader.py`<br>`src/trainer.py`<br>`api/main.py`<br>`api/schemas.py`<br>`tests/test_data_loader.py`<br>`tests/test_trainer.py` |
| **Member 2: Deep Learning & Full-Stack Lead** *(Geospatial & Dashboard)* | Spatial clustering (`MiniBatchKMeans`), PyTorch Geometric ST-GNN (Graph WaveNet), feature engineering (`W_ij` matrix), and Next.js / Leaflet Fleet Operational Dashboard. | `src/feature_builder.py`<br>`src/st_gnn_model.py`<br>`src/evaluate.py`<br>`dashboard/` (Next.js/React)<br>`tests/test_feature_builder.py` |

---

## 📅 Phase-by-Phase Execution & Contribution Timeline

```text
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                               PHASE BREAKDOWN                                   │
 ├───────────────────┬───────────────────┬───────────────────┬─────────────────────┤
 │ Phase 1 (Week 1)  │ Phase 2 (Week 2)  │ Phase 3 (Week 3)  │ Phase 4 & 5 (W4-5)  │
 │ Data & Ingestion  │ Spatial & Lags    │ GBDT & ST-GNN     │ FastAPI & Next.js   │
 └───────────────────┴───────────────────┴───────────────────┴─────────────────────┘
```

---

### 🟢 Phase 1: Data Ingestion Pipeline & Data Quality (Week 1)

* **Goal**: Establish continuous hourly dataset grids, handle missing values, and ingest multi-source datasets (Ola, Uber GPS, OpenWeather, NYC TLC).

#### Member 1 Tasks:
* Implement `OlaDataLoader` in `src/data_loader.py` (temporal 1-hour resampling `freq='1h'`, linear interpolation for weather).
* Create synthetic data generators for hourly Ola ride requests and OpenWeather exogenous logs.
* Write automated unit test suite `tests/test_data_loader.py`.
* **Git Branch**: `feature/data-pipeline`

#### Member 2 Tasks:
* Configure raw GPS coordinate parser for spatial trip logs (Lat, Lon, Timestamp).
* Implement NYC TLC benchmark data ingestion and hourly zonal aggregation matrix.
* Set up root environment dependencies (`requirements.txt`, `.gitignore`, `src/__init__.py`).
* **Git Branch**: `feature/data-pipeline`

---

### 🟡 Phase 2: Geospatial Clustering & Spatiotemporal Feature Engineering (Week 2)

* **Goal**: Partition pickup points into localized demand zones, build ACF/PACF autocorrelation lags, and construct the Haversine Spatial Graph Adjacency Matrix ($W_{ij}$).

#### Member 1 Tasks:
* Construct ACF/PACF temporal lag features ($t-1, t-2, t-3, t-24, t-48, t-168$) per spatial zone.
* Implement non-leakage trailing rolling weather statistics ($3\text{h}, 6\text{h}, 24\text{h}$ shifted windows).
* Build Direct Multi-Step Horizon target vectors (`target_h1 \dots target_h4`).
* **Git Branch**: `feature/feature-engineering`

#### Member 2 Tasks:
* Implement `MiniBatchKMeans` spatial clustering in `src/feature_builder.py` targeting the **6 Chennai Hotspot Centroids** (Chennai Central, T. Nagar, OMR IT Corridor, Velachery, Guindy, CMBT).
* Construct the **Gaussian Thresholded Haversine Spatial Graph Adjacency Matrix ($W_{ij}$)** ($6 \times 6$) for ST-GNN.
* Implement smooth trigonometric cyclical time encodings ($\sin/\cos$ for hour, day of week, month).
* Write unit tests in `tests/test_feature_builder.py`.
* **Git Branch**: `feature/spatial-clustering`

---

### 🔴 Phase 3: Dual-Paradigm Model Architecture & Training Engine (Week 3)

* **Goal**: Train GBDT Trio ensembles alongside PyTorch Geometric ST-GNN and tune hyperparameters via Optuna.

#### Member 1 Tasks:
* Implement GBDT training engine in `src/trainer.py` for **XGBoost** (Tweedie/Poisson loss $1 < p < 2$), **LightGBM**, and **CatBoost**.
* Configure **Optuna** automated hyperparameter optimization with TPE sampling.
* Build the **Weighted Stacking Meta-Ensemble** combining GBDT out-of-fold predictions.
* **Git Branch**: `feature/gbdt-ensemble`

#### Member 2 Tasks:
* Implement **Spatiotemporal Graph Neural Network (ST-GNN / Graph WaveNet)** in PyTorch Geometric using $W_{ij}$ graph convolutions.
* Implement model evaluation suite in `src/evaluate.py` calculating **WAPE**, MAE, RMSE, $R^2$, and zero-count residuals.
* Generate **SHAP (SHapley Additive exPlanations)** feature importance dependence plots.
* **Git Branch**: `feature/st-gnn-model`

---

### 🔵 Phase 4: Full-Stack Web Architecture & REST API (Week 4)

* **Goal**: Build an asynchronous FastAPI inference microservice and interactive Next.js / Leaflet Fleet Management Dashboard.

#### Member 1 Tasks:
* Implement **FastAPI** REST microservice in `api/main.py`:
  * `POST /api/v1/predict` (serves multi-step $t+1 \dots t+4$ hour forecast vectors).
  * `GET /api/v1/analytics/metrics` (serves live WAPE, MAE, RMSE stats).
* Define Pydantic v2 schemas in `api/schemas.py`.
* **Git Branch**: `feature/fastapi-backend`

#### Member 2 Tasks:
* Develop **Next.js 14 / React** Fleet Operations Dashboard in `dashboard/`:
  * Interactive **Leaflet.js / Mapbox** spatial heatmaps overlaying Chennai landmarks.
  * Real-time hourly demand actual vs. predicted curves using **Recharts**.
  * Dynamic zone selection & fleet rebalancing alert pills.
* **Git Branch**: `feature/nextjs-dashboard`

---

### 🟣 Phase 5: Evaluation Benchmarking, Documentation & Presentation (Week 5)

* **Goal**: Finalize documentation, verify cross-city generalizability, and prepare presentation slide deck for viva defense.

#### Member 1 Tasks:
* Benchmark models on secondary **NYC TLC Taxi Dataset** for cross-city generalizability.
* Finalize **`PROJECT_PROPOSAL.md`** & **`ML_PROPOSAL.md`**.
* Prepare Slide Deck Speaker Notes for GBDT Tweedie loss and API backend slides in **`SLIDES_PPT.md`**.

#### Member 2 Tasks:
* Finalize **`README.md`** & **`walkthrough.md`** with embedded screenshot/recording artifacts.
* Prepare Slide Deck Speaker Notes for ST-GNN graph convolution and Leaflet UI slides in **`SLIDES_PPT.md`**.
* Perform end-to-end integration test of FastAPI backend + Next.js dashboard.

---

## 🗣️ Viva Presentation & Oral Defense Distribution

When presenting the project to faculty examiners, divide the oral presentation as follows:

| Speaker | Topics Covered in Viva Defense | Key Slides (`SLIDES_PPT.md`) |
| :--- | :--- | :--- |
| **Member 1** | • Problem Statement & Real-World Business Need<br>• Data Ingestion Pipeline & Temporal Resampling<br>• GBDT Trio (XGBoost Tweedie Loss, LightGBM, CatBoost)<br>• FastAPI Inference REST Microservice Architecture | Slides 1, 2, 4, 6, 7 (GBDTs), 9 (Backend), 11 |
| **Member 2** | • Chennai Spatial Clustering (`MiniBatchKMeans`) & Hotspots<br>• Haversine Distance & Spatial Graph Adjacency Matrix ($W_{ij}$)<br>• PyTorch Geometric Spatiotemporal GNN (ST-GNN)<br>• Next.js + Leaflet Operational Fleet Dashboard & WAPE Benchmarks | Slides 3, 5, 7 (ST-GNN), 8, 9 (Frontend UI), 10, 12 |

---

## 🛠️ Git Collaboration Rules & Workflow

1. **Main Branch Protection**: `main` branch holds production-ready, passing code only.
2. **Feature Branching**: Always create feature branches named `feature/<track-name>` (e.g. `git checkout -b feature/data-pipeline`).
3. **Commit Message Standards**: Use conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
4. **Pull Request Review**: Before merging to `main`, run unit tests:
   ```bash
   python3 tests/test_data_loader.py
   python3 tests/test_feature_builder.py
   ```
