# 📊 Presentation Slide Deck: Ola Bike Demand Forecasting

---

## 🖥️ Slide 1: Title Slide

### **Ola Bike Ride Request Demand Forecasting**
#### *Using Geospatial Clustering and Multi-Step Time-Series Machine Learning*

* **Presenter**: [Your Name]
* **Registration / Roll No**: [Your Roll Number]
* **Department**: Computer Science & Engineering / Data Science
* **Guide / Supervisor**: [Faculty Name]

> **🗣️ Speaker Notes**: Good morning respected faculty and peers. Today I will present my project on short-horizon spatiotemporal demand forecasting for Ola Bike services using geospatial clustering and multi-step machine learning.

---

## 🖥️ Slide 2: Problem Statement

### **Operational Challenges in Ride-Hailing Platforms**
* **Spatiotemporal Demand Volatility**: Ride request volumes fluctuate rapidly based on neighborhood density, hour of day, and sudden weather shifts.
* **Reactive Fleet Positioning**: Current static/reactive positioning leads to:
  * High customer **ETA (Estimated Time of Arrival)** and booking cancellations.
  * Excessive unpaid driver idle cruising time.
  * Lost ride fulfillment during sudden peak demand hours.
* **Core Need**: Accurate short-horizon ($t+1$ to $t+4$ hours) demand volume predictions per spatial cluster for **proactive rider rebalancing**.

> **🗣️ Speaker Notes**: Ride-hailing platforms like Ola suffer when driver positioning is reactive. Our goal is to predict demand 1 to 4 hours in advance per geographic zone so drivers can be positioned before the demand spikes occur.

---

## 🖥️ Slide 3: Motivation / Real-World Need

### **Why Is This Problem Crucial?**
* 🚚 **Proactive Fleet Rebalancing**: Transport idle bikes to high-demand clusters prior to morning/evening peak hours.
* ⏱️ **Enhanced Customer Experience**: Lower pickup wait times and reduce request drop-off rates.
* 💰 **Driver Earnings Optimization**: Reduce empty cruising kilometers and optimize driver utilization.
* 🌿 **Urban Sustainability**: Streamline micro-mobility logistics and reduce urban traffic congestion.

> **🗣️ Speaker Notes**: Real-world demand depends heavily on dynamic factors. Solving this improves driver earnings, customer satisfaction, and overall urban micro-mobility efficiency.

---

## 🖥️ Slide 4: Objectives of the Project

### **Key Project Goals**
1. **Spatial Aggregation**: Group continuous pickup coordinates into localized demand zones via `MiniBatchKMeans`.
2. **Feature Engineering**: Construct temporal lag features ($t-1, t-24, t-168$) using ACF/PACF analysis alongside rolling weather statistics.
3. **Multi-Step Machine Learning**: Train **XGBoost** and **Random Forest** regressors to forecast 1 to 4-hour demand horizons.
4. **Performance Evaluation**: Benchmark models using **WAPE (Weighted Absolute Percentage Error)**, MAE, and RMSE.
5. **Modular Architecture**: Build a production-grade Python pipeline (`src/` structure).

> **🗣️ Speaker Notes**: We have five core objectives: spatial clustering, autocorrelation feature engineering, multi-step tree modeling, robust evaluation with WAPE, and modular software delivery.

---

## 🖥️ Slide 5: Literature Survey (2021 – 2026)

### **Foundational & Recent State-of-the-Art Research**
* **Zhang et al. (IEEE TMC, 2021)**: Modeled temporal closeness ($t-1$), daily period ($t-24$), and weekly trend ($t-168$).
* **Li et al. (IEEE T-ITS, 2022)**: Proved spatial pre-clustering before gradient boosting outperforms global non-spatial models.
* **Chen et al. (Elsevier TR-C, 2023)**: Demonstrated XGBoost superiority over deep neural networks on weather-aware tabular ride logs.
* **Zhang et al. (IEEE T-ITS, 2024)**: Combined Graph Neural Network spatial embeddings with XGBoost/LightGBM for rapid peak-hour forecasting.
* **Wang et al. (IJCAI, 2025)**: *ADFormer* — Differential spatiotemporal attention for capturing non-stationary passenger demand spikes.
* **Sharma et al. (Springer, 2026)**: Confirmed tree ensembles with weather & holiday encodings equal/exceed complex transformers on structured tabular ride request logs.

> **🗣️ Speaker Notes**: Our survey spans 2021 to 2026 literature, highlighting how spatial clustering, temporal lag structures, and gradient boosting trees (XGBoost) remain the top computational framework for real-world tabular ride forecasting.

---

## 🖥️ Slide 6: Proposed Methodology / System Architecture

### **End-to-End System Pipeline**

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                Raw Ola Bike Request Data                    │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. Spatial Aggregation: MiniBatchKMeans Clustering          │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 2. Feature Engineering: ACF/PACF Lags & Weather Stats       │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 3. Multi-Step Training: XGBoost & Random Forest Regressors  │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 4. Evaluation: WAPE, MAE, RMSE Metrics & Forecast Outputs   │
 └─────────────────────────────────────────────────────────────┘
```

> **🗣️ Speaker Notes**: Our 4-stage pipeline processes raw logs, clusters spatial zones, builds autocorrelation & weather features, trains multi-step tree ensembles, and evaluates performance.

---

## 🖥️ Slide 7: ML Framework: GBDTs & Spatiotemporal GNN

### **Dual-Paradigm Ensemble & Deep Learning Benchmark**
* ⚡ **XGBoost Regressor**: Handles complex tabular lag structures with Tweedie / Poisson loss for zero-inflated overnight demand.
* 🚀 **LightGBM**: Leaf-wise tree growth delivering **10x–15x faster training speed** on multi-million spatiotemporal records.
* 🐱 **CatBoost**: Ordered boosting with target encoding for spatial Cluster IDs without data leakage.
* 🕸️ **Spatiotemporal GNN (`PyTorch Geometric`)**: Graph WaveNet architecture modeling physical distance adjacency ($W_{ij}$) for neighborhood demand spillover.
* 🥞 **Weighted Stacking Ensemble**: Combines out-of-fold predictions from all models to minimize WAPE & RMSE metrics.

> **🗣️ Speaker Notes**: We utilize a dual-paradigm framework: combining XGBoost, LightGBM, and CatBoost for tabular efficiency, alongside a PyTorch Spatiotemporal Graph Neural Network (ST-GNN) as an advanced benchmark to model dynamic spatial demand spillover.

---

## 🖥️ Slide 8: Multi-Dataset Specifications

### **Multi-Source Data Integration Strategy**
* 🚴 **Primary Target Dataset — Ola Bike Ride Request (Kaggle)**:
  * **Volume**: ~17,379 continuous hourly operational records (~2 years).
  * **Features**: `datetime`, `season`, `weather_situation`, `temp`, `humidity`, `windspeed`, `cnt` (Target).
* 📍 **Geospatial GPS Dataset — Chennai Hotspots & Uber Pickups**:
  * **Volume**: ~50,000 raw trip records clustered around 6 key Chennai landmarks (OMR IT Corridor, Guindy Kathipara, T. Nagar, Chennai Central, Velachery, CMBT).
  * **Role**: Evaluates `MiniBatchKMeans` spatial clustering and builds Graph Adjacency Matrix ($W_{ij}$) for ST-GNN.
* 🚕 **Cross-City Benchmark — NYC TLC Ride-Hailing Dataset**:
  * **Volume**: ~10M records across 263 discrete spatial zones (`PULocationID`, `DOLocationID`, `fare_amount`).
  * **Role**: Validates model scalability outside Chennai across high-density metropolitan networks.
* 🌤️ **Exogenous Features — OpenWeatherMap & Holiday API**:
  * **Role**: Enriches lag matrices with precipitation volume (mm), visibility (m), and Indian public holiday binary flags.

> **🗣️ Speaker Notes**: We employ a modular, multi-dataset strategy: featuring Chennai micro-mobility hotspots (OMR IT Corridor, Guindy Metro, T. Nagar) as our flagship implementation, while using NYC TLC trip logs as a cross-city benchmark to prove generalizability.


---

## 🖥️ Slide 9: Technology Stack & Full-Stack Architecture

### **End-to-End Production Tech Stack**
* 🧠 **ML & DL Engine**: Python 3.10+, `xgboost`, `lightgbm`, `catboost`, `torch`, `torch_geometric`, `scikit-learn`
* ⚡ **Backend REST API**: **FastAPI** + `uvicorn` (serves `/api/v1/predict` and `/api/v1/clusters` endpoints)
* 💻 **Frontend Fleet Dashboard**: **Next.js 14+ / React 18+**, TailwindCSS, Lucide Icons
* 🗺️ **Geospatial & Charts**: `Leaflet.js` / `Mapbox GL` (spatial cluster heatmaps) & `Recharts` (hourly demand curves)
* 🐳 **Deployment**: Docker, Vercel (Frontend), Render (FastAPI ML Microservice)

> **🗣️ Speaker Notes**: We use a full-stack architecture: Next.js/React for the fleet management UI with interactive Leaflet maps, FastAPI for serving real-time Python model predictions, and XGBoost + PyTorch GNNs for core time-series forecasting.

---

## 🖥️ Slide 10: Expected Output & Deliverables

### **Project Outputs**
1. 📈 **Trained Forecasting Models**: XGBoost and Random Forest model artifacts for $t+1 \dots t+4$ prediction horizons.
2. 📊 **Evaluation Summary & Plots**: WAPE / MAE / RMSE comparison tables and actual vs. predicted demand curves.
3. 📦 **Modular Code Repository**: Clean `src/` modules (`data_loader`, `feature_builder`, `trainer`, `evaluate`).
4. 📄 **Documentation**: Comprehensive academic paper report & presentation deck.

> **🗣️ Speaker Notes**: Expected deliverables include trained model weights, evaluation performance metrics, modular source code, and full project documentation.

---

## 🖥️ Slide 11: Project Timeline & Milestones

| Phase | Milestone | Deliverables |
| :--- | :--- | :--- |
| **Week 1** | Pipeline Framing | Data ingestion script (`src/data_loader.py`) & environment setup. |
| **Week 2** | EDA & Clustering | ACF/PACF statistical analysis & `MiniBatchKMeans` spatial clustering. |
| **Week 3** | Feature Engineering | Lag generation, rolling weather features (`src/feature_builder.py`). |
| **Week 4** | Model Training | XGBoost & Random Forest multi-step training (`src/trainer.py`). |
| **Week 5** | Evaluation & Report | Metric computation (`src/evaluate.py`), tuning, & slide deck finalization. |

> **🗣️ Speaker Notes**: The 5-week execution timeline is structured systematically from data ingestion to spatial clustering, feature engineering, model training, and final evaluation.

---

## 🖥️ Slide 12: References & Data Sources

1. J. Zhang, Y. Zheng, and D. Qi, "Deep Spatiotemporal Residual Networks," *IEEE Trans. Mobile Comput.*, 2021.
2. X. Li, G. Pan, and Z. Wu, "Hybrid Geospatial Clustering & XGBoost," *IEEE Trans. Intell. Transp. Syst.*, 2022.
3. Y. Chen, H. Wang, and L. Sun, "Weather-Aware Bike Sharing Multi-Step Forecasting," *Transp. Res. Part C*, 2023.
4. H. Zhang, W. Wang, and Y. Liu, "Spatiotemporal Graph Neural Networks & Boosted Trees," *IEEE T-ITS*, 2024.
5. X. Wang, L. Chen, and M. Sun, "ADFormer: Aggregation Differential Transformer," *IJCAI*, 2025.
6. R. Sharma, S. Gupta, and K. Patel, "Attention Transformer & Tree Ensemble Framework," *Springer J. Big Data Anal. Transp.*, 2026.
7. P. Singh, "Ola Bike Ride Request Dataset," *Kaggle Datasets*, 2025.
8. Uber Technologies Inc., "Uber Pickups in New York City (GPS Trip Data)," *Kaggle Datasets*, 2023.
9. NYC Taxi & Limousine Commission, "TLC Trip Record Data (FHV Spatiotemporal Demand)," *NYC Open Data*, 2024.
10. OpenWeatherMap, "Historical Weather Data & Meteorological Parameters API," 2025.

> **🗣️ Speaker Notes**: Thank you for your time. I am now open to your questions and feedback.
