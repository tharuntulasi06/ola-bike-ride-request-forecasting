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

## 🖥️ Slide 5: Literature Survey

### **Key Benchmark Studies**
* **Zhang et al. (IEEE TMC, 2021)** — *Deep Spatiotemporal Residual Networks for Crowd Flow*
  * *Key Finding*: Urban mobility prediction requires modeling closeness ($t-1$), daily period ($t-24$), and weekly trend ($t-168$).
* **Li et al. (IEEE T-ITS, 2022)** — *Hybrid Geospatial Clustering & XGBoost for Ride-Hailing*
  * *Key Finding*: Spatial pre-clustering before gradient boosting outperforms global non-spatial time-series models.
* **Chen et al. (Elsevier TR-C, 2023)** — *Weather-Aware Bike Sharing Multi-Step Forecasting*
  * *Key Finding*: Gradient boosting trees handle non-linear weather interactions (rain, temp, humidity) better than standard baseline models.

> **🗣️ Speaker Notes**: Our methodology builds on recent IEEE and Elsevier literature, combining spatial pre-clustering with temporal lag structures and weather interaction features.

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

## 🖥️ Slide 7: Recommended Algorithm

### **Primary Model: XGBoost Regressor**
* **Why XGBoost?**
  * **Tabular Time-Series Dominance**: Superior performance on structured lag and weather features.
  * **Tweedie / Poisson Loss**: Handles zero-inflated demand during off-peak night hours efficiently.
  * **Feature Interaction Handling**: Automatically learns non-linear interactions between weather severity and peak hours.

### **Baseline Benchmark: Random Forest Regressor**
* Non-parametric ensemble baseline to validate XGBoost performance gain.

> **🗣️ Speaker Notes**: We selected XGBoost as our primary regressor because of its proven accuracy on tabular time-series features and support for zero-inflated Tweedie loss functions.

---

## 🖥️ Slide 8: Multi-Dataset Specifications

### **Multi-Source Data Integration Strategy**
* 🚴 **Primary Target Dataset — Ola Bike Ride Request (Kaggle)**:
  * **Volume**: ~17,379 continuous hourly operational records (~2 years).
  * **Features**: `datetime`, `season`, `weather_situation`, `temp`, `humidity`, `windspeed`, `cnt` (Target).
* 📍 **Geospatial GPS Dataset — Uber NYC Spatiotemporal Pickups**:
  * **Volume**: ~4.5M raw trip records with precise `Lat`, `Lon`, `Date/Time` coordinates.
  * **Role**: Evaluates `MiniBatchKMeans` spatial clustering for zone hotspot allocation.
* 🚕 **Cross-City Benchmark — NYC TLC Ride-Hailing Dataset**:
  * **Volume**: ~10M records across 263 discrete spatial zones (`PULocationID`, `DOLocationID`, `fare_amount`).
  * **Role**: Validates model scalability across high-density urban mobility networks.
* 🌤️ **Exogenous Features — OpenWeatherMap & Holiday API**:
  * **Role**: Enriches lag matrices with precipitation volume (mm), visibility (m), and public holiday binary flags.

> **🗣️ Speaker Notes**: We employ a multi-dataset strategy: using the primary Ola dataset for temporal forecasting, Uber NYC raw GPS logs for MiniBatchKMeans spatial clustering, NYC TLC data for cross-city benchmarking, and OpenWeatherMap for exogenous feature enrichment.

---

## 🖥️ Slide 9: Technology Stack

### **Development & ML Environment**
* **Language**: Python 3.10+
* **ML Ensembles**: `xgboost`, `scikit-learn`
* **Data & Time Series Processing**: `pandas`, `numpy`, `statsmodels`
* **Visualization**: `matplotlib`, `seaborn`
* **Serialization & Versioning**: `joblib`, `Git` & GitHub

> **🗣️ Speaker Notes**: The technology stack is built on Python 3.10+ using industry-standard libraries like pandas, statsmodels for ACF analysis, and XGBoost for ensemble learning.

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

1. J. Zhang, Y. Zheng, and D. Qi, "Deep Spatiotemporal Residual Networks for Citywide Crowd Flows Prediction," *IEEE Transactions on Mobile Computing*, vol. 20, no. 12, pp. 3250–3265, 2021.
2. X. Li, G. Pan, and Z. Wu, "Short-Term Ride-Hailing Demand Forecasting: A Hybrid Geospatial Clustering and XGBoost Approach," *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 11204–11215, 2022.
3. Y. Chen, H. Wang, and L. Sun, "Weather-Aware Bike Sharing Demand Forecasting Using Multi-Step Tree Ensemble Methods," *Transportation Research Part C: Emerging Technologies*, vol. 148, p. 104012, 2023.
4. P. Singh, "Ola Bike Ride Request Dataset," *Kaggle Datasets*, 2025.
5. Uber Technologies Inc., "Uber Pickups in New York City (GPS Trip Data)," *Kaggle Datasets / Uber Movement*, 2023.
6. NYC Taxi & Limousine Commission, "TLC Trip Record Data (FHV Spatiotemporal Demand)," *NYC Open Data*, 2024.
7. OpenWeatherMap, "Historical Weather Data & Meteorological Parameters API," 2025.

> **🗣️ Speaker Notes**: Thank you for your time. I am now open to your questions and feedback.
