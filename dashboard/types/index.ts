export interface ClusterInfo {
  cluster_id: number;
  landmark_name: string;
  latitude: number;
  longitude: number;
}

export interface ClusterListResponse {
  city: string;
  clusters: ClusterInfo[];
}

export interface PredictionRequest {
  city?: string;
  cluster_id: number;
  horizon?: number;
  temp?: number;
  rain_1h?: number;
}

export interface PredictionResponse {
  city: string;
  cluster_id: number;
  landmark_name: string;
  horizon: number;
  actual_demand?: number;
  predicted_demand: number;
  actual_surge?: number;
  predicted_surge?: number;
}

export interface RebalanceRecommendation {
  origin_cluster_id: number;
  origin_name: string;
  destination_cluster_id: number;
  destination_name: string;
  recommended_transfer_qty: number;
  estimated_transit_time_mins: number;
  revenue_uplift_inr: number;
  priority: "High" | "Medium" | "Low";
}

export interface DbZoneSummary {
  cluster_id?: number;
  spatial_cluster_id?: number;
  total_records?: number;
  total_rides?: number;
  mean_hourly_demand?: number;
  avg_demand_per_hour?: number;
  max_peak_demand?: number;
  peak_hour?: number;
  avg_trip_duration_mins?: number;
  avg_temperature?: number;
}

export interface EvaluationMetrics {
  project?: string;
  city?: string;
  metrics?: {
    MAE: number;
    RMSE: number;
    WAPE: number;
    R2: number;
  };
  horizons?: Array<{
    horizon: string;
    wape: number;
    mae: number;
    rmse: number;
    r2: number;
  }>;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  category: string;
}
