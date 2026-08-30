from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PredictionRequest(BaseModel):
    city: str = Field(default="chennai", example="chennai")
    cluster_id: int = Field(default=0, ge=0, le=5, example=0)
    horizon: int = Field(default=1, ge=1, le=4, example=1)
    temp: Optional[float] = Field(default=30.5, example=30.5)
    rain_1h: Optional[float] = Field(default=0.0, example=0.0)


class PredictionResponse(BaseModel):
    city: str
    cluster_id: int
    landmark_name: str
    horizon: int
    predicted_demand: float
    unit: str = "ride_requests"
    model_version: str = "GBDT-Trio-v1.0"


class ClusterInfo(BaseModel):
    cluster_id: int
    landmark_name: str
    latitude: float
    longitude: float


class ClusterListResponse(BaseModel):
    city: str
    clusters: List[ClusterInfo]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
