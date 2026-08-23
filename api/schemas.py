from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str = "healthy"


class PredictionRequest(BaseModel):
    city: str = Field(..., description="City name or identifier")
    cluster_id: int = Field(..., description="Spatial cluster / zone identifier")
    horizon: int = Field(default=1, description="Forecast horizon in intervals (e.g. 15-min blocks)")


class PredictionResponse(BaseModel):
    city: str
    cluster_id: int
    horizon: int
    predicted_demand: float
