"""
models.py - Pydantic models for FastAPI request/response validation.
Defines schemas for stock input, analysis output, and charts.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum


class RatingEnum(str, Enum):
    """Buy/Sell/Hold ratings."""
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class StockInput(BaseModel):
    """
    Request body for /analyze endpoint.
    """
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock ticker (e.g., AAPL, MSFT)")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL"
            }
        }


class Metric(BaseModel):
    """
    Individual fundamental metric.
    """
    name: str
    value: float
    benchmark: Optional[str] = None  # e.g., "Good: <15"


class AnalysisResponse(BaseModel):
    """
    Core analysis result.
    """
    symbol: str
    score: float = Field(..., ge=0, le=100, description="Overall score (0-100)")
    rating: RatingEnum
    metrics: List[Metric]


class Chart(BaseModel):
    """
    Plotly chart data (JSON string).
    """
    type: str  # e.g., "metrics_bar", "price_trend"
    json: str  # Plotly figure.to_json()
    title: str


class FullAnalysisResponse(BaseModel):
    """
    Complete API response with disclaimer.
    """
    symbol: str
    score: float = Field(..., ge=0, le=100)
    rating: RatingEnum
    metrics: Dict[str, Any]  # Raw yfinance metrics subset
    charts: Dict[str, str]   # {chart_type: plotly_json}
    disclaimer: str = Field(
        ...,
        description="Legal disclaimer",
        example="Educational tool only. Not financial advice."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "score": 82.5,
                "rating": "Buy",
                "metrics": {"forwardPE": 12.5, "debtToEquity": 45},
                "charts": {
                    "metrics_bar": "<plotly_json_here>",
                    "price_trend": "<plotly_json_here>"
                },
                "disclaimer": "Educational tool only. Not financial advice."
            }
        }


class HealthCheckResponse(BaseModel):
    """
    /health endpoint response.
    """
    status: str = "healthy"
    version: str = "1.0.0"
