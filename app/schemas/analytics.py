"""Government analytics Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel


class AccidentTrend(BaseModel):
    month: str     # e.g. "Jan 2026"
    count: int


class RiskPrediction(BaseModel):
    region: str
    risk_level: Literal["High", "Medium", "Low"]
    predicted_accidents: int
