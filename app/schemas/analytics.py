"""Government analytics Pydantic schemas."""
from typing import Literal, Optional

from pydantic import BaseModel


class AccidentTrend(BaseModel):
    month: str     # e.g. "Jan 2026"
    count: int


class RiskPrediction(BaseModel):
    region: str
    risk_level: Literal["High", "Medium", "Low"]
    predicted_accidents: int


class ForecastPoint(BaseModel):
    period: str
    predicted_accidents: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    risk_level: Optional[Literal["High", "Medium", "Low"]] = None


class ForecastRiskThresholds(BaseModel):
    medium_min: int
    high_min: int
    method: str


class ForecastResponse(BaseModel):
    status: str
    model_name: str
    model_version: Optional[str] = None
    points: list[ForecastPoint]
    message: Optional[str] = None
    risk_thresholds: Optional[ForecastRiskThresholds] = None

class AssociationRule(BaseModel):
    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float


class AssociationRulesResponse(BaseModel):
    status: str
    total_rules: int
    thresholds: dict
    rules: list[AssociationRule]
    message: Optional[str] = None
