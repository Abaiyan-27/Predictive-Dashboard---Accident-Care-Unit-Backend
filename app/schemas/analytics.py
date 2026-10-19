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
    """One rule from Association_Rule_Mining.ipynb, in its plain-language form."""
    rank: int                  # the notebook's order (by lift)
    rule: str                  # "When ...: in N% of these accidents, ..."
    interpretation: str
    strength: Literal["very strong", "strong", "moderate", "mild"]
    topic: str
    accidents: int             # past accidents the whole pattern covers
    confidence: float
    lift: float
    if_items: list[str]        # e.g. "Helmet Worn=no"
    then_items: list[str]


class AssociationRulesResponse(BaseModel):
    status: str
    total_rules: int
    matched_rules: int
    total_accidents: Optional[int] = None
    thresholds: dict
    stability: dict
    topics: dict[str, int]
    strengths: dict[str, int]
    disclaimer: str
    rules: list[AssociationRule]
    message: Optional[str] = None
