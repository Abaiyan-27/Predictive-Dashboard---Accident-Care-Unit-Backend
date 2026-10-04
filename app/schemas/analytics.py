"""Government analytics Pydantic schemas."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, model_validator


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


# ── Per-model statistics (Government model pages) ───────────────────────────

ModelKey = Literal["severity", "stay_duration", "transfer", "discharge"]


class LabelCount(BaseModel):
    label: str
    count: int
    percentage: Optional[float] = None   # None when there is nothing to divide by


class MonthlyLabelCounts(BaseModel):
    month: str
    total: int
    counts: dict[str, int]


class HospitalLabelCounts(BaseModel):
    hospital: str
    total: int
    counts: dict[str, int]


class DecisionSummary(BaseModel):
    total: int
    accepted: int
    overridden: int
    acceptance_rate: Optional[float] = None
    distribution: list[LabelCount]


class ModelVersionUsage(BaseModel):
    contract_version: Optional[str] = None
    artifact_name: Optional[str] = None
    first_seen_at: Optional[datetime] = None
    predictions: int


class ModelSummary(BaseModel):
    """Aggregate-only statistics for one model; no patient-level fields."""

    model_key: ModelKey
    title: str
    labels: list[str]
    decision_labels: list[str]
    decision_role: Literal["nurse", "doctor"]
    total_predictions: int
    episodes_with_prediction: int
    current: list[LabelCount]
    unavailable: int
    decisions: DecisionSummary
    monthly: list[MonthlyLabelCounts]
    by_hospital: list[HospitalLabelCounts]
    versions: list[ModelVersionUsage]
    generated_at: datetime

    @model_validator(mode="after")
    def _labels_follow_the_model_contract(self) -> "ModelSummary":
        from app.services.government_analytics import DECIDABLE_LABELS, MODEL_LABELS

        valid = MODEL_LABELS[self.model_key]
        if tuple(self.labels) != valid or [item.label for item in self.current] != list(valid):
            raise ValueError(f"{self.model_key} outputs must be exactly {list(valid)}")
        if tuple(self.decision_labels) != DECIDABLE_LABELS[self.model_key] or [
            item.label for item in self.decisions.distribution
        ] != list(DECIDABLE_LABELS[self.model_key]):
            raise ValueError(f"{self.model_key} decisions must use {list(DECIDABLE_LABELS[self.model_key])}")
        for group in [*self.monthly, *self.by_hospital]:
            if set(group.counts) != set(valid):
                raise ValueError(f"{self.model_key} counts must use exactly {list(valid)}")
        return self


class GovernmentStatistics(BaseModel):
    total_accidents: int
    total_patients: int
    total_hospitals: int
    total_transfers: int
    pending_transfers: int
    transfers_by_status: dict[str, int]
