"""Accident-related Pydantic schemas — AccidentRecord, InjuryDescription, Treatment, SocioeconomicInfo, HospitalPatient."""
from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from uuid import UUID
from datetime import datetime


# ── Accident Record ──────────────────────────────────────────────────────────

class AccidentRecordCreatePayload(BaseModel):
    patient_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    first_hospital_id: Optional[UUID] = None
    managed_by: Optional[UUID] = None
    vehicle_type: Optional[str] = None
    helmet_worn: Optional[str] = None
    road_type: Optional[str] = None
    speed: Optional[str] = None
    incident_at: Optional[datetime] = None
    road_condition: Optional[str] = None
    visibility: Optional[str] = None
    road_signals_exist: Optional[str] = None
    collision_force_from: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    illicit_drugs: Optional[str] = None
    status: Optional[str] = "intake"
    # ML model feature columns
    mode_of_transport: Optional[str] = None
    time_to_hospital: Optional[str] = None
    collision_with: Optional[str] = None


class AccidentRecordOut(BaseModel):
    accident_id: UUID
    patient_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    first_hospital_id: Optional[UUID] = None
    managed_by: Optional[UUID] = None
    created_by: Optional[UUID] = None
    vehicle_type: Optional[str] = None
    helmet_worn: Optional[str] = None
    road_type: Optional[str] = None
    speed: Optional[str] = None
    incident_at: Optional[datetime] = None
    road_condition: Optional[str] = None
    visibility: Optional[str] = None
    road_signals_exist: Optional[str] = None
    collision_force_from: Optional[str] = None
    alcohol_consumption: Optional[str] = None
    illicit_drugs: Optional[str] = None
    created_at: Optional[datetime] = None
    status: Optional[str] = None
    # ML model feature columns
    mode_of_transport: Optional[str] = None
    time_to_hospital: Optional[str] = None
    collision_with: Optional[str] = None


# ── Injury Description ───────────────────────────────────────────────────────

class InjuryDescriptionPayload(BaseModel):
    site_of_injury: Optional[str] = None
    type_of_injury: Optional[str] = None


class InjuryDescriptionOut(BaseModel):
    injury_id: UUID
    accident_id: Optional[UUID] = None
    site_of_injury: Optional[str] = None
    type_of_injury: Optional[str] = None


# ── Treatment ────────────────────────────────────────────────────────────────

class TreatmentCreatePayload(BaseModel):
    accident_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    ward_number: Optional[str] = None
    reason: Optional[str] = None
    number_of_days_stay: int = Field(ge=0)
    investigation_done: str = Field(min_length=1)
    management_name_1: str = Field(min_length=1)
    management_name_2: Optional[str] = None
    decided_by: Optional[UUID] = None


class TreatmentOut(BaseModel):
    treatment_id: UUID
    accident_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    ward_number: Optional[str] = None
    reason: Optional[str] = None
    number_of_days_stay: Optional[int] = None
    investigation_done: Optional[str] = None
    management_name_1: Optional[str] = None
    management_name_2: Optional[str] = None
    decided_by: Optional[UUID] = None
    logged_by: Optional[UUID] = None
    created_at: Optional[datetime] = None


# ── Socioeconomic Info ───────────────────────────────────────────────────────

class SocioeconomicInfoPayload(BaseModel):
    family_income_before: Optional[str] = None
    family_income_after: Optional[str] = None
    family_current_status: Optional[str] = None
    insurance_claim_type: Optional[str] = None


class SocioeconomicInfoOut(BaseModel):
    patient_id: UUID
    family_income_before: Optional[str] = None
    family_income_after: Optional[str] = None
    family_current_status: Optional[str] = None
    insurance_claim_type: Optional[str] = None


# ── Hospital Patient ─────────────────────────────────────────────────────────

class HospitalPatientOut(BaseModel):
    id: UUID
    accident_id: Optional[UUID] = None
    hospital_id: Optional[UUID] = None
    entered_at: Optional[datetime] = None
    exited_at: Optional[datetime] = None


# ── Severity Prediction (ML model output) ────────────────────────────────────

class SeverityPredictionOut(BaseModel):
    prediction: Optional[str] = None       # "Moderate" | "Severe" | None (if refused)
    probability: Optional[float] = None    # model confidence (0–1)
    completeness: float                    # fraction of features filled (0–1)
    status: str                            # "OK" | "LOW CONFIDENCE" | "REFUSED"
    confidence_level: Optional[str] = None # "high" | "low" | None
    probabilities: dict[str, float] = Field(default_factory=dict)
    model_name: str = "severity"
    model_version: Optional[str] = None
    message: Optional[str] = None


# ── Combined intake form (convenience) ───────────────────────────────────────

class FullIntakeForm(BaseModel):
    """Combined form for nurse intake: creates Patient + AccidentRecord + InjuryDescriptions."""
    # Patient fields
    full_name: str = Field(min_length=1)
    age: str = Field(min_length=1)
    gender: Optional[str] = None

    # Accident fields
    hospital_id: Optional[UUID] = None
    first_hospital_id: Optional[UUID] = None
    vehicle_type: str = Field(min_length=1)
    helmet_worn: Optional[str] = None
    road_type: str = Field(min_length=1)
    speed: str = Field(min_length=1)
    incident_at: datetime
    road_condition: str = Field(min_length=1)
    visibility: str = Field(min_length=1)
    road_signals_exist: Optional[str] = None
    collision_force_from: str = Field(min_length=1)
    alcohol_consumption: str = Field(min_length=1)
    illicit_drugs: Optional[str] = None

    # ML model feature fields
    mode_of_transport: str = Field(min_length=1)
    time_to_hospital: str = Field(min_length=1)
    collision_with: str = Field(min_length=1)

    # Injuries
    injuries: List[InjuryDescriptionPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_primary_injury(self):
        if not self.injuries:
            raise ValueError("Primary injury site and description are required.")
        primary = self.injuries[0]
        if not primary.site_of_injury or not primary.type_of_injury:
            raise ValueError("Primary injury site and description are required.")
        return self
