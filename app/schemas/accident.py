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
    # Severity model raw inputs
    mode_of_travel_during_accident: Optional[str] = None
    passenger_type: Optional[str] = None
    other_injury: Optional[str] = None


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
    # Severity model raw inputs
    mode_of_travel_during_accident: Optional[str] = None
    passenger_type: Optional[str] = None
    other_injury: Optional[str] = None


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
    prediction: Optional[str] = None       # "Low" | "Borderline" | "High" | None (if refused/deferred)
    probability: Optional[float] = None    # model confidence (0–1)
    completeness: float                    # fraction of features filled (0–1)
    status: str                            # "OK" | "LOW_CONFIDENCE" | "REFUSED"
    confidence_level: Optional[str] = None # "high" | "low" | None
    probabilities: dict[str, float] = Field(default_factory=dict)
    model_name: str = "severity"
    model_version: Optional[str] = None
    message: Optional[str] = None


# ── Combined intake form (convenience) ───────────────────────────────────────

class FullIntakeForm(BaseModel):
    """Combined form for nurse intake: creates Patient + AccidentRecord + InjuryDescriptions.

    Most fields are optional to support partial data. The severity model
    can still produce a prediction with missing fields; if too many are
    absent the model returns DEFER TO CLINICAL JUDGEMENT.
    """
    # Patient fields
    full_name: str = Field(min_length=1)
    age: Optional[str] = None
    gender: Optional[str] = None

    # Accident fields
    hospital_id: Optional[UUID] = None
    first_hospital_id: Optional[UUID] = None
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

    # ML model feature fields (shared across models)
    mode_of_transport: Optional[str] = None
    time_to_hospital: Optional[str] = None
    collision_with: Optional[str] = None

    # Severity model raw inputs
    mode_of_travel_during_accident: Optional[str] = None
    passenger_type: Optional[str] = None
    other_injury: Optional[str] = None

    # Injuries
    injuries: List[InjuryDescriptionPayload] = Field(default_factory=list)

class UpdateIntakeForm(FullIntakeForm):
    full_name: Optional[str] = None

