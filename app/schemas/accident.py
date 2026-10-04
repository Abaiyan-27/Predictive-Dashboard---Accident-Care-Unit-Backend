"""Accident-related Pydantic schemas — AccidentRecord, InjuryDescription, Treatment, SocioeconomicInfo, HospitalPatient."""
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, model_validator
from uuid import UUID
from datetime import datetime

from app.contracts.severity import SEVERITY_BORDERLINE, SEVERITY_HIGH, SEVERITY_LABELS, SEVERITY_LOW


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

InjuryOrder = Literal[1, 2]  # 1 = primary injury, 2 = secondary injury


class InjuryDescriptionPayload(BaseModel):
    # Omitted orders are taken from list position; send them explicitly so a
    # secondary injury entered without a primary one stays secondary.
    injury_order: Optional[InjuryOrder] = None
    site_of_injury: Optional[str] = None
    type_of_injury: Optional[str] = None


def assign_injury_orders(injuries: List[InjuryDescriptionPayload]) -> List[InjuryDescriptionPayload]:
    explicit = [injury.injury_order for injury in injuries if injury.injury_order is not None]
    if explicit and len(explicit) != len(injuries):
        raise ValueError("Give injury_order for every injury or for none.")
    ordered = [
        injury if injury.injury_order is not None
        else injury.model_copy(update={"injury_order": position})
        for position, injury in enumerate(injuries, start=1)
    ]
    orders = [injury.injury_order for injury in ordered]
    if len(set(orders)) != len(orders):
        raise ValueError("Each injury_order may appear only once.")
    return sorted(ordered, key=lambda injury: injury.injury_order)


class InjuryDescriptionOut(BaseModel):
    injury_id: UUID
    accident_id: Optional[UUID] = None
    injury_order: InjuryOrder
    site_of_injury: Optional[str] = None
    type_of_injury: Optional[str] = None


# ── Treatment ────────────────────────────────────────────────────────────────

class TreatmentCreatePayload(BaseModel):
    accident_id: UUID
    # Taken from the nurse's account; if sent it must be the nurse's own hospital.
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

SeverityLabel = Literal[SEVERITY_LABELS]  # type: ignore[valid-type]


class SeverityPredictionOut(BaseModel):
    # Always one of app/contracts/severity_labels.json. When the model cannot
    # score (too few inputs, missing artifact, inference error) the label is
    # DEFER TO CLINICAL JUDGEMENT and `status` says why.
    prediction: SeverityLabel
    probability: Optional[float] = None    # chance of needing escalated care (0–1)
    completeness: float                    # fraction of the 12 raw inputs recorded (0–1)
    status: str                            # "OK" | "LOW_CONFIDENCE" | "REFUSED" | "UNAVAILABLE" | "ERROR"
    confidence_level: Optional[str] = None # "high" | "low" | None
    probabilities: dict[str, float] = Field(default_factory=dict)
    model_name: str = "severity_triage"
    model_version: Optional[str] = None
    message: Optional[str] = None
    # Model_Prediction row this output was stored as (None for display-only reads).
    prediction_id: Optional[str] = None


# ── Severity decision (nurse) ─────────────────────────────────────────────────

# A clinician's final Severity decision. DEFER TO CLINICAL JUDGEMENT is the
# model's "cannot score" answer, never a decision (Model_Output_Label marks it
# not human-selectable), so it is not accepted here.
HumanSeverityLabel = Literal[SEVERITY_HIGH, SEVERITY_BORDERLINE, SEVERITY_LOW]  # type: ignore[valid-type]


class SeverityDecisionIn(BaseModel):
    # accept   -> final decision = the model's prediction (decision_source MODEL)
    # override -> final decision = final_decision chosen by the nurse (NURSE)
    action: Literal["accept", "override"]
    final_decision: Optional[HumanSeverityLabel] = None
    reason: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _decision_matches_action(self):
        if self.action == "override" and self.final_decision is None:
            raise ValueError("Choose HIGH, BORDERLINE or LOW as your own decision.")
        if self.action == "accept" and self.final_decision is not None:
            raise ValueError("Accepting records the model's prediction; do not send final_decision.")
        return self


# ── Doctor model decision ────────────────────────────────────────────────────

class ModelDecisionIn(BaseModel):
    # accept   -> final decision = the model's prediction (decision_source MODEL);
    #             final_decision may be sent but must equal the prediction
    # override -> final decision = final_decision chosen by the doctor (DOCTOR);
    #             it must be a valid label for the model and differ from the prediction
    action: Literal["accept", "override"]
    final_decision: Optional[str] = Field(default=None, min_length=1, max_length=100)
    reason: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _override_needs_decision(self):
        if self.action == "override" and self.final_decision is None:
            raise ValueError("Choose your own decision to override the model.")
        return self


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

    # Injuries: at most a primary (1) and a secondary (2).
    injuries: List[InjuryDescriptionPayload] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _order_injuries(self):
        self.injuries = assign_injury_orders(self.injuries)
        return self

class UpdateIntakeForm(FullIntakeForm):
    full_name: Optional[str] = None

