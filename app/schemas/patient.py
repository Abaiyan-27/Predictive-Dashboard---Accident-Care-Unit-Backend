"""Patient Pydantic schemas."""
from typing import Literal
from pydantic import BaseModel, Field

Gender = Literal["Male", "Female", "Other"]
Severity = Literal["Critical", "Moderate", "Minor"]
Priority = Literal["High", "Medium", "Low"]


class PatientIntakeForm(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    gender: Gender
    collision_type: str = Field(..., min_length=1)
    injury_description: str = Field(..., min_length=1)


class PatientOut(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    age: int
    gender: Gender
    admission_date: str
    collision_type: str
    injury_description: str
    severity: Severity
    priority: Priority
    predicted_los_days: int
    created_by: str
    hospital_id: str


class PrognosisOut(BaseModel):
    expected_recovery_level: str
    partial_recovery_probability: int
    transfer_recommendation: str
