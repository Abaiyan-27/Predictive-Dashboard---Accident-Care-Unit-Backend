"""Patient Pydantic schemas."""
from typing import Optional
from pydantic import BaseModel, Field
from uuid import UUID


class PatientCreatePayload(BaseModel):
    full_name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    occupation: Optional[str] = None


class PatientOut(BaseModel):
    patient_id: UUID
    full_name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    occupation: Optional[str] = None
