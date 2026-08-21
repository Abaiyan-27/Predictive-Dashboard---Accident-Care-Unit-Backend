"""Hospital Pydantic schemas."""
from typing import Optional
from pydantic import BaseModel
from uuid import UUID


class HospitalOut(BaseModel):
    hospital_id: UUID
    name: str
    type: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None


class CreateHospitalPayload(BaseModel):
    name: str
    type: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None
