"""Accident Record Pydantic model — mirrors Supabase public.Accident_Record table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class AccidentRecord(BaseModel):
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
    created_at: Optional[datetime] = None
    status: Optional[str] = "intake"
    # ML model feature columns
    mode_of_transport: Optional[str] = None
    time_to_hospital: Optional[str] = None
    collision_with: Optional[str] = None
