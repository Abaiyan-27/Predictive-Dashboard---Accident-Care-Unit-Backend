"""Patient Pydantic model — mirrors Supabase public.Patient table."""
from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class Patient(BaseModel):
    patient_id: UUID
    full_name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    occupation: Optional[str] = None
